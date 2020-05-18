"""
Copyright 2019 Markus Wallerberger
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
from __future__ import print_function

import io
import sys
import contextlib
import textwrap

from . import parser
from . import lexer
from . import common

if sys.version_info >= (3, 7):
    OrderedDict = dict
else:
    from collections import OrderedDict


def sexpr_transformer(branch_map, fallback=None):
    """
    Return depth-first transformation of an AST as S-expression.

    An S-expression is either a branch or a leaf.  A leaf is an arbitrary
    string or `None`.  A branch is a tuple `(tag, *tail)`, where the `tag` is
    a string and each item of `tail` is again an S-expression.

    Construct and return a transformer which does the following: for a
    branch, we look up the tag in `branch_map`.  If it is found, the
    corresponding entry is called with tag and tail as its arguments, but we
    first run the tail through the transformer.  Any leaf is simply returned
    as-is.

    If the tag is not found in branch_map, we instead call `fallback`. In this
    case, the arguments are not processed, which allows pruning subtrees not
    interesting to the consumer.
    """
    def default_fallback(tag, *tail):
        raise ValueError("unexpected tag: {}".format(tag))
    if fallback is None:
        fallback = default_fallback

    def transformer(ast):
        if isinstance(ast, tuple):
            # Branch node
            node_type = ast[0]
            try:
                handler = branch_map[node_type]
            except KeyError:
                return fallback(*ast)
            else:
                return handler(*map(transformer, ast[1:]))
        else:
            # Leaf node
            return ast

    return transformer


class CWrapper:
    @classmethod
    def union(cls, *elems, sep=" "):
        wraps = tuple(elem.cdecl() for elem in elems)
        return cls(sep.join(w.decl for w in wraps),
                   set().union(*(w.headers for w in wraps))
                   )

    def __init__(self, decl="", headers=set()):
        self.decl = decl
        self.headers = headers

    def __str__(self):
        return "".join("#include %s\n" % h for h in self.headers) + self.decl


class PyWrapper:
    def __init__(self, annotate="", wrap=""):
        self.annotate = annotate
        self.wrap = wrap


class Context:
    """Current context"""
    def __init__(self, entities={}, derived_types={}, modules={}):
        self.entities = entities
        self.derived_types = derived_types
        self.modules = modules

    def copy(self):
        return Context(dict(self.entities),
                       dict(self.derived_types),
                       dict(self.modules))

    def update(self, other, filter=None):
        if filter is None:
            filter = lambda other_dict: other_dict

        self.entities.update(filter(other.entities))
        self.derived_types.update(filter(other.derived_types))
        self.modules.update(filter(other.modules))

    def get_module(self, name):
        try:
            return self.intrinsic_modules[name]
        except KeyError:
            return self.modules[name]

    @property
    def intrinsic_modules(self):
        return {'iso_c_binding': IsoCBindingModule()}


def fcode(obj):
    return obj.fcode()


class Node(object):
    def imbue(self, parent):
        """
        Imbues `parent` with information from `self`.

        When the objects are created, they only have information about their
        children.  `imbue` is called *after* the complete hierarchy is
        established to make children aware of their parents, allowing, e.g.,
        child nodes to change attributes of the parents.
        """
        raise NotImplementedError("imbue is not implemented")

    def resolve(self, context):
        """
        Resolves names in `self` given a current name `context`.

        This function is called after `imbue()` on each node in the order of
        appearence in the tree.  The current node takes the current `context`,
        looking up references in context and augmenting it as necessary.
        """
        raise NotImplementedError("resolve is not implemented")

    def fcode(self):
        """Get code of self as string"""
        raise NotImplementedError("want code")


class Ignored(Node):
    """Ignored node of the AST"""
    def __init__(self, *ast):
        self.ast = ast

    def imbue(self, parent):
        print("CANNOT IMBUE %s WITH %s" % (parent, self.ast[0]))

    def resolve(self, context):
        print("CANNOT RESOLVE %s" % self.ast[0])

    def fcode(self):
        return "$%s$" % self.ast[0]


class CompilationUnit(Node):
    """Top node representing one file"""
    def __init__(self, ast_version, fname, *children):
        super().__init__()
        self.ast_version = ast_version
        self.filename = fname
        self.children = children

    def imbue(self):
        for child in self.children: child.imbue(self)

    def resolve(self, context):
        for child in self.children: child.resolve(context)

    def fcode(self):
        return textwrap.dedent("""\
            ! file {file}
            ! ast version {astv}
            {decls}! end file {file}
            """).format(file=self.filename,
                        astv=str(self.ast_version),
                        decls="\n".join(map(fcode, self.children))
                        )

    def cdecl(self):
        return CWrapper.union(*self.children)


class Unspecified(Node):
    def __init__(self, name):
        self.name = name


class Subroutine(Node):
    def __init__(self, name, prefixes, argnames, bindc, decls):
        self.name = name
        self.prefixes = prefixes
        self.suffixes = (bindc,) if bindc is not None else ()
        self.argnames = argnames
        self.returns = None
        self.decls = decls

    def imbue(self, parent):
        self.cname = None
        self.args = [Unspecified(name) for name in self.argnames]
        for attr in self.prefixes + self.suffixes:
            attr.imbue(self)
        for decl in self.decls:
            decl.imbue(self)

    def resolve(self, context):
        subcontext = context.copy()
        for decl in self.decls:
            decl.resolve(subcontext)

    def fcode_header(self):
        return "{prefixes}{sep}subroutine {name}({args}) {suffixes}".format(
            prefixes=" ".join(map(fcode, self.prefixes)),
            sep=" " if self.prefixes else "",
            name=self.name,
            args=", ".join(self.argnames),
            suffixes=" ".join(map(fcode, self.suffixes))
            )

    def fcode(self):
        return "{header}\n{decls}end subroutine {name}\n".format(
            header=self.fcode_header(),
            decls="".join(map(fcode, self.decls)),
            name=self.name
            )

    def cdecl(self):
        if self.cname is None:
            print("Unable to wrap", self.cname)
            return CWrapper()

        # arg decls
        args = CWrapper.union(*self.args, sep=", ")
        ret = CWrapper("void");
        return CWrapper(
            "{ret} {name}({args});\n".format(ret=ret.decl, name=self.cname,
                                             args=args.decl),
            ret.headers | args.headers
            )


class Intent(Node):
    def __init__(self, in_, out):
        self.in_ = bool(in_)
        self.out = bool(out)

    def imbue(self, parent):
        parent.intent = self.in_, self.out

    def fcode(self):
        return "intent({}{})".format("in" * self.in_, "out" * self.out)


class BindC(Node):
    def __init__(self, cname):
        self.cname = lexer.parse_string(cname) if cname is not None else None

    def imbue(self, parent):
        if self.cname is None:
            self.cname = parent.name
        parent.cname = self.cname

    def fcode(self):
        if self.cname is None:
            return "bind(C)"
        else:
            return "bind(C, name='{}')".format(self.cname)


class EntityDecl:
    def __init__(self, type_, attrs, entity_list):
        self.entities = tuple(Entity(type_, attrs, *entity)
                              for entity in entity_list)

    def imbue(self, parent):
        for entity in self.entities:
            entity.imbue(parent)

    def resolve(self, context):
        for entity in self.entities:
            entity.resolve(context)

    def fcode(self):
        return "".join(map(fcode, self.entities))


class Entity(Node):
    def __init__(self, type_, attrs, name, shape, len_, init):
        self.type_ = type_
        self.attrs = attrs
        self.name = name
        self.shape = shape
        self.len_ = len_
        self.init = init

    def imbue(self, parent):
        # Add self to arguments
        self.entity_type = 'variable'
        self.intent = None
        self.passby = None

        # Add self to arguments if applicable
        try:
            index = getattr(parent, "argnames", []).index(self.name)
        except ValueError:
            pass
        else:
            parent.args[index] = self
            self.entity_type = 'argument'
            self.intent = True, True
            self.passby = 'reference'

        for attr in self.attrs:
            attr.imbue(self)

        self.c_pointer = (self.shape is not None or self.len_ is not None or
                          self.passby == 'reference')
        self.c_const = (self.entity_type == 'parameter' or
                        self.entity_type == 'argument' and not self.intent[1])

    def resolve(self, context):
        self.type_.resolve(context)
        context.entities[self.name] = self

    def fcode(self):
        return "{type}{attrs} :: {name}{shape}{len} {init}\n".format(
            type=fcode(self.type_),
            attrs="".join(", " + fcode(a) for a in self.attrs),
            name=self.name,
            shape=fcode(self.shape) if self.shape is not None else "",
            len=fcode(self.len_) if self.len_ is not None else "",
            init=fcode(self.init) if self.len_ is not None else ""
            )

    def cdecl(self):
        type_ = self.type_.cdecl()
        return CWrapper("{const}{type_} {ptr}{name}".format(
                            const="const " * self.c_const, type_=type_.decl,
                            ptr="*" * self.c_pointer, name=self.name),
                        type_.headers)


class PassByValue(Node):
    def __init__(self): pass

    def imbue(self, parent): parent.passby = 'value'

    def resolve(self): pass

    def fcode(self): return "value"


class Opaque:
    def __init__(self, module, name):
        self.module = module
        self.name = name

    def fcode(self): return "%%" + self.name


class CPtrType:
    def __init__(self, module, name):
        self.module = module
        self.name = name

    def fcode(self):
        return "type(%%{})".format(self.name)

    def cdecl(self):
        return CWrapper("void *")


class IsoCBindingModule:
    NAME = "iso_c_binding"
    TAGS = (
        'c_int', 'c_short', 'c_long', 'c_long_long', 'c_signed_char',
        'c_size_t', 'c_int8_t', 'c_int16_t', 'c_int32_t', 'c_int64_t',
        'c_int_least8_t', 'c_int_least16_t', 'c_int_least32_t',
        'c_int_least64_t', 'c_int_fast8_t', 'c_int_fast16_t', 'c_int_fast32_t',
        'c_int_fast64_t', 'c_intmax_t', 'c_intptr_t', 'c_ptrdiff_t', 'c_float',
        'c_double', 'c_long_double', 'c_float128', 'c_float_complex',
        'c_double_complex', 'c_long_double_complex', 'c_bool', 'c_char'
        )
    TYPES = 'c_ptr', 'c_funptr'

    @property
    def name(self): return self.NAME

    @property
    def modtype(self): return "intrinsic"

    @property
    def context(self):
        return Context(
            entities={tag: Opaque(self, tag) for tag in self.TAGS},
            derived_types={name: CPtrType(self, name) for name in self.TYPES}
            )


class Module(Node):
    def __init__(self, name, decls, contained):
        super().__init__()
        self.name = name
        self.decls = decls
        self.contained = contained

    def imbue(self, parent):
        parent.modules[self.name] = self

    def fcode(self):
        return textwrap.dedent("""\
            module {name}
            {decls}
            contains
            {contained}
            end module {name}
            """).format(name=self.name,
                        decls="".join(map(fcode, self.decls)),
                        contained="\n".join(map(fcode, self.contained))
                        )


class Use(Node):
    def __init__(self, modulename, attrs, only, *symbollist):
        super().__init__()
        self.modulename = modulename
        self.attrs = attrs
        self.only = only is not None
        self.symbollist = symbollist

    def imbue(self, parent): pass

    def filter_names(self, names):
        # TODO: handle renames and only
        return names

    def resolve(self, context):
        try:
            self.ref = context.get_module(self.modulename)
        except KeyError:
            print("Cannot find module")
            self.ref = None
            return

        context.update(self.ref.context, self.filter_names)

    def fcode(self):
        return "use {name}{sep}{only}{symlist}\n".format(
            name=self.modulename,
            sep=", " if self.only or self.symbollist else "",
            only="only: " if self.only else "",
            symlist=", ".join(map(fcode, self.symbollist)))


class Ref(Node):
    def __init__(self, name):
        self.name = name

    def resolve(self, context):
        try:
            self.ref = context.entities[self.name]
        except KeyError:
            print("DID NOT FIND %s" % self.name)
            self.ref = None

    def fcode(self):
        if self.ref is None:
            return self.name
        return self.ref.fcode()


class PrimitiveType(Node):
    FTYPE = '$$$null_type$$$'
    KIND_MAPS = {}

    def __init__(self, kind):
        self.kind = kind

    def resolve(self, context):
        if self.kind is not None:
            self.kind.resolve(context)

    def fcode(self):
        if self.kind is None:
            return self.FTYPE
        return "{ftype}(kind={kind})".format(ftype=self.FTYPE,
                                             kind=fcode(self.kind))

    def cdecl(self):
        if self.kind is None:
            raise RuntimeError("Cannot wrap")
        ctype, cheaders = self.KIND_MAPS[self.kind.fcode()][:2]
        return CWrapper(ctype, cheaders)


class IntegerType(PrimitiveType):
    FTYPE = 'integer'

    _STDDEF = {'<stddef.h>'}
    _STDINT = {'<stdint.h>'}
    _NONE = set()
    KIND_MAPS = {
        '%%c_int':         ('int',         _NONE,   'intc',     'c_int'),
        '%%c_short':       ('short',       _NONE,   'short',    'c_short'),
        '%%c_long':        ('long',        _NONE,   'int_',     'c_long'),
        '%%c_long_long':   ('long long',   _NONE,   'longlong', 'c_longlong'),
        '%%c_signed_char': ('signed char', _NONE,   'byte',     'c_byte'),
        '%%c_size_t':      ('size_t',      _STDDEF, 'intp',     'c_ssize_t'),
        '%%c_int8_t':      ('int8_t',      _STDINT, 'int8',     'c_int8'),
        '%%c_int16_t':     ('int16_t',     _STDINT, 'int16',    'c_int16'),
        '%%c_int32_t':     ('int32_t',     _STDINT, 'int32',    'c_int32'),
        '%%c_int64_t':     ('int64_t',     _STDINT, 'int64',    'c_int64'),
        '%%c_intptr_t':    ('intptr_t',    _STDINT, 'intp',     'c_ssize_t'),
        '%%c_ptrdiff_t':   ('ptrdiff_t',   _STDINT, 'intp',     'c_ssize_t'),
        }
    KIND_MAPS.update({
        '1': KIND_MAPS['%%c_int8_t'],
        '2': KIND_MAPS['%%c_int16_t'],
        '4': KIND_MAPS['%%c_int32_t'],
        '8': KIND_MAPS['%%c_int64_t']
        })


class RealType(PrimitiveType):
    FTYPE = 'real'

    _NONE = set()
    KIND_MAPS = {
        '%%c_float':       ('float',       _NONE, 'float32', 'c_float'),
        '%%c_double':      ('double',      _NONE, 'float64', 'c_double'),
        '%%c_long_double': ('long double', _NONE, 'longdouble', 'c_longdouble')
        }
    KIND_MAPS.update({
        '4': KIND_MAPS['%%c_float'],
        '8': KIND_MAPS['%%c_double']
        })


class ComplexType(PrimitiveType):
    FTYPE = 'complex'

    _NONE = set()
    KIND_MAPS = {
        '%%c_float_complex': ('float _Complex', _NONE, 'complex64', 'c_float'),
        '%%c_double_complex': ('double _Complex', _NONE, 'complex128', 'c_double'),
        '%%c_long_double_complex':
                ('long double _Complex', _NONE, 'clongdouble', 'c_longdouble'),
        }
    KIND_MAPS.update({
        '8': KIND_MAPS['%%c_float_complex'],
        '16': KIND_MAPS['%%c_double_complex']
        })


class LogicalType(PrimitiveType):
    FTYPE = 'logical'
    KIND_MAPS = {
        '%%c_bool': ('_Bool', set(), 'bool_', 'c_bool'),
        }
    KIND_MAPS['1'] = KIND_MAPS['%%c_bool']

    def cdecl(self):
        if self.kind is None:
            return CWrapper("_Bool")
        return PrimitiveType.cdecl(self)


class CharacterType:
    KIND_MAPS = {
        '%%c_char': ('char', set(), 'char', 'c_char'),
        }
    KIND_MAPS['1'] = KIND_MAPS['%%c_char']

    def __init__(self, char_sel):
        if char_sel is None:
            char_sel = None, None
        len_, kind = char_sel
        self.len_ = len_
        self.kind = kind

    def imbue(self, parent):
        # TODO: imbue length onto entities
        pass

    def resolve(self, context):
        if self.len_ is not None:
            self.len_.resolve(context)
        if self.kind is not None:
            self.kind.resolve(context)

    def fcode(self):
        if self.kind is None and self.len_ is None:
            return "character"
        return "character({})".format(
            ", ".join("{}={}".format(k, v if isinstance(v,str) else v.fcode())
                      for k, v in {'len':self.len_, 'kind':self.kind}.items()
                      if v is not None))

    def cdecl(self):
        if self.kind is None:
            return CWrapper("char")
        ctype, cheaders = self.KIND_MAPS[self.kind.fcode()][:2]
        return CWrapper(ctype, cheaders)


class IntLiteral(Node):
    def __init__(self, token):
        self.token = token
        self.value = int(token)
        self.kind = None

    def imbue(self, parent): pass

    def resolve(self, context): pass

    def fcode(self): return self.token


class DerivedTypeRef(Node):
    def __init__(self, name):
        self.name = name

    def resolve(self, context):
        try:
            self.ref = context.derived_types[self.name]
        except KeyError:
            print("DID NOT FIND %s" % self.name)
            self.ref = None

    def fcode(self):
        if self.ref is not None:
            return self.ref.fcode()
        return "type({})".format(self.name)

    def cdecl(self):
        return self.ref.cdecl()


def unpack(arg):
    """Unpack a single argument as-is"""
    return arg


def unpack_sequence(*items):
    """Return sequence of itmes as a tuple"""
    return items


HANDLERS = {
    'compilation_unit':  CompilationUnit,
    'filename':          unpack,
    'ast_version':       unpack_sequence,

    'module_decl':       Module,
    'declaration_block': unpack_sequence,
    'contained_block':   unpack_sequence,
    'use_stmt':          Use,

    'subroutine_decl':   Subroutine,
    'arg_list':          unpack_sequence,
    'sub_prefix_list':   unpack_sequence,
    'bind_c':            BindC,

    'entity_decl':       EntityDecl,
    'entity_attrs':      unpack_sequence,
    'entity_list':       unpack_sequence,
    'entity':            unpack_sequence,

    'intent':            Intent,
    'value':             PassByValue,

    'integer_type':      IntegerType,
    'real_type':         RealType,
    'complex_type':      ComplexType,
    'character_type':    CharacterType,
    'logical_type':      LogicalType,
    'derived_type':      DerivedTypeRef,
    'kind_sel':          unpack,
    'char_sel':          unpack_sequence,

    'id':                lambda name: name.lower(),
    'ref':               Ref,
    'int':               IntLiteral,
    }

TRANSFORMER = sexpr_transformer(HANDLERS, Ignored)


if __name__ == '__main__':
    fname = sys.argv[1]
    slexer = lexer.lex_buffer(open(fname))
    ast = parser.compilation_unit(parser.TokenStream(slexer, fname=fname))
    asr = TRANSFORMER(ast)
    asr.imbue()
    asr.resolve(Context())
    print (asr.fcode(), end="", file=sys.stderr)
    print ()
    print (str(asr.cdecl()))
