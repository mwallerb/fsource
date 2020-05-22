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
import warnings

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
    def union(cls, elems, sep="", ignore_errors=False):
        wraps = tuple(elem.cdecl() for elem in elems)
        return cls(sep.join(w.decl for w in wraps),
                   set().union(*(w.headers for w in wraps)),
                   sum((w.fails for w in wraps), start=())
                   )

    @classmethod
    def fail(cls, name, msg, subfails=()):
        return cls(fails=(name, msg, subfails))

    def __init__(self, decl="", headers=set(), fails=()):
        self.decl = decl
        self.headers = headers
        self.fails = fails

    def get(self):
        if self.fails:
            warnings.warn("FAIL:" + str(self.fails))
        return "".join("#include %s\n" % h for h in self.headers) + self.decl


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
        warnings.warn("CANNOT IMBUE %s WITH %s" % (parent, self.ast[0]))

    def resolve(self, context):
        warnings.warn("CANNOT RESOLVE %s" % self.ast[0])

    def fcode(self):
        return "$%s$" % self.ast[0]


class CompilationUnit(Node):
    """Top node representing one file"""
    def __init__(self, ast_version, fname, *children):
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
        return CWrapper.union(self.children)


class Unspecified(Node):
    def __init__(self, name):
        self.name = name


class Subprogram(Node):
    FTYPE = '@@@'

    def __init__(self, name, prefixes, argnames, suffixes, decls):
        self.name = name
        self.prefixes = prefixes
        self.suffixes = suffixes
        self.argnames = argnames
        self.decls = decls

    def imbue(self, parent):
        self.cname = None
        self.args = [Unspecified(name) for name in self.argnames]
        self.retval = Unspecified(self.name) if self.FTYPE == 'function' else None
        for obj in self.prefixes + self.suffixes + self.decls:
            obj.imbue(self)

    def resolve(self, context):
        subcontext = context.copy()
        for obj in self.prefixes + self.suffixes + self.decls:
            obj.resolve(subcontext)

    def fcode_header(self):
        return "{prefixes}{sep}{tag} {name}({args}) {suffixes}".format(
            prefixes=" ".join(map(fcode, self.prefixes)),
            sep=" " if self.prefixes else "",
            tag=self.FTYPE,
            name=self.name,
            args=", ".join(self.argnames),
            suffixes=" ".join(map(fcode, self.suffixes))
            )

    def fcode(self):
        return "{header}\n{decls}end {tag} {name}\n".format(
            header=self.fcode_header(),
            decls="".join(map(fcode, self.decls)),
            tag=self.FTYPE,
            name=self.name
            )

    def cdecl(self):
        if self.cname is None:
            return CWrapper.fail(self.name, "bind(C) suffix missing")

        # arg decls
        args = CWrapper.union(self.args, sep=", ")
        if self.retval is not None:
            # pylint: disable=no-member
            ret = self.retval.type_.cdecl()
        else:
            ret = CWrapper("void")
        if args.fails or ret.fails:
            return CWrapper.fail(self.name, "failed to wrap arguments",
                                 args.fails + ret.fails)
        return CWrapper(
            "{ret} {name}({args});\n".format(ret=ret.decl, name=self.cname,
                                             args=args.decl),
            ret.headers | args.headers
            )


class Subroutine(Subprogram):
    FTYPE = 'subroutine'

    def __init__(self, name, prefixes, argnames, bindc, decls):
        Subprogram.__init__(self, name, prefixes, argnames,
                            (bindc,) if bindc is not None else (), decls)


class Function(Subprogram):
    FTYPE = 'function'

    def __init__(self, name, prefixes, argnames, suffixes, decls):
        Subprogram.__init__(self, name, prefixes, argnames, suffixes, decls)


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
        self.cname = None if cname is None else cname.value

    def imbue(self, parent):
        if self.cname is None:
            self.cname = parent.name
        parent.cname = self.cname

    def resolve(self, context): pass

    def fcode(self):
        if self.cname is None:
            return "bind(C)"
        else:
            return "bind(C, name='{}')".format(self.cname)


class DimensionAttr(Node):
    def __init__(self, shape):
        self.shape = shape

    def imbue(self, parent):
        if parent.shape is not None:
            raise RuntimeError("duplicate shape")
        parent.shape = self.shape
        parent.attrs = tuple(a for a in parent.attrs if a is not self)


class ResultName(Node):
    def __init__(self, name):
        self.name = name

    def imbue(self, parent):
        parent.retval.name = self.name

    def resolve(self, context): pass

    def fcode(self):
        return "result({})".format(self.name)



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

    def cdecl(self):
        return CWrapper.union(self.entities)


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
        self.cname = None

        if isinstance(parent, DerivedType):
            parent.fields.append(self)
            self.entity_type = 'field'
        elif isinstance(parent, Module):
            parent.modulevars.append(self)
            self.entity_type = 'modulevar'
        elif isinstance(parent, Subprogram):
            # Add self to arguments if applicable
            if self.name in parent.argnames:
                parent.args[parent.argnames.index(self.name)] = self
                self.entity_type = 'argument'
                self.intent = True, True
                self.passby = 'reference'
            elif parent.retval is not None and self.name == parent.retval.name:
                parent.retval = self
                self.entity_type = 'return'
                self.passby = 'value'

        for attr in self.attrs:
            attr.imbue(self)

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
        if self.entity_type == 'modulevar':
            if self.cname is None:
                return CWrapper.fail(self.name,
                                     "bind(C) missing on module variable")
            decl = "extern {const}{type_}{ptr}{name}{shape};\n"
        elif self.entity_type == 'field':
            decl = "  {const}{type_}{ptr}{name}{shape};\n"
        else:
            decl = "{const}{type_}{ptr}{name}{shape}"

        type_ = self.type_.cdecl()
        if type_.fails:
            return CWrapper.fail(self.name, "cannot wrap entity type",
                                 type_.fails)

        type_.decl += "" if type_.decl.endswith("*") else " "
        pointer = (self.len_ is not None or
                   self.passby == 'reference' and self.shape is None)
        const = (self.entity_type == 'parameter' or
                 self.entity_type == 'argument' and not self.intent[1])
        shape = self.shape.cdecl() if self.shape is not None else CWrapper("")
        return CWrapper(decl.format(
                            const="const " * const, type_=type_.decl,
                            ptr="*" * pointer, name=self.name, shape=shape.decl
                            ),
                        type_.headers | shape.headers)

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


class DerivedType(Node):
    def __init__(self, name, attrs, tags, decls, proc):
        self.name = name
        self.attrs = attrs
        self.tags = tags
        self.decls = decls
        self.procs = [] if proc is None else proc

    def imbue(self, parent):
        self.cname = None
        for attr in self.attrs:
            attr.imbue(self)

        self.is_private = False
        self.is_sequence = False
        for tag in self.tags:
            tag.imbue(self)

        self.fields = []
        for decl in self.decls:
            decl.imbue(self)

    def resolve(self, context):
        context.derived_types[self.name] = self
        context = context.copy()
        for decl in self.decls:
            decl.resolve(context)
        for proc in self.procs:
            proc.resolve(context)

    def fcode(self):
        return "type{attrs} :: {name}\n{tags}{decls}end type {name}\n".format(
                    name=self.name,
                    attrs="".join(", " + a.fcode() for a in self.attrs),
                    tags="".join(map(fcode, self.tags)),
                    decls="".join(map(fcode, self.decls))
                    )

    def cdecl(self):
        if self.cname is None:
            return CWrapper.fail(self.name, "bind(C) prefix missing")
        fields = CWrapper.union(self.fields)
        if fields.fails:
            return CWrapper.fail(self.name, "failed to wrap fields", fields.fails)
        return CWrapper("struct {name} {{\n{fields}}};\n".format(
                            name=self.cname, fields=fields.decl),
                        fields.headers)


class Module(Node):
    def __init__(self, name, decls, contained):
        self.name = name
        self.decls = decls
        self.contained = contained

    def imbue(self, _):
        self.modulevars = []
        for obj in self.decls + self.contained:
            obj.imbue(self)

    def resolve(self, context):
        subcontext = context.copy()
        for decl in self.decls + self.contained:
            decl.resolve(subcontext)
        self.context = subcontext

        # Module only becomes available after it's declared.
        context.modules[self.name] = self

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

    def cdecl(self):
        return CWrapper.union(self.decls + self.contained)



class Use(Node):
    def __init__(self, modulename, attrs, only, *symbollist):
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
            warnings.warn("Cannot find module " + self.modulename)
            self.ref = None
            return

        context.update(self.ref.context, self.filter_names)

    def fcode(self):
        return "use {name}{sep}{only}{symlist}\n".format(
            name=self.modulename,
            sep=", " if self.only or self.symbollist else "",
            only="only: " if self.only else "",
            symlist=", ".join(map(fcode, self.symbollist)))

    def cdecl(self):
        # TODO: Maybe uses pull in dependencies?
        return CWrapper()


class Ref(Node):
    def __init__(self, name):
        self.name = name
        self.ref = None

    def resolve(self, context):
        try:
            self.ref = context.entities[self.name]
        except KeyError:
            warnings.warn("DID NOT FIND %s" % self.name)

    def fcode(self):
        if self.ref is None:
            return self.name
        return self.ref.fcode()


class PrimitiveType(Node):
    FTYPE = '$$$null_type$$$'
    KIND_MAPS = {}

    def __init__(self, kind):
        self.kind = kind

    def imbue(self, parent):
        # A type may imbue a function as its return type.
        if isinstance(parent, Function):
            retval = Entity(self, (), parent.name, None, None, None)
            retval.imbue(parent)

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
            return CWrapper.fail(self.fcode(), "kind has no iso_c_binding")
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


class Literal(Node):
    @classmethod
    def parse(cls, token): raise NotImplementedError()

    def __init__(self, token):
        self.token = token
        self.value = self.parse(token)
        self.kind = None

    def imbue(self, parent): pass

    def resolve(self, context): pass

    def fcode(self): return self.token


class IntLiteral(Literal):
    @classmethod
    def parse(cls, token):
        return int(token)


class StringLiteral(Literal):
    @classmethod
    def parse(cls, token):
        return lexer.parse_string(token)


class Dim(Node):
    def __init__(self, lower, upper):
        if lower is None:
            lower = IntLiteral("1")
        self.lower = lower
        self.upper = upper

    def imbue(self, parent): pass

    def resolve(self, context):
        self.lower.resolve(context)


class ExplicitDim(Dim):
    def resolve(self, context):
        self.lower.resolve(context)
        self.upper.resolve(context)

    def imbue(self, parent):
        if parent.shape_type == 'implied':
            raise RuntimeError("Cannot follow implied with explicit dim")

    def fcode(self):
        return "{}:{}".format(self.lower.fcode(), self.upper.fcode())

    def cdecl(self):
        try:
            size = self.upper.value - self.lower.value
        except AttributeError:
            return CWrapper.fail(self.fcode(), "Shape is not constant")
        return CWrapper("[{}]".format(size))


class ImpliedDim(Dim):
    def imbue(self, parent):
        if parent.shape_type != 'explicit':
            raise RuntimeError("Cannot follow non-explicit with implied dim")
        parent.shape_type = 'implied'

    def fcode(self):
        return "{}:*".format(self.lower.fcode())

    def cdecl(self):
        return CWrapper("[]")


class DeferredDim(Dim):
    def imbue(self, parent):
        if parent.shape_type == 'implied':
            raise RuntimeError("Cannot follow implied with explicit dim")
        parent.shape_type = 'deferred'

    def fcode(self):
        return "{}:".format(self.lower.fcode())

    def cdecl(self):
        # Deferred dims cause variables to carry its dimension parameters with
        # them in a non-bind(C) way
        return CWrapper.fail(self.fcode(), "unable to wrap deferred dimension")


class Shape(Node):
    def __init__(self, *dims):
        self.dims = dims

    def imbue(self):
        self.shape_type = 'explicit'
        for dim in self.dims:
            dim.imbue(self)

    def resolve(self, context):
        for dim in self.dims:
            dim.resolve(context)

    def fcode(self):
        return "({})".format(",".join(map(fcode, self.dims)))

    def cdecl(self):
        return CWrapper.union(self.dims)


class DerivedTypeRef(Node):
    def __init__(self, name):
        self.name = name

    def imbue(self, parent):
        # A type may imbue a function as its return type.
        # TODO: duplication with PrimitiveType, merge? Also, somewhat hacky
        if isinstance(parent, Function):
            retval = Entity(self, (), parent.name, None, None, None)
            retval.imbue(parent)

    def resolve(self, context):
        try:
            self.ref = context.derived_types[self.name]
        except KeyError:
            warnings.warn("DID NOT FIND %s" % self.name)
            self.ref = None

    def fcode(self):
        if self.ref is not None:
            return self.ref.fcode()
        return "type({})".format(self.name)

    def cdecl(self):
        if self.ref is None:
            return CWrapper.fail(self.name, "type declaration not found")
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

    'type_decl':         DerivedType,
    'type_attrs':        unpack_sequence,
    'type_tags':         unpack_sequence,
    'component_block':   unpack_sequence,

    'subroutine_decl':   Subroutine,
    'arg_list':          unpack_sequence,
    'sub_prefix_list':   unpack_sequence,
    'function_decl':     Function,
    'func_prefix_list':  unpack_sequence,
    'func_suffix_list':  unpack_sequence,
    'bind_c':            BindC,
    'result':            ResultName,

    'entity_decl':       EntityDecl,
    'entity_attrs':      unpack_sequence,
    'entity_list':       unpack_sequence,
    'entity':            unpack_sequence,

    'intent':            Intent,
    'value':             PassByValue,
    'dimension':         DimensionAttr,

    'shape':             Shape,
    'explicit_dim':      ExplicitDim,
    'implied_dim':       ImpliedDim,
    'deferred_dim':      DeferredDim,

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
    'string':            StringLiteral,
    }

TRANSFORMER = sexpr_transformer(HANDLERS, Ignored)
