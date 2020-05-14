"""
Copyright 2019 Markus Wallerberger
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
from __future__ import print_function

import io
import sys
import contextlib

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


class SyntaxWriter:
    """
    IO object that handles indentation.
    """
    def __init__(self, out=None, indenttext="    "):
        if out is None:
            out = io.StringIO()
        self.out = out
        self.indenttext = indenttext
        self.prefix = ""
        self._write = self.out.write
        self._newline = True

    def _handle_newline(self):
        if self._newline:
            self._newline = False
            self._write(self.prefix)

    def write(self, text):
        self._handle_newline()
        self._write(text)

    def writeline(self, line=None):
        self._handle_newline()
        if line:
            self._write(line)
        self._write("\n")
        self._newline = True

    def handle(self, *objs, sep=None):
        if not objs:
            return
        objs[0].write_code(self)
        for obj in objs[1:]:
            if sep: self._write(sep)
            obj.write_code(self)

    @contextlib.contextmanager
    def indent(self, header=None):
        if header is not None:
            self.writeline(header)

        # Add to prefix
        oldprefix = self.prefix
        try:
            self.prefix += self.indenttext
            yield
        finally:
            self.prefix = oldprefix


class CWrapper:
    def __init__(self):
        self.headers = set()
        self.decls = []

    def get(self):
        return "\n".join(self.decls)


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

    def resolve(self, objects, types):
        """Resolves references"""
        raise NotImplementedError("resolve is not implemented")

    def write_code(self, out):
        """Write code of self to SyntaxWriter"""
        raise NotImplementedError("write_code not implemented")

    def get_code(self):
        """Get code of self as string"""
        out = SyntaxWriter()
        self.write_code(out)
        return str(out.out.getvalue())

    __str__ = get_code


class Ignored(Node):
    """Ignored node of the AST"""
    def __init__(self, *ast):
        self.ast = ast

    def imbue(self, parent):
        print("CANNOT IMBUE %s WITH %s" % (parent, self.ast[0]))

    def resolve(self, objects, types):
        print("CANNOT RESOLVE %s" % self.ast[0])

    def write_code(self, out):
        if out._newline:
            out.writeline("$%s"  % self.ast[0])
        else:
            out.write("$%s"  % self.ast[0])


class CompilationUnit(Node):
    """Top node representing one file"""
    def __init__(self, ast_version, fname, *children):
        super().__init__()
        self.ast_version = ast_version
        self.filename = fname
        self.children = children

    def imbue(self):
        for child in self.children: child.imbue(self)

    def resolve(self, objects={}, types={}):
        for child in self.children: child.resolve(objects, types)

    def write_code(self, out):
        out.writeline("! file %s" % self.filename)
        out.writeline("! ast version %s" % ".".join(self.ast_version))
        out.handle(*self.children, sep='\n')
        out.writeline("! end file %s" % self.filename)

    def write_cdecl(self, wrap):
        wrap.decls += ["/* Start declarations for file %s */" % self.filename]
        for obj in self.children:
            obj.write_cdecl(wrap)
        wrap.decls += ["/* End declarations for file %s */" % self.filename]


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

    def resolve(self, objects, types):
        objects = dict(objects)
        types = dict(types)
        for decl in self.decls:
            decl.resolve(objects, types)

    def write_code_header(self, out):
        out.handle(*self.prefixes, sep=" ")
        out.write("subroutine %s(%s)" % (self.name, ", ".join(self.argnames)))
        if self.suffixes:
            out.write(" ")
            out.handle(*self.suffixes, sep=" ")

    def write_code(self, out):
        self.write_code_header(out)
        with out.indent(""):
            out.handle(*self.decls)
        out.writeline("end subroutine %s" % self.name)

    def write_cdecl(self, wrap):
        if self.cname is None:
            wrap.decls += ["/* NOTE: unable to wrap %s. */" % self.name]
            return

        # arg decls
        headers = set()
        args = []
        for arg in self.args:
            arg_str, arg_headers = arg.get_cdecl()
            headers |= arg_headers
            args.append(arg_str)
        args = ", ".join(args)

        wrap.decls += ["void %s(%s);" % (self.cname, args)]
        wrap.headers |= headers


class Intent(Node):
    def __init__(self, in_, out):
        self.in_ = bool(in_)
        self.out = bool(out)

    def imbue(self, parent):
        parent.intent = self.in_, self.out

    def write_code(self, out):
        out.write("intent(%s%s)" % ("in" * self.in_, "out" * self.out))


class BindC(Node):
    def __init__(self, cname):
        self.cname = lexer.parse_string(cname) if cname is not None else None

    def imbue(self, parent):
        if self.cname is None:
            self.cname = parent.name
        parent.cname = self.cname

    def write_code(self, out):
        if self.cname is None:
            out.write("bind(C)")
        else:
            out.write("bind(C, name='%s')" % self.cname)


class EntityDecl:
    def __init__(self, type_, attrs, entity_list):
        self.entities = tuple(Entity(type_, attrs, *entity)
                              for entity in entity_list)

    def imbue(self, parent):
        for entity in self.entities:
            entity.imbue(parent)

    def resolve(self, objects, types):
        for entity in self.entities:
            entity.resolve(objects, types)

    def write_code(self, out):
        out.handle(*self.entities)


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

    def resolve(self, objects, types):
        self.type_.resolve(objects, types)
        objects[self.name] = self

    def write_code(self, out):
        out.handle(self.type_)
        if self.attrs:
            out.write(", ")
            out.handle(*self.attrs, sep=", ")
        out.write(" :: %s" % self.name)
        if self.shape:
            out.handle(self.shape)
        if self.len_:
            out.handle(self.len_)
        if self.init:
            out.handle(self.init)
        out.writeline()

    def get_cdecl(self):
        type_decl, type_header = self.type_.get_cdecl()

        cdecl = "%s%s %s%s" % ("const " * self.c_pointer, type_decl,
                               "*" * self.c_const, self.name)
        return cdecl, type_header


class Opaque:
    def __init__(self, module, name):
        self.module = module
        self.name = name

    @property
    def fqname(self): return "%%" + self.name


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

    def __init__(self):
        self.objects = {tag: Opaque(self, tag) for tag in self.TAGS}
        self.types = {name: Opaque(self, name) for name in self.TYPES}


class Module(Node):
    def __init__(self, name, decls, contained):
        super().__init__()
        self.name = name
        self.decls = decls
        self.contained = contained

    def imbue(self, parent):
        parent.modules[self.name] = self

    def write_code(self, out):
        with out.indent("module %s" % self.name):
            out.handle(*self.decls)
        with out.indent("contains"):
            out.handle(*self.contained, sep="\n")
        out.writeline("end module %s" % self.name)


class Use(Node):
    def __init__(self, modulename, attrs, only, *symbollist):
        super().__init__()
        self.modulename = modulename
        self.attrs = attrs
        self.only = only is not None
        self.symbollist = symbollist

    def imbue(self, parent): pass

    def resolve(self, objects, types):
        # TODO: do something slightly more useful
        if self.modulename != 'iso_c_binding':
            return
        self.ref = IsoCBindingModule()
        objects.update(self.ref.objects)
        types.update(self.ref.types)

    def write_code(self, out):
        out.write("use %s" % self.modulename)
        if self.only:
            out.write(", only: ")
        elif self.symbollist:
            out.write(", ")
        out.handle(*self.symbollist, sep=", ")
        out.writeline()


class Ref(Node):
    def __init__(self, name):
        self.name = name
        self.fqname = name

    def resolve(self, objects, types):
        try:
            self.ref = objects[self.name]
        except KeyError:
            print("DID NOT FIND %s" % self.name)
            self.ref = None
        else:
            self.fqname = self.ref.fqname

    def write_code(self, out):
        out.write(self.fqname)


class IntegerType(Node):
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

    def __init__(self, kind):
        self.kind = kind

    def resolve(self, objects, types):
        if self.kind is not None:
            self.kind.resolve(objects, types)

    def write_code(self, out):
        out.write(self.FTYPE)
        if self.kind is not None:
            out.write("(kind=")
            out.handle(self.kind)
            out.write(")")

    def get_cdecl(self):
        if self.kind is None:
            raise RuntimeError("Cannot wrap")
        ctype, cheaders, _, _ = self.KIND_MAPS[self.kind.fqname]
        return ctype, cheaders


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
    'intent':            Intent,
    'entity_decl':       EntityDecl,
    'entity_attrs':      unpack_sequence,
    'entity_list':       unpack_sequence,
    'entity':            unpack_sequence,

    'integer_type':      IntegerType,
    'kind_sel':          unpack,

    'id':                lambda name: name.lower(),
    'ref':               Ref,
    }

TRANSFORMER = sexpr_transformer(HANDLERS, Ignored)


EXACT_MAPPINGS = (
    # Fortran   ISO_C_BINDING    size  C              Numpy      Ctypes
    # -------------------------------------------------------------------------
    ('logical', 'c_bool',        1,    '_Bool',       'bool_',   'c_bool'),
    ('character', 'c_char',      1,    'char',        'char',    'c_char'),
    ('integer', 'c_int',         None, 'int',         'intc',    'c_int'),
    ('integer', 'c_short',       None, 'short',       'short',   'c_short'),
    ('integer', 'c_long',        None, 'long',        'int_',    'c_long'),
    ('integer', 'c_long_long',   None, 'long long',  'longlong', 'c_longlong'),
    ('integer', 'c_signed_char', 1,    'signed char', 'byte',    'c_byte'),
    ('integer', 'c_size_t',      None, 'ssize_t',     'intp',    'c_ssize_t'),
    ('integer', 'c_int8_t',      1,    'int8_t',      'int8',    'c_int8'),
    ('integer', 'c_int16_t',     2,    'int16_t',     'int16',   'c_int16'),
    ('integer', 'c_int32_t',     4,    'int32_t',     'int32',   'c_int32'),
    ('integer', 'c_int64_t',     8,    'int64_t',     'int64',   'c_int64'),
    ('integer', 'c_intptr_t',    None, 'intptr_t',    'intp',    'c_ssize_t'),
    ('integer', 'c_ptrdiff_t',   None, 'ptrdiff_t',   'intp',    'c_ssize_t'),
    ('real',    'c_float',       4,    'float',       'float32', 'c_float'),
    ('real',    'c_double',      8,    'double',      'float64', 'c_double'),
    ('real',    'c_long_double', None, 'long double', 'longdouble',
                                                               'c_longdouble'),
    ('complex', 'c_float_complex', 8, 'float _Complex', 'complex64',
                                                                   '@c_float'),
    ('complex', 'c_double_complex', 16, 'double _Complex', 'complex128',
                                                                  '@c_double'),
    ('complex', 'c_long_double_complex', None, 'long double _Complex',
                                               'clongdouble', '@c_longdouble'),
    )


if __name__ == '__main__':
    fname = sys.argv[1]
    slexer = lexer.lex_buffer(open(fname))
    ast = parser.compilation_unit(parser.TokenStream(slexer, fname=fname))
    asr = TRANSFORMER(ast)
    asr.imbue()
    asr.resolve()
    print (str(asr), end="", file=sys.stderr)

    decl = CWrapper()
    asr.write_cdecl(decl)
    print (decl.get())
