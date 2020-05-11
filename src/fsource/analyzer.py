"""
Copyright 2019 Markus Wallerberger
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
from __future__ import print_function
from collections import OrderedDict

import io
import sys
import contextlib

from . import parser
from . import lexer
from . import common


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
    if fallback is None:
        def fallback(tag, *tail):
            raise ValueError("unexpected tag: {}".format(tag))

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
            if sep: out.write(sep)
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


class Node(object):
    def __str__(self):
        writer = SyntaxWriter()
        writer.handle(self)
        return writer.out.getvalue()

    def write_code(self, out):
        raise NotImplementedError("write_code not implemented")


class Ignored(Node):
    """Ignored node of the AST"""
    def __init__(self, *ast):
        super().__init__()
        self.ast = ast

    def write_code(self, out):
        if out._newline:
            out.writeline("$%s"  % self.ast[0])
        else:
            out.write("$%s"  % self.ast[0])

    def get_code(self):
        out = SyntaxWriter()
        self.write_code(out)
        return str(out)


class CompilationUnit(Node):
    """Top node representing one file"""
    def __init__(self, ast_version, fname, *objs):
        super().__init__()
        self.ast_version = ast_version
        self.filename = fname
        self.objs = objs

    def write_code(self, out):
        out.writeline("! FILE %s" % self.filename)
        out.writeline("! AST VERSION %s" % (self.ast_version,))
        out.handle(*self.objs, sep='\n')
        out.writeline("! END FILE %s" % self.filename)


class Module(Node):
    def __init__(self, name, decls, contained):
        super().__init__()
        self.name = name
        self.decls = decls
        self.contained = contained

    def imbue(self, parent):
        parent.modules[self.name] = self

    def write_code(self, out):
        with out.indent("MODULE %s" % self.name):
            out.handle(*self.decls)
        with out.indent("CONTAINS"):
            out.handle(*self.contained, sep="\n")
        out.writeline("END MODULE %s" % self.name)


class Use(Node):
    def __init__(self, modulename, attrs, only, *symbollist):
        super().__init__()
        self.modulename = modulename
        self.attrs = attrs
        self.only = only is not None
        self.symbollist = symbollist

    def imbue(self, parent):
        parent.use[self.modulename] = self

    def write_code(self, out):
        out.write("USE %s" % self.modulename)
        if self.only:
            out.write(", ONLY: ")
        elif self.symbollist:
            out.write(", ")
        out.handle(*self.symbollist, sep=", ")
        out.writeline()


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

    'id':                lambda name: name.lower(),
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
    fname = '../tests/data/simple.f90'
    slexer = lexer.lex_buffer(open(fname))
    ast = parser.compilation_unit(parser.TokenStream(slexer, fname=fname))
    asr = TRANSFORMER(ast)
    print (str(asr), end="")
