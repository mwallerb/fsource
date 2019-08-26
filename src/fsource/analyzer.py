"""
Copyright 2019 Markus Wallerberger
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
from __future__ import print_function
from collections import OrderedDict

import io

from . import parser
from . import lexer
from . import common

def sexpr_transformer(branch_map, fallback):
    """Return depth-first transformation of an AST as S-expression."""
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
    @classmethod
    def getstr(cls, obj):
        out = cls()
        obj.write_code(out)
        return str(out)

    def __init__(self, out=None):
        if out is None:
            out = io.StringIO()
        self.out = out
        self.indenttext = ""
        self.level = 0

    def write(self, text):
        self.out.write(text)

    def writeline(self, line):
        self.out.write(self.indenttext + line + "\n")

    def writeindent(self, header=""):
        self.out.write(self.indenttext + header)

    def indent(self, header=None):
        if header is not None:
            self.writeline(header)
        self.level += 1
        self.indenttext = "    " * self.level

    def unindent(self, footer=None):
        if self.level == 0:
            raise ValueError("Unable to unindent")
        self.level -= 1
        self.indenttext = "    " * self.level
        if footer is not None:
            self.writeline(footer)

    def __str__(self):
        return self.out.getvalue()


class Node(object):
    def __str__(self):
        return SyntaxWriter.getstr(self)

    def write_code(self, out):
        raise NotImplementedError("write_code not implemented")


class Ignored(Node):
    """Ignored node of the AST"""
    def __init__(self, *ast):
        super().__init__()
        self.ast = ast

    def write_code(self, out):
        out.writeline("(IGNORED: %s)"  % repr(self.ast))


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
        for obj in self.objs:
            obj.write_code(out)
            out.writeline("")
        out.writeline("! END FILE %s" % self.filename)

# USE something, ONLY:

class Module(Node):
    def __init__(self, name, decls, contained):
        super().__init__()
        self.name = name
        self.decls = decls
        self.contained = contained

    def imbue(self, parent):
        parent.modules[self.name] = self

    def write_code(self, out):
        out.indent("MODULE %s" % self.name)
        for decl in self.decls:
            decl.write_code(out)
        out.unindent()
        out.indent("CONTAINS")
        for obj in self.contained:
            obj.write_code(out)
        out.unindent("END MODULE %s" % self.name)


class Use(Node):
    def __init__(self, modulename, symbollist):
        super().__init__()
        self.modulename = modulename
        self.symbollist = symbollist

    def imbue(self, parent):
        parent.use[self.modulename] = self

    def write_code(self, out):
        out.writeindent("USE %s" % self.modulename)
        self.symbollist.write_code(out)
        out.write("\n")


class Imports:
    def __init__(self, importall, *seq):
        self.importall = importall
        self.seq = seq



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

if __name__ == '__main__':
    fname = '../tests/data/simple.f90'
    slexer = lexer.lex_buffer(open(fname))
    ast = parser.compilation_unit(parser.TokenStream(slexer, fname=fname))
    asr = TRANSFORMER(ast)
    print (str(asr), end="")
