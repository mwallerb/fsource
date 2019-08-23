"""
Copyright 2019 Markus Wallerberger
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
from __future__ import print_function
from collections import OrderedDict

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


class Ignored:
    """Ignored node of the AST"""
    def __init__(self, *ast):
        self.ast = ast

    def __str__(self): return self.get_code()

    def get_code(self, ls="\n", endls="\n"):
        return " &%s!IGNORED: %s%s" % (ls, repr(self.ast), ls)


class CompilationUnit:
    """Top node representing one file"""
    def __init__(self, ast_version, fname, *objs):
        self.ast_version = ast_version
        self.filename = fname
        self.objs = objs

    def get_code(self, ls="\n", endls="\n"):
        s = "! FILE %s%s" % (self.filename, ls)
        s += "! AST VERSION %s%s" % (self.ast_version, ls)
        for obj in self.objs:
            s += obj.get_code(ls)
        s += "! END FILE %s%s" % (self.filename, endls)
        return s


def unpack(arg):
    """Unpack a single argument as-is"""
    return arg

def unpack_sequence(fn=None):
    """Return sequence of itmes as a tuple"""
    if fn is None:
        def unpack_sequence_fn(*items): return items
    else:
        def unpack_sequence_fn(*items): return tuple(map(fn, items))

    return unpack_sequence_fn


HANDLERS = {
    'compilation_unit': CompilationUnit,
    'filename':         unpack,
    'ast_version':      unpack_sequence(),
    }

TRANSFORMER = sexpr_transformer(HANDLERS, Ignored)

if __name__ == '__main__':
    fname = '../tests/data/lexfixed.f'
    slexer = lexer.lex_buffer(open(fname))
    ast = parser.compilation_unit(parser.TokenStream(slexer, fname=fname))
    asr = TRANSFORMER(ast)
    print (asr.get_code())
