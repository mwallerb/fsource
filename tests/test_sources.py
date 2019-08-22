from __future__ import print_function
import sys
import os.path
import io

from fsource import lexer
from fsource import parser

HEREPATH = os.path.abspath(os.path.dirname(__file__))

def parsefile(fname):
    path = os.path.join(HEREPATH, "data", fname)
    mylexer = lexer.lex_buffer(io.open(path, 'r'))

    # first check that lexer works
    tokens = tuple(mylexer)

    # then check that parser works
    tokenstr = parser.TokenStream(tokens, fname=fname)
    ast = parser.compilation_unit(tokenstr, fname)

def test_nastylex():
    parsefile("nastylex.F90")

