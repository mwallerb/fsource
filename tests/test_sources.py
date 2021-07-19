"""
Tests for Fortran files in data/

Copyright 2019 Markus Wallerberger.
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
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
    ast = parser.compilation_unit(tokenstr)

def test_nastylex():
    parsefile("nastylex.F90")

def test_inplacearr():
    parsefile("inplacearr.f90")

def test_lexfixed():
    parsefile("lexfixed.f")

def test_simple():
    parsefile("simple.f90")

def test_shareddo():
    parsefile("shareddo.f")

def test_unprefixed():
    parsefile("unprefixed.f90")
