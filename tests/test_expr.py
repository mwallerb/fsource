"""
Tests for expression parser

Copyright 2019 Markus Wallerberger.
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
import re
from fsource import expr
from fsource import lexer
from fsource import parser


def get_toks(lexer, text):
    tokens = *lexer.line_tokens(text), (None, len(text), 0, '<$>')
    return parser.TokenStream(tokens)


def simple_grammar():
    return expr.ExprGrammar([
            expr.Literal(2, 'int'),
            expr.Parenthesized(1, '(', ')')
        ], [
            expr.Infix(1, '^', 'right', 'pow'),
        ], [
            expr.Infix(1, '*', 'left', 'mul'),
            expr.Infix(1, '/', 'left', 'div'),
        ], [
            expr.Infix(1, '+', 'left', 'add'),
            expr.Infix(1, '-', 'left', 'sub'),
        ])


def simple_lexer_re():
    return lexer.RegexLexer.create(
        ("OP",  r"[-+*/^()]"),
        ("INT", r"[\d]+"),
        whitespace="[ \t]*"
        )


def test_simple_grammar():
    lexer_re = simple_lexer_re()
    grammar = simple_grammar()

    tokens = get_toks(lexer_re, "3*(41+1)")
    assert grammar.parser(tokens) == \
                ('mul', ('int', '3'), ('add', ('int', '41'), ('int', '1')))
