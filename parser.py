#!/usr/bin/env python
from __future__ import print_function
#import lexer
import re

token_pat = re.compile("\s*(?:(\d+)|(.))")
def tokenize(program):


def tokenize(program):
    for number, operator in token_pat.findall(program):
        if number:
            yield Literal()
        elif operator == "+":
            yield InfixSymbol("+", 10)
        else:
            raise SyntaxError("unknown operator")
    yield EndOfInput()
    yield EndOfInput()


class Parser:
    def __init__(self, it):
        self.it = iter(it)
        self.lookahead = next(self.it)

    def consume(self):
        token = self.lookahead
        self.lookahead = next(self.it)
        return token

    def get_handler(self, tok_code, token):
        pass

    def expression(self, min_glue=0):
        print ("EXPR", min_glue)
        result = token.head(self)
        while token.glue >= min_glue:
            token = self.advance()
            result = token.tail(self, result)
        return result

class Literal:
    def __init__(self, action=None):
        if action is None:
            action = lambda x: "'%s'" % x
        self.action = action
        self.glue = 10000     # shall not be used

    def head(self, parser):
        print ("LIT", parser.token)
        return self.action(parser.token)

    def tail(self, parser, head):
        raise ValueError("Two literals in a row are forbidden")

class InfixSymbol:
    def __init__(self, symbol, glue, action=None, assoc='left'):
        if action is None:
            action = lambda x, y: "(%s %s %s)" % (symbol, x, y)
        if assoc not in ('left', 'right'):
            raise ValueError("Associativity must be either left or right")
        if glue < 0:
            raise ValueError("Glue must be a non-negative number")

        self.symbol = symbol
        self.glue = glue
        self.subglue = glue + (0 if assoc == 'right' else 1)
        self.action = action

    def head(self, parser):
        raise ValueError("XXX is an infix operator")

    def tail(self, parser, head):
        print (self.symbol)
        right = parser.expression(self.subglue)
        return self.action(head, right)

class EndOfInput:
    def __init__(self):
        self.glue = -1000

    def head(self, parser):
        raise ValueError("End token must not be consumed")

    def tail(self, parser, head):
        raise ValueError("End token must not be consumed")


token_pat = re.compile("\s*(?:(\d+)|(.))")
def tokenize(program):
    for number, operator in token_pat.findall(program):
        if number:
            yield Literal()
        elif operator == "+":
            yield InfixSymbol("+", 10)
        else:
            raise SyntaxError("unknown operator")
    yield EndOfInput()
    yield EndOfInput()

program = "1+2"
print (",".join(map(str, tokenize(program))))
it = tokenize(program)
print (Parser(it).expression())
