#!/usr/bin/env python
from __future__ import print_function
import lexer
import re


class TokenStream:
    def __init__(self, lexer):
        self.lexer = iter(lexer)
        self.cat = None
        self.token = None
        self.advance()

    def advance(self):
        token = self.token
        try:
            self.cat, self.token = next(self.lexer)
            print ("ADV", self.cat, self.token)
        except StopIteration:
            self.cat, self.token = 0, '<$>'
        return token


class LiteralHandler:
    def __init__(self, action=None):
        if action is None:
            action = lambda x: "'%s'" % x

        self.action = action
        self.glue = 10000     # shall not be used

    def head(self, tokens):
        print ("LIT", tokens.token)
        return self.action(tokens.advance())

    def tail(self, tokens, head_result):
        raise ValueError("Two literals in a row are forbidden")


class InfixHandler:
    def __init__(self, parser, symbol, glue, action=None, assoc='left'):
        if action is None:
            action = lambda x, y: "(%s %s %s)" % (symbol, x, y)
        if assoc not in ('left', 'right'):
            raise ValueError("Associativity must be either left or right")
        if glue < 0:
            raise ValueError("Glue must be a non-negative number")

        self.parser = parser
        self.symbol = symbol
        self.glue = glue
        self.subglue = glue + (0 if assoc == 'right' else 1)
        self.action = action

    def head(self, tokens):
        raise ValueError("XXX is an infix operator")

    def tail(self, tokens, head_result):
        print (self.symbol)
        tokens.advance()
        tail_result = self.parser.expression(tokens, self.subglue)
        return self.action(head_result, tail_result)


class EndOfInputHandler:
    def __init__(self):
        self.glue = -1000

    def head(self, tokens):
        raise ValueError("End token must not be consumed")

    def tail(self, tokens, head_result):
        raise ValueError("End token must not be consumed")


class Parser:
    def __init__(self):
        self.handlers = (
                EndOfInputHandler(),
                LiteralHandler(),
                InfixHandler(self, '+', 10),
                InfixHandler(self, '**', 30, assoc='right'),
                InfixHandler(self, '*', 20),
                )

    def expression(self, tokens, min_glue=0):
        print ("EXPR", min_glue)
        handler = self.handlers[tokens.cat]
        result = handler.head(tokens)

        handler = self.handlers[tokens.cat]
        while handler.glue >= min_glue:
            result = handler.tail(tokens, result)
            handler = self.handlers[tokens.cat]

        return result


lexre = re.compile("\s*(?:(\d+)|(\+)|(\*\*)|(\*))")
program = "1 + 3 * 4 ** 5 ** 6 + 2"
slexer = lexer.tokenize_regex(lexre, program)
tokens = TokenStream(slexer)
parser = Parser()
print (parser.expression(tokens))

#token_pat = re.compile("\s*(?:(\d+)|(.))")
#def tokenize(program):
    #for number, operator in token_pat.findall(program):
        #if number:
            #yield Literal()
        #elif operator == "+":
            #yield InfixSymbol("+", 10)
        #else:
            #raise SyntaxError("unknown operator")
    #yield EndOfInput()
    #yield EndOfInput()

#program = "1+2"
#print (",".join(map(str, tokenize(program))))
#it = tokenize(program)
#print (Parser(it).expression())
