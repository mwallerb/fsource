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
            #print ("ADV", self.cat, self.token)
        except StopIteration:
            self.cat, self.token = 0, '<$>'
        return token

    def expect(self, expected):
        if self.token != expected:
            raise ValueError("Expected %s, got %s", expected, self.token)
        self.advance()


class LiteralHandler:
    def __init__(self, action=None):
        if action is None:
            action = lambda x: "'%s'" % x

        self.action = action
        self.glue = None

    def head(self, tokens):
        return self.action(tokens.advance())

    def tail(self, tokens, head_result):
        raise ValueError("Two literals in a row are forbidden")


class PrefixHandler:
    def __init__(self, parser, symbol, subglue, action=None):
        if action is None:
            action = lambda x: "(%s %s)" % (symbol, x)

        self.parser = parser
        self.symbol = symbol
        self.subglue = subglue
        self.action = action

    def head(self, tokens):
        tokens.advance()
        result = self.parser.expression(tokens, self.subglue)
        return self.action(result)

    def tail(self, tokens, head_result):
        raise ValueError("%s is an prefix operator" % self.symbol)


class InfixHandler:
    def __init__(self, parser, symbol, glue, assoc, action=None):
        if action is None:
            action = lambda x, y: "(%s %s %s)" % (symbol, x, y)
        if assoc not in ('left', 'right'):
            raise ValueError("Associativity must be either left or right")

        self.parser = parser
        self.symbol = symbol
        self.glue = glue
        self.subglue = glue + (0 if assoc == 'right' else 1)
        self.action = action

    def head(self, tokens):
        raise ValueError("%s is an infix operator" % self.symbol)

    def tail(self, tokens, head_result):
        tokens.advance()
        tail_result = self.parser.expression(tokens, self.subglue)
        return self.action(head_result, tail_result)


class PrefixOrInfix:
    def __init__(self, prefix, infix):
        self.prefix = prefix
        self.infix = infix
        self.glue = infix.glue

    def head(self, tokens):
        return self.prefix.head(tokens)

    def tail(self, tokens, head_result):
        return self.infix.tail(tokens, head_result)


class ParensHandler:
    def __init__(self, parser, begin_symbol, end_symbol, inner_glue, action=None):
        if action is None:
            action = lambda x: "(%s%s %s)" % (begin_symbol, end_symbol, x)

        self.parser = parser
        self.begin_symbol = begin_symbol
        self.end_symbol = end_symbol
        self.inner_glue = inner_glue
        self.action = action

    def head(self, tokens):
        tokens.advance()
        result = self.parser.expression(tokens, self.inner_glue)
        tokens.expect(self.end_symbol)
        return self.action(result)

    def tail(self, tokens, head_result):
        raise ValueError("Invalid call type %s" % self.begin_symbol)


class SubscriptHandler:
    def __init__(self, parser, begin_symbol, end_symbol, glue, inner_glue, action=None):
        if action is None:
            action = lambda x, y: "(%s%s %s %s)" % (begin_symbol, end_symbol, x, y)

        self.parser = parser
        self.begin_symbol = begin_symbol
        self.end_symbol = end_symbol
        self.glue = glue
        self.inner_glue = inner_glue
        self.action = action

    def head(self, tokens):
        raise ValueError("Invalid parenthesis type %s" % self.begin_symbol)

    def tail(self, tokens, head_result):
        tokens.advance()
        tail_result = self.parser.expression(tokens, self.inner_glue)
        tokens.expect(self.end_symbol)
        return self.action(head_result, tail_result)


class EndExpressionMarker:
    def __init__(self, inner_glue):
        self.glue = inner_glue - 1


class Parser:
    def __init__(self):
        self.handlers = (
                EndExpressionMarker(0),
                LiteralHandler(),
                PrefixOrInfix(
                    PrefixHandler(self, '+', 100),
                    InfixHandler(self, '+', 10, 'left'),
                    ),
                InfixHandler(self, '**', 30, 'right'),
                InfixHandler(self, '*', 20, 'right'),
                PrefixOrInfix(
                    ParensHandler(self, '(', ')', 0),
                    SubscriptHandler(self, '(', ')', 150, 0),
                ),
                EndExpressionMarker(0),
                )

    def expression(self, tokens, min_glue=0):
        #print ("EXPR", min_glue)
        handler = self.handlers[tokens.cat]
        result = handler.head(tokens)

        handler = self.handlers[tokens.cat]
        while handler.glue >= min_glue:
            result = handler.tail(tokens, result)
            handler = self.handlers[tokens.cat]

        return result


lexre = re.compile("\s*(?:(\w+)|(\+)|(\*\*)|(\*)|(\()|(\)))")
program = "+1 + 3 * 4 ** 5 ** sin(6 + 1) + 2"
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
