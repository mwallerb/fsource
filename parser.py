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

    def marker(self, marker):
        if self.token == marker:
            self.advance()
            return True
        else:
            return False


class IgnoreHandler:
    def __init__(self):
        self.glue = 100000

    def expr(self, tokens, head_result):
        tokens.advance()
        return head_result


class LiteralHandler:
    def __init__(self, action=None):
        if action is None: action = lambda x: "'%s'" % x
        self.action = action
        self.glue = 100000

    def expr(self, tokens, head_result):
        if head_result is not None:
            raise ValueError("Two literals in a row are forbidden")
        return self.action(tokens.advance())


class PrefixHandler:
    def __init__(self, parser, symbol, subglue, action=None):
        if action is None: action = lambda x: "(%s %s)" % (symbol, x)
        self.parser = parser
        self.symbol = symbol
        self.subglue = subglue
        self.action = action

    def expr(self, tokens, head_result):
        if head_result is not None:
            raise ValueError("%s is a prefix operator" % self.symbol)
        tokens.advance()
        result = self.parser.expression(tokens, self.subglue)
        return self.action(result)


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

    def expr(self, tokens, head_result):
        if head_result is None:
            raise ValueError("%s is an infix operator" % self.symbol)
        tokens.advance()
        tail_result = self.parser.expression(tokens, self.subglue)
        return self.action(head_result, tail_result)


class PrefixOrInfix:
    def __init__(self, prefix, infix):
        self.prefix = prefix
        self.infix = infix
        self.glue = infix.glue

    def expr(self, tokens, head_result):
        if head_result is None:
            return self.prefix.expr(tokens, None)
        else:
            return self.infix.expr(tokens, head_result)


class ParensHandler:
    def __init__(self, parser, begin_symbol, end_symbol, inner_glue, action=None):
        if action is None:
            action = lambda x: "(%s%s %s)" % (begin_symbol, end_symbol, x)

        self.parser = parser
        self.begin_symbol = begin_symbol
        self.end_symbol = end_symbol
        self.inner_glue = inner_glue
        self.action = action

    def expr(self, tokens, head_result):
        if head_result is not None:
            raise ValueError("Invalid call type %s" % self.begin_symbol)
        tokens.advance()
        result = self.parser.expression(tokens, self.inner_glue)
        tokens.expect(self.end_symbol)
        return self.action(result)


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

    def expr(self, tokens, head_result):
        if head_result is None:
            raise ValueError("Invalid parenthesis type %s" % self.begin_symbol)
        tokens.advance()
        tail_result = self.parser.expression(tokens, self.inner_glue)
        tokens.expect(self.end_symbol)
        return self.action(head_result, tail_result)


class EndExpressionMarker:
    def __init__(self, inner_glue):
        self.glue = inner_glue - 1

    def expr(self, tokens, head_result):
        raise ValueError("Empty expression")


class SliceHandler:
    def __init__(self, parser, symbol, glue, action=None):
        if action is None:
            action = lambda x, y, z: "(%s %s %s %s)" % (symbol, x, y, z)

        self.parser = parser
        self.symbol = symbol
        self.glue = glue
        self.action = action

    def expr(self, tokens, head_result):
        slice_begin = head_result
        tokens.advance()
        slice_end = self.parser.expression(tokens, self.glue + 1, False)
        if tokens.marker(self.symbol):
            slice_stride = self.parser.expression(tokens, self.glue + 1)
            if tokens.token == self.symbol:
                raise ValueError("Too many %s" % self.symbol)
        else:
            slice_stride = None
        return self.action(slice_begin, slice_end, slice_stride)


class Parser:
    def __init__(self):
        operators = {
            ",":      InfixHandler(self, ",",     -20, 'left'),
            ":":      SliceHandler(self, ":",     -10),
            ".eqv.":  InfixHandler(self, ".eqv.",  20, 'right'),
            ".neqv.": InfixHandler(self, ".neqv.", 20, 'right'),
            ".or.":   InfixHandler(self, ".or.",   30, 'right'),
            ".and.":  InfixHandler(self, ".and.",  40, 'right'),
            ".not.":  PrefixHandler(self, ".not.",  50),
            ".eq.":   InfixHandler(self, ".eq.",   60, 'left'),
            ".ne.":   InfixHandler(self, ".neq.",  60, 'left'),
            ".le.":   InfixHandler(self, ".le.",   60, 'left'),
            ".lt.":   InfixHandler(self, ".lt.",   60, 'left'),
            ".ge.":   InfixHandler(self, ".ge.",   60, 'left'),
            ".gt.":   InfixHandler(self, ".gt.",   60, 'left'),
            "//":     InfixHandler(self, "//",     70, 'left'),
            "+":      PrefixOrInfix(
                        PrefixHandler(self, "+",   110),
                        InfixHandler (self, "+",    80, 'left')
                        ),
            "-":      PrefixOrInfix(
                        PrefixHandler(self, "-",   110),
                        InfixHandler(self, "-",    80, 'left')
                        ),
            "*":      InfixHandler(self, "*",      90, 'left'),
            "**":     InfixHandler(self, "**",    100, 'right'),
            "%":      InfixHandler(self, "%",     130, 'left'),
            "_":      InfixHandler(self, "_",     130, 'left'),
            "(":      PrefixOrInfix(
                        ParensHandler(self, '(', ')', 0),
                        SubscriptHandler(self, '(', ')', 140, -20),     # TODO
                        ),
            ")":      EndExpressionMarker(-20),
            ".true.": LiteralHandler(),
            ".false.": LiteralHandler(),
            }

        # Fortran 90 operator aliases
        operators["=="] = operators[".eq."]
        operators["/="] = operators[".ne."]
        operators["<="] = operators[".le."]
        operators[">="] = operators[".ge."]
        operators["<"]  = operators[".lt."]
        operators[">"]  = operators[".gt."]

        cat_switch = (
                EndExpressionMarker(0),         # end of input
                IgnoreHandler(),                # line number
                IgnoreHandler(),                # preprocessor
                EndExpressionMarker(0),         # end of stmt
                IgnoreHandler(),                # line cont'n
                IgnoreHandler(),                # comment
                LiteralHandler(),               # string
                LiteralHandler(),               # float
                LiteralHandler(),               # int
                LiteralHandler(),               # radix
                EndExpressionMarker(0),         # bracketed slash
                None,                           # operator (11)
                PrefixOrInfix(
                        PrefixHandler(self, '.unary.', 120),
                        InfixHandler(self, '.binary.', 10, 'left')
                        ),
                LiteralHandler(),               # word
                )

        self._operators = operators
        self._cat_switch = cat_switch

    def _get_handler(self, cat, token):
        if cat == 11:
            return self._operators[token.lower()]
        else:
            return self._cat_switch[cat]

    def expression(self, tokens, min_glue=0, expect_result=True):
        result = None
        while True:
            handler = self._get_handler(tokens.cat, tokens.token)
            if handler.glue < min_glue:
                if expect_result and result is None:
                    raise ValueError("Expecting expression")
                return result
            result = handler.expr(tokens, result)


lexre = lexer.LEXER_REGEX
program = "+1 + 3 * x(::1,2:3) * 4 ** 5 ** sin(6, 1) + 2"
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
