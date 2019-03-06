#!/usr/bin/env python
from __future__ import print_function
import lexer
import re

class NoMatch(Exception):
    pass


class TokenStream:
    def __init__(self, tokens, pos=0):
        self.tokens = tokens
        self.pos = pos
        self.stack = []

    def peek(self):
        try:
            return self.tokens[self.pos]
        except IndexError:
            if self.pos > len(self.tokens):
                raise StopIteration()
            return lexer.CAT_DOLLAR, '<$>'

    def __next__(self):
        value = self.peek()
        self.pos += 1
        return value

    def push(self):
        self.stack.append(self.pos)

    def backtrack(self):
        self.pos = self.stack.pop()

    def commit(self):
        self.stack.pop()

    # FIXME: compatibility - remove

    def expect(self, expected):
        cat, token = self.peek()
        if token.lower() != expected:
            raise NoMatch()
        next(self)

    def expect_cat(self, expected):
        cat, token = self.peek()
        if cat != expected:
            raise NoMatch()
        return next(self)[1]

    def marker(self, expected):
        cat, token = self.peek()
        if token.lower() == expected:
            next(self)
            return True
        else:
            return False


def rule(fn):
    def rule_setup(self, tokens, *args):
        tokens.push()
        try:
            value = fn(self, tokens, *args)
        except:
            tokens.backtrack()
            raise
        else:
            tokens.commit()
            return value
    return rule_setup


class LockedIn:
    def __init__(self, tokens):
        self.tokens = tokens

    def __enter__(self): pass

    def __exit__(self, exc_type, exc_val, traceback):
        if exc_type is NoMatch:
            raise ValueError("Parsing failure")


class LiteralHandler:
    def __init__(self, action):
        self.action = action

    def handle(self, tokens):
        return self.action(next(tokens)[1])


class PrefixHandler:
    def __init__(self, expression, symbol, subglue, action):
        self.expression = expression
        self.symbol = symbol
        self.subglue = subglue
        self.action = action

    def handle(self, tokens):
        next(tokens)
        operand = self.expression.parse(tokens, self.subglue)
        return self.action(operand)


class InfixHandler:
    def __init__(self, expression, symbol, glue, assoc, action):
        if assoc not in ('left', 'right'):
            raise ValueError("Associativity must be either left or right")

        self.expression = expression
        self.symbol = symbol
        self.glue = glue
        self.subglue = glue + (0 if assoc == 'right' else 1)
        self.action = action

    def handle(self, tokens, lhs):
        next(tokens)
        rhs = self.expression.parse(tokens, self.subglue)
        return self.action(lhs, rhs)


class DoCtrlParser:
    def __init__(self, expression, action):
        self.expression = expression
        self.action = action

    @rule
    def parse(self, tokens):
        dovar = tokens.expect_cat(lexer.CAT_WORD)
        tokens.expect('=')
        start = self.expression.parse(tokens)
        tokens.expect(',')
        stop = self.expression.parse(tokens)
        if tokens.marker(','):
            step = self.expression.parse(tokens)
        else:
            step = None
        return self.action(dovar, start, stop, step)


class ImpliedDoParser:
    def __init__(self, expression, do_ctrl, action):
        self.expression = expression
        self.do_ctrl = do_ctrl
        self.action = action

    @rule
    def parse(self, tokens):
        args = []
        tokens.expect('(')
        args.append(self.expression.parse(tokens))
        tokens.expect(',')
        while True:
            try:
                do_ctrl = self.do_ctrl.parse(tokens)
            except NoMatch:
                args.append(self.expression.parse(tokens))
                tokens.expect(',')
            else:
                tokens.expect(')')
                return self.action(do_ctrl, *args)


class InplaceArrayHandler:
    def __init__(self, expression, impl_do, action):
        self.expression = expression
        self.impl_do = impl_do
        self.action = action

    def handle(self, tokens):
        next(tokens)
        seq = []
        if tokens.marker('/)'):
            return self.action()
        while True:
            try:
                seq.append(self.impl_do.parse(tokens))
            except NoMatch:
                seq.append(self.expression.parse(tokens))
            if tokens.marker('/)'):
                return self.action(*seq)
            tokens.expect(',')


class ParensHandler:
    def __init__(self, expression, action):
        self.expression = expression
        self.action = action

    def handle(self, tokens):
        next(tokens)
        expr = self.expression.parse(tokens)
        tokens.expect(')')
        return self.action(expr)


class SliceParser:
    def __init__(self, expression, action):
        self.expression = expression
        self.action = action

    @rule
    def parse(self, tokens):
        try:
            slice_begin = self.expression.parse(tokens)
        except NoMatch:
            slice_begin = None
        tokens.expect(':')
        try:
            slice_end = self.expression.parse(tokens)
        except NoMatch:
            slice_end = None
        if tokens.marker(":"):
            slice_stride = self.expression.parse(tokens)
        else:
            slice_stride = None
        return self.action(slice_begin, slice_end, slice_stride)


class SubscriptHandler:
    def __init__(self, expression, slice_, glue, action):
        self.expression = expression
        self.slice_ = slice_
        self.glue = glue
        self.action = action

    def handle(self, tokens, lhs):
        next(tokens)
        seq = []
        if tokens.marker(')'):
            return self.action(lhs)
        while True:
            try:
                seq.append(self.slice_.parse(tokens))
            except NoMatch:
                seq.append(self.expression.parse(tokens))
            if tokens.marker(')'):
                return self.action(lhs, *seq)
            tokens.expect(',')


class ExpressionParser:
    def __init__(self, actions):
        do_ctrl_parser = DoCtrlParser(self, actions.do_ctrl)
        impl_do_parser = ImpliedDoParser(self, do_ctrl_parser, actions.array)
        slice_parser = SliceParser(self, actions.slice)

        prefix_ops = {
            ".not.":  PrefixHandler(self, ".not.",  50, actions.not_),
            "+":      PrefixHandler(self, "+",     110, actions.pos),
            "-":      PrefixHandler(self, "-",     110, actions.neg),
            "(":      ParensHandler(self, actions.parens),
            "(/":     InplaceArrayHandler(self, impl_do_parser, actions.array)
            }
        infix_ops = {
            ".eqv.":  InfixHandler(self, ".eqv.",  20, 'right', actions.eqv),
            ".neqv.": InfixHandler(self, ".neqv.", 20, 'right', actions.neqv),
            ".or.":   InfixHandler(self, ".or.",   30, 'right', actions.or_),
            ".and.":  InfixHandler(self, ".and.",  40, 'right', actions.and_),
            ".eq.":   InfixHandler(self, ".eq.",   60, 'left',  actions.eq),
            ".ne.":   InfixHandler(self, ".neq.",  60, 'left',  actions.ne),
            ".le.":   InfixHandler(self, ".le.",   60, 'left',  actions.le),
            ".lt.":   InfixHandler(self, ".lt.",   60, 'left',  actions.lt),
            ".ge.":   InfixHandler(self, ".ge.",   60, 'left',  actions.ge),
            ".gt.":   InfixHandler(self, ".gt.",   60, 'left',  actions.gt),
            "//":     InfixHandler(self, "//",     70, 'left',  actions.concat),
            "+":      InfixHandler(self, "+",      80, 'left',  actions.plus),
            "-":      InfixHandler(self, "-",      80, 'left',  actions.minus),
            "*":      InfixHandler(self, "*",      90, 'left',  actions.mul),
            "/":      InfixHandler(self, "/",      90, 'left',  actions.div),
            "**":     InfixHandler(self, "**",    100, 'right', actions.pow),
            "%":      InfixHandler(self, "%",     130, 'left',  actions.resolve),
            "_":      InfixHandler(self, "_",     130, 'left',  actions.kind),
            "(":      SubscriptHandler(self, slice_parser, 140, actions.call),
            }

        # Fortran 90 operator aliases
        infix_ops["=="] = infix_ops[".eq."]
        infix_ops["/="] = infix_ops[".ne."]
        infix_ops["<="] = infix_ops[".le."]
        infix_ops[">="] = infix_ops[".ge."]
        infix_ops["<"]  = infix_ops[".lt."]
        infix_ops[">"]  = infix_ops[".gt."]

        prefix_cats = {
            lexer.CAT_STRING:     LiteralHandler(actions.string),
            lexer.CAT_FLOAT:      LiteralHandler(actions.float),
            lexer.CAT_INT:        LiteralHandler(actions.int),
            lexer.CAT_RADIX:      LiteralHandler(actions.radix),
            lexer.CAT_BOOLEAN:    LiteralHandler(actions.bool),
            lexer.CAT_CUSTOM_DOT: PrefixHandler(self, '.unary.', 120, actions.unary),
            lexer.CAT_WORD:       LiteralHandler(actions.word),
            }
        infix_cats = {
            lexer.CAT_CUSTOM_DOT: InfixHandler(self, '.binary.', 10, 'left', actions.binary)
            }

        self._infix_ops = infix_ops
        self._infix_cats = infix_cats
        self._prefix_ops = prefix_ops
        self._prefix_cats = prefix_cats

    def _get_prefix_handler(self, cat, token):
        try:
            if cat == lexer.CAT_OP:
                return self._prefix_ops[token.lower()]
            else:
                return self._prefix_cats[cat]
        except KeyError:
            raise NoMatch()

    def _get_infix_handler(self, cat, token):
        try:
            if cat == lexer.CAT_OP:
                return self._infix_ops[token.lower()]
            else:
                return self._infix_cats[cat]
        except KeyError:
            raise NoMatch()

    def parse(self, tokens, min_glue=0):
        # Get prefix
        handler = self._get_prefix_handler(*tokens.peek())
        result = handler.handle(tokens)

        # Cycle through appropriate infixes:
        while True:
            try:
                handler = self._get_infix_handler(*tokens.peek())
            except NoMatch:
                return result
            else:
                if handler.glue < min_glue:
                    return result
                result = handler.handle(tokens, result)


def _opt(x): return "null" if x is None else x

class DefaultActions:
    # Blocks
    def block(self, *s): return "(block %s)" % " ".join(map(_opt, s))
    def if_(self, *b): return "(if %s)" % " ".join(map(_opt, b))
    def arith_if(self, a, b, c): return "(arith_if %s %s %s)" % (a, b, c)
    def where(self, *b): return "(where %s)" % " ".join(map(_opt, b))

    # Expression actions
    def eqv(self, l, r): return "(eqv %s %s)" % (l, r)
    def neqv(self, l, r): return "(neqv %s %s)" % (l, r)
    def or_(self, l, r): return "(or %s %s)" % (l, r)
    def and_(self, l, r): return "(and %s %s)" % (l, r)
    def not_(self, op): return "(not %s)" % op
    def eq(self, l, r): return "(eq %s %s)" % (l, r)
    def ne(self, l, r): return "(ne %s %s)" % (l, r)
    def le(self, l, r): return "(le %s %s)" % (l, r)
    def lt(self, l, r): return "(lt %s %s)" % (l, r)
    def ge(self, l, r): return "(ge %s %s)" % (l, r)
    def gt(self, l, r): return "(gt %s %s)" % (l, r)
    def concat(self, l, r): return "(// %s %s)" % (l, r)
    def plus(self, l, r): return "(+ %s %s)" % (l, r)
    def pos(self, op): return "(pos %s)" % op
    def minus(self, l, r): return "(- %s %s)" % (l, r)
    def neg(self, op): return "(neg %s)" % op
    def mul(self, l, r): return "(* %s %s)" % (l, r)
    def div(self, l, r): return "(/ %s %s)" % (l, r)
    def pow(self, l, r): return "(pow %s %s)" % (l, r)
    def resolve(self, l, r): return "(%% %s %s)" % (l, r)
    def kind(self, l, r): return "(_ %s %s)" % (l, r)
    def parens(self, op): return op
    def call(self, fn, *args): return "(() %s %s)" % (fn, " ".join(args))
    def slice(self, b, e, s): return "(: %s %s %s)" % tuple(map(_opt, (b,e,s)))
    def array(self, *args): return "(array %s)" % " ".join(args)
    def impl_do(self, c, *args): return "(implied_do %s %s)" % (c, " ".join(args))
    def do_ctrl(self, v, b, e, s): return "(do_ctrl %s %s %s %s)" % (v, b, e, _opt(s))
    def unary(self, op): return "(unary %s)" % op
    def binary(self, l, r): return "(binary %s %s)" % (l, r)

    # Literals actions
    def bool(self, tok): return tok.lower()[1:-1]
    def int(self, tok): return tok
    def float(self, tok): return tok
    def string(self, tok): return "(string %s)" % repr(lexer.parse_string(tok))
    def radix(self, tok): return "(radix %s)" % tok
    def word(self, tok): return repr(tok)


lexre = lexer.LEXER_REGEX
#program = """x(3:1, 4, 5::2) * &   ! something
#&  (3 + 5)"""
program = "+1 + 3 * x(::1, 2:3) * (/ /) * 4 ** (5 + 1) ** sin(.true., 1) + (/ 1, 2, (i, i=1,5), 3 /)"
#program = """
    #if (x == 3) then
        #call something(3)
        #return
    #else if (x == 5) then
        #call something(4)
    #else
        #call something(5)
    #end
    #where (a == 3)
        #return
    #endwhere

    #"""
slexer = lexer.tokenize_regex(lexre, program)
tokens = TokenStream(list(slexer))
expression = ExpressionParser(DefaultActions())
print (expression.parse(tokens))
#parser = BlockParser(DefaultActions())
#print (parser.block(tokens))
