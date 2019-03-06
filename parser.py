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
    def rule_setup(tokens, actions, *args):
        tokens.push()
        try:
            value = fn(tokens, actions, *args)
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

    def __call__(self, tokens, actions):
        return getattr(actions, self.action)(next(tokens)[1])


class PrefixHandler:
    def __init__(self, subglue, action):
        self.subglue = subglue
        self.action = action

    def __call__(self, tokens, actions):
        next(tokens)
        operand = expr(tokens, actions, self.subglue)
        return getattr(actions, self.action)(operand)


class InfixHandler:
    def __init__(self, glue, assoc, action):
        if assoc not in ('left', 'right'):
            raise ValueError("Associativity must be either left or right")
        self.glue = glue
        self.subglue = glue + (0 if assoc == 'right' else 1)
        self.action = action

    def __call__(self, tokens, actions, lhs):
        next(tokens)
        rhs = expr(tokens, actions, self.subglue)
        return getattr(actions, self.action)(lhs, rhs)


@rule
def do_ctrl(tokens, actions):
    dovar = tokens.expect_cat(lexer.CAT_WORD)
    tokens.expect('=')
    start = expr(tokens, actions)
    tokens.expect(',')
    stop = expr(tokens, actions)
    if tokens.marker(','):
        step = expr(tokens, actions)
    else:
        step = None
    return actions.do_ctrl(dovar, start, stop, step)


@rule
def implied_do(tokens, actions):
    args = []
    tokens.expect('(')
    args.append(expr(tokens, actions))
    tokens.expect(',')
    while True:
        try:
            do_ctrl_result = do_ctrl(tokens, actions)
        except NoMatch:
            args.append(expr(tokens, actions))
            tokens.expect(',')
        else:
            tokens.expect(')')
            return actions.impl_do(do_ctrl_result, *args)


class InplaceArrayHandler:
    def __call__(self, tokens, actions):
        next(tokens)
        seq = []
        if tokens.marker('/)'):
            return actions.array()
        while True:
            try:
                seq.append(implied_do(tokens, actions))
            except NoMatch:
                seq.append(expr(tokens, actions))
            if tokens.marker('/)'):
                return actions.array(*seq)
            tokens.expect(',')


class ParensHandler:
    def __call__(self, tokens, actions):
        next(tokens)
        inner_expr = expr(tokens, actions)
        tokens.expect(')')
        return actions.parens(inner_expr)


@rule
def slice_(tokens, actions):
    try:
        slice_begin = expr(tokens, actions)
    except NoMatch:
        slice_begin = None
    tokens.expect(':')
    try:
        slice_end = expr(tokens, actions)
    except NoMatch:
        slice_end = None
    if tokens.marker(":"):
        slice_stride = expr(tokens, actions)
    else:
        slice_stride = None
    return actions.slice(slice_begin, slice_end, slice_stride)


class SubscriptHandler:
    def __init__(self, glue):
        self.glue = glue

    def __call__(self, tokens, actions, lhs):
        next(tokens)
        seq = []
        if tokens.marker(')'):
            return actions.call(lhs)
        while True:
            try:
                seq.append(slice_(tokens, actions))
            except NoMatch:
                seq.append(expr(tokens, actions))
            if tokens.marker(')'):
                return actions.call(lhs, *seq)
            tokens.expect(',')


class ExpressionHandler:
    def __init__(self):
        prefix_ops = {
            ".not.":  PrefixHandler( 50, 'not_'),
            "+":      PrefixHandler(110, 'pos'),
            "-":      PrefixHandler(110, 'neg'),
            "(":      ParensHandler(),
            "(/":     InplaceArrayHandler()
            }
        infix_ops = {
            ".eqv.":  InfixHandler( 20, 'right', 'eqv'),
            ".neqv.": InfixHandler( 20, 'right', 'neqv'),
            ".or.":   InfixHandler( 30, 'right', 'or_'),
            ".and.":  InfixHandler( 40, 'right', 'and_'),
            ".eq.":   InfixHandler( 60, 'left',  'eq'),
            ".ne.":   InfixHandler( 60, 'left',  'ne'),
            ".le.":   InfixHandler( 60, 'left',  'le'),
            ".lt.":   InfixHandler( 60, 'left',  'lt'),
            ".ge.":   InfixHandler( 60, 'left',  'ge'),
            ".gt.":   InfixHandler( 60, 'left',  'gt'),
            "//":     InfixHandler( 70, 'left',  'concat'),
            "+":      InfixHandler( 80, 'left',  'plus'),
            "-":      InfixHandler( 80, 'left',  'minus'),
            "*":      InfixHandler( 90, 'left',  'mul'),
            "/":      InfixHandler( 90, 'left',  'div'),
            "**":     InfixHandler(100, 'right', 'pow'),
            "%":      InfixHandler(130, 'left',  'resolve'),
            "_":      InfixHandler(130, 'left',  'kind'),
            "(":      SubscriptHandler(140),
            }

        # Fortran 90 operator aliases
        infix_ops["=="] = infix_ops[".eq."]
        infix_ops["/="] = infix_ops[".ne."]
        infix_ops["<="] = infix_ops[".le."]
        infix_ops[">="] = infix_ops[".ge."]
        infix_ops["<"]  = infix_ops[".lt."]
        infix_ops[">"]  = infix_ops[".gt."]

        prefix_cats = {
            lexer.CAT_STRING:     LiteralHandler('string'),
            lexer.CAT_FLOAT:      LiteralHandler('float'),
            lexer.CAT_INT:        LiteralHandler('int'),
            lexer.CAT_RADIX:      LiteralHandler('radix'),
            lexer.CAT_BOOLEAN:    LiteralHandler('bool'),
            lexer.CAT_CUSTOM_DOT: PrefixHandler(120, 'unary'),
            lexer.CAT_WORD:       LiteralHandler('word'),
            }
        infix_cats = {
            lexer.CAT_CUSTOM_DOT: InfixHandler(10, 'left', 'binary')
            }

        self._infix_ops = infix_ops
        self._infix_cats = infix_cats
        self._prefix_ops = prefix_ops
        self._prefix_cats = prefix_cats

    def get_prefix_handler(self, cat, token):
        try:
            if cat == lexer.CAT_OP:
                return self._prefix_ops[token.lower()]
            else:
                return self._prefix_cats[cat]
        except KeyError:
            raise NoMatch()

    def get_infix_handler(self, cat, token):
        try:
            if cat == lexer.CAT_OP:
                return self._infix_ops[token.lower()]
            else:
                return self._infix_cats[cat]
        except KeyError:
            raise NoMatch()

EXPR_HANDLER = ExpressionHandler()

@rule
def expr(tokens, actions, min_glue=0):
    # Get prefix
    handler = EXPR_HANDLER.get_prefix_handler(*tokens.peek())
    result = handler(tokens, actions)

    # Cycle through appropriate infixes:
    while True:
        try:
            handler = EXPR_HANDLER.get_infix_handler(*tokens.peek())
        except NoMatch:
            return result
        else:
            if handler.glue < min_glue:
                return result
            result = handler(tokens, actions, result)


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
actions = DefaultActions()
print (expr(tokens, actions))
#parser = BlockParser(DefaultActions())
#print (parser.block(tokens))
