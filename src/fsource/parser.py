"""
Parser and abstract syntax tree generator for free-form Fortran.

Takes a stream of tokens from a lexer and parses it, creating an
abstract syntax tree (AST) with the following structure: each node is
either a terminal or non-terminal node.  A terminal node can either be
None, True, False, or any string. A non-terminal node is a tuple, where
the first item is a string describing the type of node and subsequent
items (if any) are (child) nodes.

The parser is basically a recursive descent parser [1], with a couple of
improvements to speed it up:

 1. the Fortran grammar is extremely prefix-heavy, so where applicable
    we dispatch rules based on the first token using `prefixes()`
    instead of trying them in order and then backtracking if they
    don't match.

 2. the expression parser `expr()` uses top-down operator precedence
    parsing [2], which dispatches based on the token type (also for
    infix operators). This does away with backtracking and is therefore
    guaranteed linear runtime.

[1]: https://en.wikipedia.org/wiki/Recursive_descent_parser
[2]: https://doi.org/10.1145/512927.512931

Copyright 2019 Markus Wallerberger
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
from __future__ import print_function
import re

from . import __version_tuple__
from . import lexer
from . import common


class NoMatch(Exception):
    """Current rule does not match, try next one if available.

    This class is used by the recursive descent parser for backtracking: by
    raising `NoMatch`, you indicate that the current rule did not match, the
    parser should backtrack and try the next rule if available.

    See also: `rule()`, `TokenStream.backtrack()`
    """


class EndOfBlock(Exception):
    """A performance-enhancing exception"""


class ParserError(common.ParsingError):
    """Current rule does not match even though it should, fail meaningfully.

    This class indicates a parsing error in the current rule: it should be
    raised if the current not only does not match, but we are sure no other
    rule can match, i.e., this is not valid Fortran.  This is useful because
    it allows detecting errors early and close to the actual error condition.

    See also: `LockedIn`
    """
    @property
    def error_type(self): return "parser error"

    def __init__(self, tokens, msg):
        row, col, _, token = tokens.peek()
        line = tokens.current_line()

        common.ParsingError.__init__(self, tokens.fname, row, col,
                                     col+len(token), line, msg)


class TokenStream:
    """Iterator over tokens, maintaining a stack of rollback points.

    `TokenStream` is an iterator over a sequence of `tokens`.  However, it
    also maintains a stack of rollback points: `push()` adds a rollback point
    to the stack at the current token, `commit()` removes the last rollback
    point, whereas `backtrack()` removes and returns to the rollback point.
    """
    def __init__(self, tokens, fname=None, pos=0):
        self.fname = fname
        self.tokens = tuple(tokens)
        self.pos = pos

    def peek(self):
        """Get current token without consuming it"""
        return self.tokens[self.pos]

    def advance(self):
        """Advance to next token"""
        self.pos += 1

    def __iter__(self):
        return self

    def __next__(self):
        """Return current token and advance iterator"""
        pos = self.pos
        self.pos += 1
        return self.tokens[pos]

    next = __next__       # Python 2

    def produce(self, header, *args):
        """Produce a node in the abstract syntax tree."""
        return (header,) + args

    def current_line(self):
        """Return current line"""
        # HACK: this builds up the line from tokens, which is ugly. Also, it
        # does not preserve the type of whitespace (' ' vs '\t')
        # Get token range for the currrent line
        row = self.peek()[0]
        try:
            offset = next(i for (i, (r, _, _, _)) in
                          enumerate(self.tokens[self.pos-1::-1]) if r != row)
            start = self.pos - offset
        except StopIteration:
            start = 0
        try:
            offset = next(i for (i, (r, _, _, _)) in
                          enumerate(self.tokens[self.pos:]) if r != row)
            stop = self.pos + offset
        except StopIteration:
            stop = len(self.tokens)

        # Build up line
        line = ''
        for _, col, _, token in self.tokens[start:stop]:
            line += ' ' * (col - len(line)) + token
        return line


def rule(fn):
    """Decorator for decursive descent rule.

    The `rule` decorator implements recursive descent logic using token
    streams: before entering `fn`, it creates a rollback point, which is
    removed on exit.  If `fn` does not match, i.e., raises `NoMatch`, we
    backtrack to the rollback point.
    """
    def rule_setup(tokens, *args):
        begin_pos = tokens.pos
        try:
            return fn(tokens, *args)
        except NoMatch:
            tokens.pos = begin_pos
            raise

    return rule_setup


class LockedIn:
    """Context manager converting `NoMatch` to `ParserError`.

    This context manager allows a rule to be "locked in", i.e., we know that
    the rule should match and any non-matching part after that is in fact an
    error.
    """
    def __init__(self, tokens, err):
        self.tokens = tokens
        self.err = err

    def __enter__(self): pass

    def __exit__(self, exc_type, exc_val, traceback):
        if exc_type is NoMatch:
            raise ParserError(self.tokens, self.err)


def expect(tokens, expected):
    """Matches the next token being `expected`"""
    token = tokens.peek()[3]
    if token.lower() != expected:
        raise NoMatch()
    tokens.advance()

def expect_cat(tokens, expected):
    """Matches the next token of category `expected`"""
    cat = tokens.peek()[2]
    if cat != expected:
        raise NoMatch()
    return next(tokens)[3]

def marker(tokens, expected):
    """Return if next token is `expected`, and consume token if so."""
    token = tokens.peek()[3]
    if token.lower() == expected:
        tokens.advance()
        return True
    else:
        return False

def comma_sequence(inner_rule, production_tag, allow_empty=False):
    """A comma-separated list of items matching `inner_rule`."""
    def comma_sequence_rule(tokens):
        vals = []
        try:
            vals.append(inner_rule(tokens))
        except NoMatch:
            if allow_empty:
                return tokens.produce(production_tag)
            raise
        try:
            while marker(tokens, ','):
                vals.append(inner_rule(tokens))
        except NoMatch:
            raise ParserError(tokens, "Expecting item in comma-separated list")
        return tokens.produce(production_tag, *vals)

    return comma_sequence_rule

def ws_sequence(inner_rule, production_tag):
    """A whitespace-separated list of items matching `inner_rule`."""
    def ws_sequence_rule(tokens):
        items = []
        try:
            while True:
                items.append(inner_rule(tokens))
        except NoMatch:
            return tokens.produce(production_tag, *items)

    return ws_sequence_rule

def optional(inner_rule, *args):
    """Matches `inner_rule`, returning `None` if it does not match."""
    def optional_rule(tokens):
        try:
            return inner_rule(tokens, *args)
        except NoMatch:
            return None

    return optional_rule

def tag(expected, production_tag):
    """A single keyword"""
    @rule
    def tag_rule(tokens):
        expect(tokens, expected)
        return (production_tag,)

    return tag_rule

def tag_stmt(expected, production_tag):
    """A single keyword, terminated by end of statement"""
    @rule
    def tag_rule(tokens):
        expect(tokens, expected)
        eos(tokens)
        return (production_tag,)

    return tag_rule

def matches(rule_, tokens):
    """Attempts to match `rule_`, returning whether it succeeded"""
    try:
        rule_(tokens)
        return True
    except NoMatch:
        return False

def null_rule(_tokens, produce=None):
    """Match empty"""
    return produce

def prefix(expected, my_rule, production_tag):
    """A tag `expected` followed by a rule `my_rule`"""
    @rule
    def prefix_rule(tokens):
        expect(tokens, expected)
        value = my_rule(tokens)
        return (production_tag, value)

    return prefix_rule

def prefixes(handlers):
    """Fast dispatch of different rules based on a dictionary of prefixes.

    Expects a dictionary `handlers`, where each key is a prefix, and each
    value is a rule.  Note that the rule should include matching the prefix.
    This provides a speedup over traditional recursive descent: rather than
    trying the rules one-by-one, we can directly jump to the correct candidate.
    """
    def prefixes_rule(tokens):
        token = tokens.peek()[3]
        try:
            handler = handlers[token.lower()]
        except KeyError:
            raise NoMatch()
        return handler(tokens)

    return prefixes_rule

def composite(word1, word2):
    """Two words, optionally separated by whitespace.

    Fortran allows omitting whitespace in certain clauses, i.e., both `ENDIF`
    and `END IF` are equivalent forms of ending an if statement.  The lexer
    cannot disambiguate this, because `ENDIF` is also a valid variable name.
    """
    comp = word1 + word2
    @rule
    def composite_rule(tokens):
        if marker(tokens, word1):
            expect(tokens, word2)
        else:
            expect(tokens, comp)

    return composite_rule

def eos(tokens):
    """End of statement"""
    return expect_cat(tokens, lexer.CAT_EOS)

def int_(tokens):
    """Integer literal"""
    return tokens.produce('int', expect_cat(tokens, lexer.CAT_INT))

def string_(tokens):
    """String literal"""
    return tokens.produce('string', expect_cat(tokens, lexer.CAT_STRING))

def identifier(tokens):
    """Identifier in a non-expression context"""
    return tokens.produce('id', expect_cat(tokens, lexer.CAT_WORD))

def id_ref(tokens):
    """Identifier in a expression context (reference)"""
    return tokens.produce('ref', expect_cat(tokens, lexer.CAT_WORD))

def custom_op(tokens):
    """Non-intrinsic (user-defined) operator"""
    return tokens.produce('custom_op', expect_cat(tokens, lexer.CAT_CUSTOM_DOT))

@rule
def do_ctrl(tokens):
    """Numeric loop control clause"""
    dovar = identifier(tokens)
    expect(tokens, '=')
    start = expr(tokens)
    expect(tokens, ',')
    stop = expr(tokens)
    if marker(tokens, ','):
        step = expr(tokens)
    else:
        step = None
    return tokens.produce('do_ctrl', dovar, start, stop, step)

@rule
def implied_do(tokens):
    args = []
    expect(tokens, '(')
    args.append(expr(tokens))
    expect(tokens, ',')
    while True:
        try:
            do_ctrl_result = do_ctrl(tokens)
            break
        except NoMatch:
            args.append(expr(tokens))
            expect(tokens, ',')

    expect(tokens, ')')
    return tokens.produce('impl_do', do_ctrl_result, *args)

def inplace_array(open_delim, close_delim):
    @rule
    def inplace_array_rule(tokens):
        seq = []
        expect(tokens, open_delim)
        with LockedIn(tokens, "invalid inplace array"):
            if marker(tokens, close_delim):
                return tokens.produce('array')
            while True:
                try:
                    seq.append(implied_do(tokens))
                except NoMatch:
                    seq.append(expr(tokens))
                if marker(tokens, close_delim):
                    return tokens.produce('array', *seq)
                expect(tokens, ',')

    return inplace_array_rule

def _slice_tail(tokens, slice_begin):
    with LockedIn(tokens, "invalid slice object"):
        slice_end = _optional_expr(tokens)
        if marker(tokens, ":"):
            slice_stride = expr(tokens)
        else:
            slice_stride = None
        return tokens.produce('slice', slice_begin, slice_end, slice_stride)

def argument(tokens):
    try:
        item = expr(tokens)
    except NoMatch:
        expect(tokens, ':')
        return _slice_tail(tokens, None)

    discr = tokens.peek()[3]
    if discr == '=':
        if item[0] != 'ref':
            raise ParserError(tokens, "invalid argument name")
        tokens.advance()
        value = expr(tokens)
        return tokens.produce('arg', item, value)
    elif discr == ':':
        tokens.advance()
        return _slice_tail(tokens, item)
    else:
        return item

subscript_sequence = comma_sequence(argument, 'sub_list', allow_empty=True)

def prefix_op_handler(subglue, action):
    def prefix_op_handle(tokens):
        tokens.advance()
        operand = expr(tokens, subglue)
        return tokens.produce(action, operand)

    return prefix_op_handle

def custom_unary_handler(subglue):
    def custom_unary_handle(tokens):
        operator = custom_op(tokens)
        operand = expr(tokens, subglue)
        return tokens.produce('unary', operator, operand)

    return custom_unary_handle

def infix_op_handler(subglue, action):
    def infix_op_handle(tokens, lhs):
        tokens.advance()
        rhs = expr(tokens, subglue)
        return tokens.produce(action, lhs, rhs)

    return infix_op_handle

def custom_binary_handler(subglue):
    def custom_binary_handle(tokens, lhs):
        operator = custom_op(tokens)
        rhs = expr(tokens, subglue)
        return tokens.produce('binary', operator, lhs, rhs)

    return custom_binary_handle

def literal_handler(action):
    # We don't need to check for the token type here, since we have already
    # done so at the dispatch phase for expr()
    def literal_handle(tokens):
        return tokens.produce(action, next(tokens)[3])
    return literal_handle

def parens_expr_handler(tokens):
    tokens.advance()
    inner_expr = expr(tokens)
    token = next(tokens)[3]
    if token == ')':
        return inner_expr
    elif token == ',':
        imag_part = expr(tokens)
        expect(tokens, ')')
        return tokens.produce('complex', inner_expr, imag_part)
    else:
        raise ParserError(tokens, "expecting end parenthesis")

def call_handler(tokens, lhs):
    tokens.advance()
    seq = subscript_sequence(tokens)
    expect(tokens, ')')
    return tokens.produce('call', lhs, *seq[1:])

def resolve_handler(tokens, lhs):
    tokens.advance()
    rhs = id_ref(tokens)
    return tokens.produce('resolve', lhs, rhs)


_PREFIX_OP_HANDLERS = {
    "not":    prefix_op_handler( 50, 'not_'),
    "+":      prefix_op_handler(110, 'pos'),
    "-":      prefix_op_handler(110, 'neg'),
    "(":      parens_expr_handler,
    "(/":     inplace_array('(/', '/)'),
    "[":      inplace_array('[', ']')
    }

_INFIX_OP_HANDLERS = {
    "eqv":    ( 20, infix_op_handler( 20, 'eqv')),
    "neqv":   ( 20, infix_op_handler( 20, 'neqv')),
    "or":     ( 30, infix_op_handler( 30, 'or_')),
    "and":    ( 40, infix_op_handler( 40, 'and_')),
    "eq":     ( 60, infix_op_handler( 61, 'eq')),
    "ne":     ( 60, infix_op_handler( 61, 'ne')),
    "le":     ( 60, infix_op_handler( 61, 'le')),
    "lt":     ( 60, infix_op_handler( 61, 'lt')),
    "ge":     ( 60, infix_op_handler( 61, 'ge')),
    "gt":     ( 60, infix_op_handler( 61, 'gt')),
    "//":     ( 70, infix_op_handler( 71, 'concat')),
    "+":      ( 80, infix_op_handler( 81, 'plus')),
    "-":      ( 80, infix_op_handler( 81, 'minus')),
    "*":      ( 90, infix_op_handler( 91, 'mul')),
    "/":      ( 90, infix_op_handler( 91, 'div')),
    "**":     (100, infix_op_handler(100, 'pow')),
    "_":      (130, infix_op_handler(131, 'kind')),
    "%":      (140, resolve_handler),
    "(":      (140, call_handler),
    }
_INFIX_OP_HANDLERS.update({
    "==":    _INFIX_OP_HANDLERS["eq"],
    "/=":    _INFIX_OP_HANDLERS["ne"],
    "<=":    _INFIX_OP_HANDLERS["le"],
    ">=":    _INFIX_OP_HANDLERS["ge"],
    "<":     _INFIX_OP_HANDLERS["lt"],
    ">":     _INFIX_OP_HANDLERS["gt"]
    })


_PREFIX_CAT_HANDLERS = {
    lexer.CAT_STRING:     literal_handler('string'),
    lexer.CAT_FLOAT:      literal_handler('float'),
    lexer.CAT_INT:        literal_handler('int'),
    lexer.CAT_RADIX:      literal_handler('radix'),
    lexer.CAT_BOOLEAN:    literal_handler('bool'),
    lexer.CAT_CUSTOM_DOT: custom_unary_handler(120),
    lexer.CAT_WORD:       literal_handler('ref'),
    }
_INFIX_CAT_HANDLERS = {
    lexer.CAT_CUSTOM_DOT: (10, custom_binary_handler(11))
    }


def expr_handler(cat_handlers, op_handlers):
    def get_handler(cat, token):
        if cat == lexer.CAT_SYMBOLIC_OP:
            return op_handlers[token]
        elif cat == lexer.CAT_BUILTIN_DOT:
            return op_handlers[token.lower()]
        else:
            return cat_handlers[cat]
    return get_handler

expr_prefix_handler = expr_handler(_PREFIX_CAT_HANDLERS, _PREFIX_OP_HANDLERS)

expr_infix_handler = expr_handler(_INFIX_CAT_HANDLERS, _INFIX_OP_HANDLERS)

def expr(tokens, min_glue=0):
    # Get prefix
    try:
        handler = expr_prefix_handler(*tokens.peek()[2:])
    except KeyError:
        raise NoMatch()
    try:
        result = handler(tokens)

        # Cycle through appropriate infixes:
        while True:
            try:
                glue, handler = expr_infix_handler(*tokens.peek()[2:])
            except KeyError:
                return result
            if glue < min_glue:
                return result
            result = handler(tokens, result)
    except NoMatch:
        raise ParserError(tokens, "Invalid expression")

_optional_expr = optional(expr)


def lvalue(tokens):
    # lvalue is subject to stricter scrutiny, than an expression, since it
    # is used in the assignment statement.
    result = id_ref(tokens)
    try:
        while True:
            token = tokens.peek()[3]
            if token == '(':
                result = call_handler(tokens, result)
            elif token == '%':
                result = resolve_handler(tokens, result)
            else:
                return result
    except NoMatch:
        raise ParserError(tokens, "Invalid lvalue")

# -----------

@rule
def kind_selector(tokens):
    if marker(tokens, '*'):
        with LockedIn(tokens, "invalid star kind"):
            kind_ = int_(tokens)
    else:
        expect(tokens, '(')
        with LockedIn(tokens, "invalid kind selector"):
            if marker(tokens, 'kind'):
                expect(tokens, '=')
            kind_ = expr(tokens)
            expect(tokens, ')')
    return tokens.produce('kind_sel', kind_)

@rule
def keyword_arg(tokens, choices=None):
    sel = expect_cat(tokens, lexer.CAT_WORD).lower()
    expect(tokens, '=')
    if sel not in choices:
        raise NoMatch()
    return sel

@rule
def char_len(tokens):
    if marker(tokens, '*'):
        return '*'
    if marker(tokens, ':'):
        return ':'
    return expr(tokens)

@rule
def char_len_suffix(tokens):
    expect(tokens, '*')
    with LockedIn(tokens, "invalid character length suffix"):
        if marker(tokens, '('):
            len_ = char_len(tokens)
            expect(tokens, ')')
        else:
            len_ = int_(tokens)
        return len_

_optional_len_kind_kwd = optional(keyword_arg, ('len', 'kind'))

@rule
def char_selector(tokens):
    len_ = None
    kind = None

    try:
        len_ = char_len_suffix(tokens)
    except NoMatch:
        expect(tokens, '(')
        with LockedIn(tokens, "invalid character selector"):
            sel = _optional_len_kind_kwd(tokens)
            if sel == 'len' or sel is None:
                len_ = char_len(tokens)
            else:
                kind = expr(tokens)

            if marker(tokens, ','):
                sel = _optional_len_kind_kwd(tokens)
                if sel is None:
                    sel = 'kind' if kind is None else 'len'
                if sel == 'len':
                    len_ = char_len(tokens)
                else:
                    kind = expr(tokens)

            expect(tokens, ')')

    return tokens.produce('char_sel', len_, kind)

def _typename_handler(tokens):
    expect(tokens, 'type')
    expect(tokens, '(')
    with LockedIn(tokens, "invalid derived type specifier"):
        typename = identifier(tokens)
        expect(tokens, ')')
        return ('derived_type', typename)

def _class_handler(tokens):
    expect(tokens, 'class')
    expect(tokens, '(')
    with LockedIn(tokens, "invalid class specifier"):
        if marker(tokens, '*'):
            typename = None
        else:
            typename = identifier(tokens)
        expect(tokens, ')')
        return ('class_', typename)

def double_precision_type(tokens):
    expect(tokens, 'doubleprecision')
    return tokens.produce('real_type', 'double')

def double_type(tokens):
    expect(tokens, 'double')
    if marker(tokens, 'precision'):
        return tokens.produce('real_type', 'double')
    else:
        expect(tokens, 'complex')
        return tokens.produce('complex_type', 'double')

_TYPE_SPEC_HANDLERS = {
    'integer':   prefix('integer', optional(kind_selector), 'integer_type'),
    'real':      prefix('real', optional(kind_selector), 'real_type'),
    'double':    double_type,
    'doubleprecision': double_precision_type,
    'complex':   prefix('complex', optional(kind_selector), 'complex_type'),
    'character': prefix('character', optional(char_selector), 'character_type'),
    'logical':   prefix('logical', optional(kind_selector), 'logical_type'),
    'type':      _typename_handler,
    'class':     _class_handler,
    }

type_spec = prefixes(_TYPE_SPEC_HANDLERS)

_NONPARAM_TYPE_SPEC_HANDLERS = {
    'integer':   prefix('integer', null_rule, 'integer_type'),
    'real':      prefix('real', null_rule, 'real_type'),
    'complex':   prefix('complex', null_rule, 'complex_type'),
    'character': prefix('character', null_rule, 'character_type'),
    'logical':   prefix('logical', null_rule, 'logical_type'),
    }

nonparam_type_spec = prefixes(_NONPARAM_TYPE_SPEC_HANDLERS)

@rule
def lower_bound(tokens):
    lower = _optional_expr(tokens)
    expect(tokens, ':')
    return lower

optional_lower_bound = optional(lower_bound)

@rule
def dim_spec(tokens):
    lower = optional_lower_bound(tokens)
    if marker(tokens, '*'):
        # Implied dimension
        return tokens.produce('implied_dim', lower, '*')
    try:
        # Explicit dimension
        upper = expr(tokens)
        return tokens.produce('explicit_dim', lower, upper)
    except NoMatch:
        # Deferred dimension
        return tokens.produce('deferred_dim', lower, None)

dimspec_sequence = comma_sequence(dim_spec, 'shape')

@rule
def shape(tokens):
    expect(tokens, '(')
    dims = dimspec_sequence(tokens)
    expect(tokens, ')')
    return dims

@rule
def intent(tokens):
    expect(tokens, 'intent')
    with LockedIn(tokens, "invalid intent"):
        expect(tokens, '(')
        if marker(tokens, 'inout'):
            in_ = True
            out = True
        else:
            in_ = marker(tokens, 'in')
            out = marker(tokens, 'out')
            if not (in_ or out):
                raise ParserError(tokens, "null intent")
        expect(tokens, ')')
        return tokens.produce('intent', in_, out)

@rule
def bind_c(tokens):
    expect(tokens, 'bind')
    expect(tokens, '(')
    expect(tokens, 'c')
    if marker(tokens, ','):
        expect(tokens, 'name')
        expect(tokens, '=')
        name = expr(tokens)
    else:
        name = None
    expect(tokens, ')')
    return tokens.produce('bind_c', name)

_ENTITY_ATTR_HANDLERS = {
    'parameter':   tag('parameter', 'parameter'),
    'public':      tag('public', 'public'),
    'private':     tag('private', 'private'),
    'bind':        bind_c,
    'allocatable': tag('allocatable', 'allocatable'),
    'dimension':   prefix('dimension', shape, 'dimension'),
    'external':    tag('external', 'external'),
    'intent':      intent,
    'intrinsic':   tag('intrinsic', 'intrinsic'),
    'optional':    tag('optional', 'optional'),
    'pointer':     tag('pointer', 'pointer'),
    'save':        tag('save', 'save'),
    'target':      tag('target', 'target'),
    'value':       tag('value', 'value'),
    'volatile':    tag('volatile', 'volatile'),
    }

entity_attr = prefixes(_ENTITY_ATTR_HANDLERS)

@rule
def double_colon(tokens):
    # FIXME this is not great
    expect(tokens, ':')
    expect(tokens, ':')

optional_double_colon = optional(double_colon)

def attribute_sequence(attr_rule, production_tag):
    @rule
    def attribute_sequence_rule(tokens):
        attrs = []
        if marker(tokens, ','):
            with LockedIn(tokens, 'invalid attribute'):
                attrs.append(attr_rule(tokens))
                while marker(tokens, ','):
                    attrs.append(attr_rule(tokens))
                double_colon(tokens)
        else:
            optional_double_colon(tokens)
        return tokens.produce(production_tag, *attrs)

    return attribute_sequence_rule

entity_attrs = attribute_sequence(entity_attr, 'entity_attrs')

def init_assign(tokens):
    expect(tokens, '=')
    with LockedIn(tokens, "invalid assignment"):
        init = expr(tokens)
        return tokens.produce('init_assign', init)

def init_point(tokens):
    expect(tokens, '=>')
    with LockedIn(tokens, "invalid pointer assignment"):
        init = expr(tokens)
        return tokens.produce('init_point', init)

def initializer(tokens):
    try:
        return init_assign(tokens)
    except NoMatch:
        return init_point(tokens)

optional_char_len_suffix = optional(char_len_suffix)

optional_shape = optional(shape)

optional_initializer = optional(initializer)

@rule
def entity(tokens):
    name = identifier(tokens)
    shape_ = optional_shape(tokens)
    len_ = optional_char_len_suffix(tokens)
    init = optional_initializer(tokens)
    return tokens.produce('entity', name, shape_, len_, init)

entity_sequence = comma_sequence(entity, 'entity_list')

@rule
def entity_decl(tokens):
    type_ = type_spec(tokens)
    attrs_ = entity_attrs(tokens)
    entities = entity_sequence(tokens)
    eos(tokens)
    return tokens.produce('entity_decl', type_, attrs_, entities)

def extends(tokens):
    expect(tokens, 'extends')
    with LockedIn(tokens, "invalid extends"):
        expect(tokens, '(')
        name = identifier(tokens)
        expect(tokens, ')')
        return tokens.produce('extends', name)

optional_bind_c = optional(bind_c)

_TYPE_ATTR_HANDLERS = {
    'abstract':    tag('abstract', 'abstract'),
    'public':      tag('public', 'public'),
    'private':     tag('private', 'private'),
    'bind':        bind_c,
    'extends':     extends
    }

type_attr = prefixes(_TYPE_ATTR_HANDLERS)

type_attrs = attribute_sequence(type_attr, 'type_attrs')

def preproc_stmt(tokens):
    return ('preproc_stmt', expect_cat(tokens, lexer.CAT_PREPROC))

def block(inner_rule, production_tag='block'):
    def block_rule(tokens):
        stmts = []
        while True:
            cat = tokens.peek()[2]
            if cat == lexer.CAT_INT:
                tokens.advance()
            elif cat == lexer.CAT_EOS:
                tokens.advance()
            elif cat == lexer.CAT_PREPROC:
                stmts.append(preproc_stmt(tokens))
            else:
                try:
                    stmts.append(inner_rule(tokens))
                except NoMatch:
                    break
        return tokens.produce(production_tag, *stmts)

    return block_rule

component_block = block(entity_decl, 'component_block')

public_stmt = tag_stmt('public', 'public')

private_stmt = tag_stmt('private', 'private')

sequence_stmt = tag_stmt('sequence', 'sequence')

_TYPE_TAG_HANDLERS = {
    'private':     private_stmt,
    'sequence':    sequence_stmt,
    }

type_tag = prefixes(_TYPE_TAG_HANDLERS)

type_tag_block = block(type_tag, 'type_tags')

optional_identifier = optional(identifier)

def pass_attr(tokens):
    expect(tokens, 'pass')
    if marker(tokens, '('):
        ident = identifier(tokens)
        expect(tokens, ')')
    else:
        ident = None
    return tokens.produce('pass', ident)

_TYPE_PROC_ATTR_HANDLERS = {
    'deferred':        tag('deferred', 'deferred'),
    'nopass':          tag('nopass', 'nopass'),
    'non_overridable': tag('non_overridable', 'non_overridable'),
    'pass':            pass_attr,
    'public':          tag('public', 'public'),
    'private':         tag('private', 'private'),
    }

type_proc_attr = prefixes(_TYPE_PROC_ATTR_HANDLERS)

type_proc_attrs = attribute_sequence(type_proc_attr, 'type_proc_attrs')

def type_proc(tokens):
    name = identifier(tokens)
    if marker(tokens, '=>'):
        ref = identifier(tokens)
    else:
        ref = None
    return tokens.produce('type_proc', name, ref)

type_proc_sequence = comma_sequence(type_proc, 'type_proc_list')

def type_proc_decl(tokens):
    expect(tokens, 'procedure')
    with LockedIn(tokens, "invalid type-bound procedure declaration"):
        if marker(tokens, '('):
            name = identifier(tokens)
            expect(tokens, ')')
        else:
            name = None
        attrs = type_proc_attrs(tokens)
        procs = type_proc_sequence(tokens)
        eos(tokens)
        return tokens.produce('type_proc_decl', name, attrs, procs)

def generic_decl(tokens):
    expect(tokens, 'generic')
    with LockedIn(tokens, "invalid generic declaration"):
        attrs = type_proc_attrs(tokens)
        name = iface_name(tokens)
        expect(tokens, '=>')
        refs = identifier_sequence(tokens)
        eos(tokens)
        return tokens.produce('generic_decl', name, attrs, refs)

def final_decl(tokens):
    expect(tokens, 'final')
    with LockedIn(tokens, "invalid final declaration"):
        optional_double_colon(tokens)
        refs = identifier_sequence(tokens)
        eos(tokens)
        return tokens.produce('final_decl', refs)

_TYPE_CONTAINS_HANDLERS = {
    'procedure':   type_proc_decl,
    'generic':     generic_decl,
    'final':       final_decl
    }

type_contains_stmt = prefixes(_TYPE_CONTAINS_HANDLERS)

type_contains_block = block(type_contains_stmt)

optional_private_stmt = optional(private_stmt)

def optional_procedures_block(tokens):
    if marker(tokens, 'contains'):
        private = optional_private_stmt(tokens)
        conts = type_contains_block(tokens)
        return tokens.produce('type_bound_procedures', private, *conts[1:])
    else:
        return None

def end_stmt(objtype, require_type=False, name_type=None):
    comp = 'end' + objtype
    if name_type is None:
        name_type = identifier

    @rule
    def end_stmt_rule(tokens):
        if marker(tokens, 'end'):
            if not marker(tokens, objtype):
                if require_type:
                    raise NoMatch()
                eos(tokens)
                return
        elif not marker(tokens, comp):
            raise NoMatch()
        try:
            name_type(tokens)
        except NoMatch:
            pass
        eos(tokens)

    return end_stmt_rule

end_type_stmt = end_stmt('type')

@rule
def type_decl(tokens):
    expect(tokens, 'type')
    attrs = type_attrs(tokens)
    name = identifier(tokens)
    with LockedIn(tokens, "invalid type declaration"):
        eos(tokens)
        tags = type_tag_block(tokens)
        decls = component_block(tokens)
        proc = optional_procedures_block(tokens)
        end_type_stmt(tokens)
        return tokens.produce('type_decl', name, attrs, tags, decls, proc)

@rule
def rename(tokens):
    alias = identifier(tokens)
    expect(tokens, '=>')
    name = identifier(tokens)
    return tokens.produce('rename', alias, name)

rename_sequence = comma_sequence(rename, 'rename_list')

@rule
def bracketed_oper(tokens):
    expect(tokens, '(')
    cat, token = next(tokens)[2:]
    if cat == lexer.CAT_CUSTOM_DOT:
        oper = tokens.produce('custom_op', token)
    elif cat == lexer.CAT_SYMBOLIC_OP or lexer.CAT_BUILTIN_DOT:
        oper = token
    expect(tokens, ')')
    return oper

@rule
def bracketed_slashes(tokens):
    # It is impossible for the lexer to disambiguate between an empty
    # in-place array (//) and bracketed slashes, so we handle it here:
    bracketed_slash_re = re.compile(r'\((//?)\)')
    tokstr = ''
    for _ in range(3):
        token = next(tokens)[3]
        tokstr += token
        match = bracketed_slash_re.match(tokstr)
        if match:
            return match.group(1)
    else:
        raise NoMatch()

@rule
def oper_spec(tokens):
    if marker(tokens, 'assignment'):
        expect(tokens, '(')
        expect(tokens, '=')
        expect(tokens, ')')
        oper = '='
    else:
        expect(tokens, 'operator')
        try:
            oper = bracketed_oper(tokens)
        except NoMatch:
            oper = bracketed_slashes(tokens)
    return tokens.produce('oper_spec', oper)

def iface_name(tokens):
    try:
        return oper_spec(tokens)
    except NoMatch:
        return identifier(tokens)

optional_iface_name = optional(iface_name)

@rule
def rename_oper(tokens):
    local_op = oper_spec(tokens)
    expect(tokens, '=>')
    use_op = oper_spec(tokens)
    return tokens.produce('use_symbol', local_op, use_op)

@rule
def rename_identifier(tokens):
    local_id = identifier(tokens)
    expect(tokens, '=>')
    use_id = identifier(tokens)
    return tokens.produce('use_symbol', local_id, use_id)

def rename_clause(tokens):
    try:
        return rename_oper(tokens)
    except NoMatch:
        return rename_identifier(tokens)

rename_sequence = comma_sequence(rename_clause, 'rename_list', allow_empty=False)

def only_item(tokens):
    try:
        return rename_clause(tokens)
    except NoMatch:
        name = identifier(tokens)
        return tokens.produce('use_symbol', None, name)

only_sequence = comma_sequence(only_item, 'only_list', allow_empty=True)

_USE_ATTR_HANDLERS = {
    'intrinsic':     tag('intrinsic',     'intrinsic'),
    'non_intrinsic': tag('non_intrinsic', 'non_intrinsic'),
    }

use_attr = prefixes(_USE_ATTR_HANDLERS)

use_attrs = attribute_sequence(use_attr, 'use_attrs')

@rule
def use_stmt(tokens):
    expect(tokens, 'use')
    with LockedIn(tokens, "invalid use statement"):
        attrs = use_attrs(tokens)
        name = identifier(tokens)
        if marker(tokens, ','):
            if marker(tokens, 'only'):
                expect(tokens, ':')
                clauses = only_sequence(tokens)
                only = "only"
            else:
                clauses = rename_sequence(tokens)
                only = None
        else:
            clauses = []
            only = None

        eos(tokens)
        return tokens.produce('use_stmt', name, attrs, only, *clauses[1:])

_letter_re = re.compile(r'^[a-zA-Z]$')

@rule
def letter_range(tokens):
    def letter():
        cand = next(tokens)[3]
        if _letter_re.match(cand):
            return cand.lower()
        else:
            raise NoMatch()

    start = letter()
    end = start
    if marker(tokens, '-'):
        end = letter()
    return tokens.produce('letter_range', start, end)

letter_range_sequence = comma_sequence(letter_range, 'letter_range_list')

@rule
def implicit_spec_param(tokens):
    type_ = type_spec(tokens)
    expect(tokens, '(')
    ranges = letter_range_sequence(tokens)
    expect(tokens, ')')
    return tokens.produce('implicit_spec', type_, ranges)

@rule
def implicit_spec_nonparam(tokens):
    # This needs to be here because otherwise, IMPLICIT INTEGER(a-z) is
    # interpreted as an INTEGER of kind (a-z), and the parsing fails when
    # trying to read the letter range.
    type_ = nonparam_type_spec(tokens)
    expect(tokens, '(')
    ranges = letter_range_sequence(tokens)
    expect(tokens, ')')
    return tokens.produce('implicit_spec', type_, ranges)

def implicit_spec(tokens):
    try:
        return implicit_spec_param(tokens)
    except NoMatch:
        return implicit_spec_nonparam(tokens)

implicit_spec_sequence = comma_sequence(implicit_spec, 'implicit_decl')

@rule
def implicit_stmt(tokens):
    expect(tokens, 'implicit')
    with LockedIn(tokens, "invalid implicit statement"):
        if marker(tokens, 'none'):
            eos(tokens)
            return tokens.produce('implicit_none')
        else:
            specs = implicit_spec_sequence(tokens)
            eos(tokens)
            return specs

def dummy_arg(tokens):
    if marker(tokens, '*'):
        return '*'
    else:
        return identifier(tokens)

dummy_arg_sequence = comma_sequence(dummy_arg, 'arg_list', allow_empty=True)

_SUB_PREFIX_HANDLERS = {
    'impure':    tag('impure', 'impure'),
    'pure':      tag('pure', 'pure'),
    'recursive': tag('recursive', 'recursive'),
    }

sub_prefix = prefixes(_SUB_PREFIX_HANDLERS)

sub_prefix_sequence = ws_sequence(sub_prefix, 'sub_prefix_list')

def optional_contained_part(tokens):
    # contains statement
    if marker(tokens, 'contains'):
        with LockedIn(tokens, "invalid declaration in contained part"):
            eos(tokens)
            return contained_block(tokens)
    else:
        return tokens.produce('contained_block')

end_subroutine_stmt = end_stmt('subroutine', require_type=False)

@rule
def subroutine_decl(tokens):
    # Header
    prefixes_ = sub_prefix_sequence(tokens)
    expect(tokens, 'subroutine')
    with LockedIn(tokens, "invalid subroutine declaration"):
        name = identifier(tokens)
        if marker(tokens, '('):
            args = dummy_arg_sequence(tokens)
            expect(tokens, ')')
        else:
            args = tokens.produce('arg_list')   # empty args
        bind_ = optional_bind_c(tokens)
        eos(tokens)

    with LockedIn(tokens, "malformed statement inside subroutine"):
        # Body
        declarations_ = declaration_part(tokens)
        execution_part(tokens)
        optional_contained_part(tokens)

        # Footer
        end_subroutine_stmt(tokens)
        return tokens.produce('subroutine_decl', name, prefixes_, args, bind_,
                              declarations_)


_FUNC_PREFIX_HANDLERS = {
    'elemental': tag('elemental', 'elemental'),
    'impure':    tag('impure', 'impure'),
    'pure':      tag('pure', 'pure'),
    'recursive': tag('recursive', 'recursive'),
    }

func_modifier = prefixes(_FUNC_PREFIX_HANDLERS)

def func_prefix(tokens):
    try:
        return func_modifier(tokens)
    except NoMatch:
        return type_spec(tokens)

func_prefix_sequence = ws_sequence(func_prefix, 'func_prefix_list')

@rule
def result_suffix(tokens):
    expect(tokens, 'result')
    expect(tokens, '(')
    res = identifier(tokens)
    expect(tokens, ')')
    return ('result', res)

def func_suffix(tokens):
    try:
        return result_suffix(tokens)
    except NoMatch:
        return bind_c(tokens)

func_suffix_sequence = ws_sequence(func_suffix, 'func_suffix_list')

func_arg_sequence = comma_sequence(identifier, 'arg_list', allow_empty=True)

end_function_stmt = end_stmt('function', require_type=False)

@rule
def function_decl(tokens):
    # Header
    prefixes_ = func_prefix_sequence(tokens)
    expect(tokens, 'function')
    with LockedIn(tokens, "invalid function declaration"):
        name = identifier(tokens)
        expect(tokens, '(')
        args = func_arg_sequence(tokens)
        expect(tokens, ')')
        suffixes = func_suffix_sequence(tokens)
        eos(tokens)

    with LockedIn(tokens, "malformed statement inside function"):
        # Body
        declarations_ = declaration_part(tokens)
        execution_part(tokens)
        optional_contained_part(tokens)

        # Footer
        end_function_stmt(tokens)
        return tokens.produce('function_decl', name, prefixes_, args, suffixes,
                              declarations_)

def subprogram_decl(tokens):
    try:
        return subroutine_decl(tokens)
    except NoMatch:
        return function_decl(tokens)

contained_block = block(subprogram_decl, 'contained_block')

identifier_sequence = comma_sequence(identifier, 'identifier_list')

@rule
def module_proc_stmt(tokens):
    expect(tokens, 'module')
    expect(tokens, 'procedure')
    with LockedIn(tokens, "invalid module procedure statement"):
        optional_double_colon(tokens)
        procs = identifier_sequence(tokens)
        eos(tokens)
        return tokens.produce('module_proc_stmt', *procs[1:])

def interface_body_stmt(tokens):
    try:
        return module_proc_stmt(tokens)
    except NoMatch:
        return subprogram_decl(tokens)

interface_body_block = block(interface_body_stmt, 'interface_body')

end_interface_stmt = end_stmt('interface', name_type=iface_name)

@rule
def interface_decl(tokens):
    expect(tokens, 'interface')
    with LockedIn(tokens, "invalid interface declaration"):
        name = optional_iface_name(tokens)
        eos(tokens)
        decls = interface_body_block(tokens)
        end_interface_stmt(tokens)
        return tokens.produce('interface_decl', name, decls)

@rule
def abstract_interface_decl(tokens):
    expect(tokens, 'abstract')
    expect(tokens, 'interface')
    with LockedIn(tokens, "invalid abstract interface declaration"):
        eos(tokens)
        decls = interface_body_block(tokens)
        end_interface_stmt(tokens)
        return tokens.produce('abstract_interface_decl', decls)

def imbue_stmt(prefix_rule, object_rule):
    object_sequence = comma_sequence(object_rule, None)
    def imbue_stmt_rule(tokens):
        prefix_ = prefix_rule(tokens)
        with LockedIn(tokens, "invalid imbue statement"):
            optional_double_colon(tokens)
            vars_ = object_sequence(tokens)
            eos(tokens)
            return tokens.produce('imbue', prefix_, *vars_[1:])
    return imbue_stmt_rule

# TODO: one can also save common blocks
imbue_save_stmt = imbue_stmt(tag('save', 'save'), identifier)

@rule
def save_all_stmt(tokens):
    expect(tokens, 'save')
    eos(tokens)
    return tokens.produce('save_all')

def save_stmt(tokens):
    try:
        return save_all_stmt(tokens)
    except NoMatch:
        return imbue_save_stmt(tokens)

@rule
def param_init(tokens):
    name = identifier(tokens)
    init = initializer(tokens)
    return tokens.produce('param_init', name, init)

param_init_sequence = comma_sequence(param_init, 'parameter_stmt')

@rule
def parameter_stmt(tokens):
    expect(tokens, 'parameter')
    with LockedIn(tokens, "invalid parameter statement"):
        expect(tokens, '(')
        seq = param_init_sequence(tokens)
        expect(tokens, ')')
        eos(tokens)
    return seq

@rule
def dimension_stmt_spec(tokens):
    id_ = identifier(tokens)
    shape_ = shape(tokens)
    return tokens.produce('dimension_stmt_spec', id_, shape_)

dimension_stmt_list = comma_sequence(dimension_stmt_spec, 'dimension_stmt')

@rule
def dimension_stmt(tokens):
    expect(tokens, 'dimension')
    optional_double_colon(tokens)
    specs_ = dimension_stmt_list(tokens)
    eos(tokens)
    return specs_

equivalence_object_sequence = comma_sequence(lvalue, 'equivalence_set')

@rule
def common_name(tokens):
    expect(tokens, '/')
    name = optional_identifier(tokens)
    expect(tokens, '/')
    return tokens.produce('common_name', name)

def optional_common_name(tokens):
    try:
        return common_name(tokens)
    except NoMatch:
        return tokens.produce('common_name', None)

@rule
def common_ref(tokens):
    name = identifier(tokens)
    shape_ = optional_shape(tokens)
    return tokens.produce('common_ref', name, shape_)

@rule
def next_common_ref(tokens):
    expect(tokens, ',')
    return common_ref(tokens)

def common_ref_sequence(tokens):
    vals = [common_ref(tokens)]
    try:
        while True:
            vals.append(next_common_ref(tokens))
    except NoMatch:
        return tokens.produce('common_ref_list', *vals)

@rule
def common_stmt(tokens):
    expect(tokens, 'common')
    name = optional_common_name(tokens)
    refs = common_ref_sequence(tokens)
    blocks = [tokens.produce('common_block', name[1], *refs[1:])]
    try:
        while True:
            if marker(tokens, ','):
                with LockedIn(tokens, 'expecting common block'):
                    name = common_name(tokens)
                    refs = common_ref_sequence(tokens)
            else:
                name = common_name(tokens)
                with LockedIn(tokens, 'expecting common block references'):
                    refs = common_ref_sequence(tokens)
            blocks.append(tokens.produce('common_block', name[1], *refs[1:]))
    except NoMatch:
        pass
    eos(tokens)
    return tokens.produce('common_stmt', *blocks)

@rule
def equivalence_set(tokens):
    expect(tokens, '(')
    seq = equivalence_object_sequence(tokens)
    expect(tokens, ')')
    return seq

equivalence_set_sequence = comma_sequence(equivalence_set, 'equivalence_stmt')

@rule
def equivalence_stmt(tokens):
    expect(tokens, 'equivalence')
    with LockedIn(tokens, "invalid equivalence statement"):
        seq = equivalence_set_sequence(tokens)
        eos(tokens)
        return seq

public_imbue_stmt = imbue_stmt(tag('public', 'public'), iface_name)

def public_or_imbue_stmt(tokens):
    try:
        return public_stmt(tokens)
    except NoMatch:
        return public_imbue_stmt(tokens)

private_imbue_stmt = imbue_stmt(tag('private', 'private'), iface_name)

def private_or_imbue_stmt(tokens):
    try:
        return private_stmt(tokens)
    except NoMatch:
        return private_imbue_stmt(tokens)


_PROC_ATTR_HANDLERS = {
    'public':      tag('public', 'public'),
    'private':     tag('private', 'private'),
    'bind':        bind_c,
    'intent':      intent,
    'intrinsic':   tag('intrinsic', 'intrinsic'),
    'optional':    tag('optional', 'optional'),
    'pointer':     tag('pointer', 'pointer'),
    'nopass':      tag('nopass', 'nopass'),
    'pass':        pass_attr,
    }

proc_attr = prefixes(_PROC_ATTR_HANDLERS)

proc_attrs = attribute_sequence(proc_attr, 'proc_attrs')

optional_init_point = optional(init_point)

@rule
def procedure(tokens):
    name = identifier(tokens)
    init = optional_init_point(tokens)
    return tokens.produce('procedure', name, init)

procedure_sequence = comma_sequence(procedure, 'procedure_list')

def procedure_decl(tokens):
    expect(tokens, 'procedure')
    expect(tokens, '(')
    iface = optional_identifier(tokens)
    expect(tokens, ')')
    attrs = proc_attrs(tokens)
    procs = procedure_sequence(tokens)
    return tokens.produce('procedure_decl', iface, attrs, procs)

def ignore_stmt(tokens):
    while True:
        cat = next(tokens)[2]
        if cat == lexer.CAT_EOS:
            return

def data_stmt(tokens):
    expect(tokens, 'data')
    ignore_stmt(tokens)
    return tokens.produce('data_stmt')

def derived_type_decl_or_entity(tokens):
    try:
        return entity_decl(tokens)
    except NoMatch:
        return type_decl(tokens)

# TODO: some imbue statements are missing here.
_DECLARATION_HANDLERS = {
    'use':         use_stmt,
    'implicit':    implicit_stmt,
    'abstract':    abstract_interface_decl,
    'interface':   interface_decl,
    'equivalence': equivalence_stmt,
    'procedure':   procedure_decl,
    'common':      common_stmt,

    'dimension':   dimension_stmt,
    'data':        data_stmt,
    'public':      public_or_imbue_stmt,
    'private':     private_or_imbue_stmt,
    'parameter':   parameter_stmt,
    'external':    imbue_stmt(tag('external', 'external'), identifier),
    'intent':      imbue_stmt(intent, identifier),
    'intrinsic':   imbue_stmt(tag('intrinsic', 'intrinsic'), identifier),
    'optional':    imbue_stmt(tag('optional', 'optional'), identifier),
    'save':        save_stmt,
    }

# Entity declarations begin with a type, but 'type' may also be a derived
# type definition
_DECLARATION_HANDLERS.update(
    {prefix: entity_decl for prefix in _TYPE_SPEC_HANDLERS})
_DECLARATION_HANDLERS['type'] = derived_type_decl_or_entity

declaration_stmt = prefixes(_DECLARATION_HANDLERS)

declaration_part = block(declaration_stmt, 'declaration_block')

@rule
def construct_tag(tokens):
    expect_cat(tokens, lexer.CAT_WORD)
    expect(tokens, ':')
    token = tokens.peek()[3]
    return token.lower()

optional_construct_tag = optional(construct_tag)

@rule
def if_clause(tokens):
    expect(tokens, 'if')
    expect(tokens, '(')
    expr(tokens)
    expect(tokens, ')')

else_if = composite('else', 'if')

@rule
def else_if_block(tokens):
    else_if(tokens)
    expect(tokens, '(')
    expr(tokens)
    expect(tokens, ')')
    optional_identifier(tokens)
    eos(tokens)
    with LockedIn(tokens, "invalid else-if block"):
        execution_part(tokens)

else_if_block_sequence = ws_sequence(else_if_block, 'else_if_sequence')

@rule
def else_block(tokens):
    expect(tokens, 'else')
    optional_identifier(tokens)
    eos(tokens)
    execution_part(tokens)

optional_else_block = optional(else_block)


end_if_stmt = end_stmt('if')

@rule
def if_construct(tokens):
    optional_construct_tag(tokens)
    if_clause(tokens)
    with LockedIn(tokens, "invalid if construct"):
        if marker(tokens, 'then'):
            eos(tokens)
            execution_part(tokens)
            else_if_block_sequence(tokens)
            optional_else_block(tokens)
            end_if_stmt(tokens)
        else:
            ignore_stmt(tokens)

@rule
def while_ctrl(tokens):
    expect(tokens, 'while')
    expect(tokens, '(')
    expr(tokens)
    expect(tokens, ')')

def loop_ctrl(tokens):
    try:
        while_ctrl(tokens)
    except NoMatch:
        do_ctrl(tokens)

optional_loop_ctrl = optional(loop_ctrl)

end_do_stmt = end_stmt('do')

def nonblock_do_block(tokens, lineno_stack):
    try:
        while True:
            cat, token = tokens.peek()[2:]
            if cat == lexer.CAT_INT:
                lineno = int(token)
                if lineno in lineno_stack:
                    break          # We found a terminating token
                tokens.advance()
            elif cat == lexer.CAT_EOS:
                tokens.advance()
            elif cat == lexer.CAT_PREPROC:
                preproc_stmt(tokens)
            else:
                try:
                    do_construct(tokens, lineno_stack)
                except NoMatch:
                    execution_stmt(tokens)
    except NoMatch:
        raise ParserError(tokens, "invalid statement in non-block do")

    # We have found the terminating statement, if it is further down the
    # stack we have mismatched blocks
    top_lineno = lineno_stack.pop()
    if top_lineno != lineno:
        raise ParserError(tokens, "non-block do blocks do not nest")

    # Fortran allows the terminating label to be shared by multiple do blocks.
    # In this case, we do not consume it since it must be used later.
    if lineno in lineno_stack:
        return

    # In other cases, we need to consume the end since it might be a "stray"
    # end do statement
    tokens.advance()
    try:
        end_do_stmt(tokens)
    except NoMatch:
        try:
            execution_stmt(tokens)
        except NoMatch:
            raise ParserError(tokens, "invalid end of non-block do")

@rule
def do_construct(tokens, lineno_stack=None):
    optional_construct_tag(tokens)
    expect(tokens, 'do')
    with LockedIn(tokens, "invalid do construct"):
        try:
            until_lineno = int(int_(tokens)[1])
        except NoMatch:
            # BLOCK DO CONSTRUCT
            marker(tokens, ',')
            optional_loop_ctrl(tokens)
            eos(tokens)
            execution_part(tokens)
            end_do_stmt(tokens)
        else:
            # NONBLOCK DO CONSTRUCT
            if lineno_stack is None:
                lineno_stack = []
            lineno_stack.append(until_lineno)

            marker(tokens, ',')
            optional_loop_ctrl(tokens)
            eos(tokens)
            nonblock_do_block(tokens, lineno_stack)

@rule
def case_slice(tokens):
    _optional_expr(tokens)
    expect(tokens, ':')
    _optional_expr(tokens)

def case_range(tokens):
    try:
        case_slice(tokens)
    except NoMatch:
        expr(tokens)

case_range_sequence = comma_sequence(case_range, 'case_range_list')

@rule
def select_case(tokens):
    expect(tokens, 'case')
    if marker(tokens, 'default'):
        pass
    else:
        expect(tokens, '(')
        case_range_sequence(tokens)
        expect(tokens, ')')
    eos(tokens)
    execution_part(tokens)

select_case_sequence = block(select_case, 'select_case_list')

end_select_stmt = end_stmt('select')

select_case_tag = composite('select', 'case')

@rule
def select_case_construct(tokens):
    optional_construct_tag(tokens)
    select_case_tag(tokens)
    with LockedIn(tokens, "invalid select case construct"):
        expect(tokens, '(')
        expr(tokens)
        expect(tokens, ')')
        eos(tokens)
        select_case_sequence(tokens)
        end_select_stmt(tokens)

@rule
def type_prefix(tokens):
    type_spec(tokens)
    double_colon(tokens)

optional_type_prefix = optional(type_prefix)

@rule
def forall_select(tokens):
    identifier(tokens)
    expect(tokens, '=')
    expr(tokens)
    expect(tokens, ':')
    expr(tokens)
    if marker(tokens, ':'):
        expr(tokens)

@rule
def forall_clause(tokens):
    expect(tokens, 'forall')
    with LockedIn(tokens, "invalid forall clause"):
        expect(tokens, '(')
        optional_type_prefix(tokens)
        forall_select(tokens)
        while marker(tokens, ','):
            try:
                forall_select(tokens)
            except NoMatch:
                expr(tokens)
                break
        expect(tokens, ')')

end_forall_stmt = end_stmt('forall')

@rule
def forall_construct(tokens):
    optional_construct_tag(tokens)
    forall_clause(tokens)
    try:
        eos(tokens)
    except NoMatch:
        # FORALL STMT
        ignore_stmt(tokens)
    else:
        # FORALL BLOCK
        execution_part(tokens)
        end_forall_stmt(tokens)

@rule
def where_clause(tokens):
    expect(tokens, 'where')
    expect(tokens, '(')
    expr(tokens)
    expect(tokens, ')')

end_where_stmt = end_stmt('where')

else_where = composite('else', 'where')

@rule
def else_where_clause(tokens):
    else_where(tokens)
    if marker(tokens, '('):
        expr(tokens)
        expect(tokens, ')')
    optional_identifier(tokens)
    eos(tokens)

@rule
def where_construct(tokens):
    optional_construct_tag(tokens)
    where_clause(tokens)
    with LockedIn(tokens, "invalid where construct"):
        try:
            eos(tokens)
        except NoMatch:
            # WHERE STMT
            ignore_stmt(tokens)
        else:
            # WHERE BLOCK
            execution_part(tokens)
            while matches(else_where_clause, tokens):
                execution_part(tokens)
            end_where_stmt(tokens)

def fast_end_handler(tokens):
    # In traditional recursive descent parsing of a block, we would try to
    # match every possible statement before deciding the block is complete
    # and trying to match an end statement.  To speed this up, we raise
    # `EndOfBlock` here whenever we encounter a valid "END something"
    begin_pos = tokens.pos
    tokens.advance()
    try:
        cat, token = tokens.peek()[2:]
        if cat == lexer.CAT_WORD or cat == lexer.CAT_EOS:
            raise EndOfBlock()
        else:
            raise NoMatch()
    finally:
        tokens.pos = begin_pos

CONSTRUCT_HANDLERS = {
    'if':         if_construct,
    'do':         do_construct,
    'select':     select_case_construct,
    'forall':     forall_construct,
    'where':      where_construct
    }

construct = prefixes(CONSTRUCT_HANDLERS)

STMT_HANDLERS = {
    'allocate':   ignore_stmt,
    'assign':     ignore_stmt,
    'backspace':  ignore_stmt,
    'call':       ignore_stmt,
    'continue':   ignore_stmt,
    'cycle':      ignore_stmt,
    'close':      ignore_stmt,
    'deallocate': ignore_stmt,
    'endfile':    ignore_stmt,
    'entry':      ignore_stmt,
    'exit':       ignore_stmt,
    'flush':      ignore_stmt,
    'format':     ignore_stmt,
    'go':         ignore_stmt,
    'goto':       ignore_stmt,
    'inquire':    ignore_stmt,
    'nullify':    ignore_stmt,
    'open':       ignore_stmt,
    'pause':      ignore_stmt,
    'print':      ignore_stmt,
    'return':     ignore_stmt,
    'stop':       ignore_stmt,
    'read':       ignore_stmt,
    'rewind':     ignore_stmt,
    'write':      ignore_stmt,

    # placement in execution part is discouraged, but occasionally used
    'data':       data_stmt,

    # use fast end handlers
    'end':           fast_end_handler,
    'endif':         fast_end_handler,
    'enddo':         fast_end_handler,
    'endwhere':      fast_end_handler,
    'endselect':     fast_end_handler,
    'endsubroutine': fast_end_handler,
    'endfunction':   fast_end_handler,
    'contains':      fast_end_handler
    }
STMT_HANDLERS.update(CONSTRUCT_HANDLERS)

prefixed_stmt = prefixes(STMT_HANDLERS)

@rule
def assignment_stmt(tokens):
    lvalue(tokens)
    oper = next(tokens)[3]
    if oper != '=' and oper != '=>':
        raise NoMatch()
    #ignore_stmt(tokens)
    with LockedIn(tokens, "invalid assignment"):
        expr(tokens)
        eos(tokens)

@rule
def tagged_construct(tokens):
    construct_tag(tokens)
    construct(tokens)

def execution_stmt(tokens):
    try:
        prefixed_stmt(tokens)
    except NoMatch:
        try:
            assignment_stmt(tokens)
        except NoMatch:
            tagged_construct(tokens)
    except EndOfBlock:
        raise NoMatch()

execution_part = block(execution_stmt, 'execution_block')

end_module_stmt = end_stmt('module', require_type=False)

@rule
def module_decl(tokens):
    expect(tokens, 'module')
    with LockedIn(tokens, "invalid module declaration"):
        name = identifier(tokens)
        eos(tokens)

    with LockedIn(tokens, "malformed statement inside module"):
        decls = declaration_part(tokens)
        cont = optional_contained_part(tokens)
        end_module_stmt(tokens)
        return tokens.produce('module_decl', name, decls, cont)

end_program_stmt = end_stmt('program', require_type=False)

@rule
def program_decl(tokens):
    expect(tokens, 'program')
    with LockedIn(tokens, "invalid program declaration"):
        name = identifier(tokens)
        eos(tokens)

    with LockedIn(tokens, "malformed statement inside program"):
        decls = declaration_part(tokens)
        execution_part(tokens)
        cont = optional_contained_part(tokens)
        end_program_stmt(tokens)
        return tokens.produce('program_decl', name, decls, cont)

block_data = composite('block', 'data')

@rule
def end_block_data_comp(tokens):
    expect(tokens, 'endblockdata')
    optional_identifier(tokens)
    eos(tokens)

@rule
def end_block_data_sep(tokens):
    expect(tokens, 'end')
    if matches(block_data, tokens):
        optional_identifier(tokens)
    eos(tokens)

def end_block_data_stmt(tokens):
    try:
        end_block_data_comp(tokens)
    except NoMatch:
        end_block_data_sep(tokens)

@rule
def block_data_decl(tokens):
    block_data(tokens)
    ident = optional_identifier(tokens)
    eos(tokens)
    with LockedIn(tokens, "invalid statement inside block data"):
        decls = declaration_part(tokens)
        end_block_data_stmt(tokens)
        return tokens.produce('block_data_decl', ident, *decls[1:])

_PROGRAM_UNIT_HANDLERS = {
    'program':    program_decl,
    'module':     module_decl,
    'block':      block_data_decl,
    'blockdata':  block_data_decl,
    }

prefixed_program_unit = prefixes(_PROGRAM_UNIT_HANDLERS)

def program_unit(tokens):
    try:
        return prefixed_program_unit(tokens)
    except NoMatch:
        return subprogram_decl(tokens)

program_unit_sequence = block(program_unit, 'program_unit_list')


def compilation_unit(tokens):
    with LockedIn(tokens, "expecting module or (sub-)program"):
        units = program_unit_sequence(tokens)
        expect_cat(tokens, lexer.CAT_DOLLAR)

    version = tokens.produce('ast_version', *map(str, __version_tuple__))
    fname = tokens.produce('filename', tokens.fname)
    return tokens.produce('compilation_unit', version, fname, *units[1:])
