# Copyright 2019 Markus Wallerberger
# Released under the GNU Lesser General Public License, Version 3 only.
# See LICENSE.txt for permissions on usage, modification and distribution
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

from . import lexer

class NoMatch(Exception):
    "Current rule does not match, try next one if available."
    pass

class ParserError(RuntimeError):
    "Current rule does not match even though it should, fail meaningfully"
    def __init__(self, tokens, msg):
        RuntimeError.__init__(self, "parsing error: %s\nnext tokens:%s"
                              % (msg, tokens.tokens[tokens.pos:tokens.pos+20]))

# CONTEXT

class TokenStream:
    def __init__(self, tokens, pos=0):
        if isinstance(tokens, lexer._string_like_types):
            tokens = lexer.lex_snippet(tokens)

        self.tokens = tuple(tokens)
        self.pos = pos
        self.stack = []

    def peek(self):
        return self.tokens[self.pos]

    def advance(self):
        self.pos += 1

    def __iter__(self):
        return self

    def __next__(self):
        pos = self.pos
        self.pos += 1
        return self.tokens[pos]

    next = __next__       # Python 2

    def push(self):
        self.stack.append(self.pos)

    def backtrack(self):
        self.pos = self.stack.pop()

    def commit(self):
        self.stack.pop()

    def produce(self, rule, *args):
        return (rule,) + args

    # FIXME: compatibility - remove

    def expect(self, expected):
        cat, token = self.peek()
        if token.lower() != expected:
            raise NoMatch()
        self.advance()

    def expect_cat(self, expected):
        cat, token = self.peek()
        if cat != expected:
            raise NoMatch()
        return next(self)[1]

    def next_is(self, expected):
        cat, token = self.peek()
        return token.lower() == expected

    def marker(self, expected):
        if self.next_is(expected):
            self.advance()
            return True
        else:
            return False

def comma_sequence(rule, production_tag, allow_empty=False):
    def comma_sequence_rule(tokens):
        vals = []
        try:
            vals.append(rule(tokens))
        except NoMatch:
            if allow_empty:
                return tokens.produce(production_tag)
            raise
        try:
            while tokens.marker(','):
                vals.append(rule(tokens))
        except NoMatch:
            raise ParserError(tokens, "Expecting item in comma-separated list")
        return tokens.produce(production_tag, *vals)

    return comma_sequence_rule

def ws_sequence(rule, production_tag):
    def ws_sequence_rule(tokens):
        items = []
        try:
            while True:
                items.append(rule(tokens))
        except NoMatch:
            return tokens.produce(production_tag, *items)

    return ws_sequence_rule

def optional(rule, *args):
    def optional_rule(tokens):
        try:
            return rule(tokens, *args)
        except NoMatch:
            return None

    return optional_rule

def tag(expected, production_tag):
    @rule
    def tag_rule(tokens):
        tokens.expect(expected)
        return (production_tag,)

    return tag_rule

def tag_stmt(expected, production_tag):
    @rule
    def tag_rule(tokens):
        tokens.expect(expected)
        eos(tokens)
        return (production_tag,)

    return tag_rule

def prefix(expected, my_rule, production_tag):
    @rule
    def prefix_rule(tokens):
        tokens.expect(expected)
        value = my_rule(tokens)
        return (production_tag, value)

    return prefix_rule

def prefixes(handlers):
    def prefixes_rule(tokens):
        cat, token = tokens.peek()
        try:
            handler = handlers[token.lower()]
        except KeyError:
            raise NoMatch()
        return handler(tokens)

    return prefixes_rule

def rule(fn):
    def rule_setup(tokens, *args):
        tokens.push()
        try:
            value = fn(tokens, *args)
        except:
            tokens.backtrack()
            raise
        else:
            tokens.commit()
            return value
    return rule_setup

def composite(word1, word2):
    comp = word1 + word2
    @rule
    def composite_rule(tokens):
        if tokens.marker(word1):
            tokens.expect(word2)
        else:
            tokens.expect(comp)

    return composite_rule

def eos(tokens):
    return tokens.expect_cat(lexer.CAT_EOS)

class LockedIn:
    def __init__(self, tokens):
        self.tokens = tokens

    def __enter__(self): pass

    def __exit__(self, exc_type, exc_val, traceback):
        if exc_type is NoMatch:
            raise ParserError(self.tokens, "Expecting token")

def int_(tokens):
    return tokens.produce('int', tokens.expect_cat(lexer.CAT_INT))

def string_(tokens):
    return tokens.produce('string', tokens.expect_cat(lexer.CAT_STRING))

def identifier(tokens):
    return tokens.produce('id', tokens.expect_cat(lexer.CAT_WORD))

def id_ref(tokens):
    return tokens.produce('ref', tokens.expect_cat(lexer.CAT_WORD))

def custom_op(tokens):
    return tokens.produce('custom_op', tokens.expect_cat(lexer.CAT_CUSTOM_DOT))

@rule
def do_ctrl(tokens):
    dovar = identifier(tokens)
    tokens.expect('=')
    start = expr(tokens)
    tokens.expect(',')
    stop = expr(tokens)
    if tokens.marker(','):
        step = expr(tokens)
    else:
        step = None
    return tokens.produce('do_ctrl', dovar, start, stop, step)

@rule
def implied_do(tokens):
    args = []
    tokens.expect('(')
    args.append(expr(tokens))
    tokens.expect(',')
    with LockedIn(tokens):
        while True:
            try:
                do_ctrl_result = do_ctrl(tokens)
            except NoMatch:
                args.append(expr(tokens))
                tokens.expect(',')
            else:
                tokens.expect(')')
                return tokens.produce('impl_do', do_ctrl_result, *args)

def inplace_array(open_delim, close_delim):
    @rule
    def inplace_array_rule(tokens):
        seq = []
        tokens.expect(open_delim)
        with LockedIn(tokens):
            if tokens.marker(close_delim):
                return tokens.produce('array')
            while True:
                try:
                    seq.append(implied_do(tokens))
                except NoMatch:
                    seq.append(expr(tokens))
                if tokens.marker(close_delim):
                    return tokens.produce('array', *seq)
                tokens.expect(',')

    return inplace_array_rule

@rule
def slice_(tokens):
    slice_begin = _optional_expr(tokens)
    tokens.expect(':')
    with LockedIn(tokens):
        slice_end = _optional_expr(tokens)
        if tokens.marker(":"):
            slice_stride = expr(tokens)
        else:
            slice_stride = None
        return tokens.produce('slice', slice_begin, slice_end, slice_stride)

@rule
def key_prefix(tokens):
    key = identifier(tokens)
    tokens.expect('=')
    return key

optional_key_prefix = optional(key_prefix)

@rule
def argument(tokens):
    key = optional_key_prefix(tokens)
    value = expr(tokens)
    return tokens.produce('arg', key, value)

def subscript_arg(tokens):
    try:
        return slice_(tokens)
    except NoMatch:
        return argument(tokens)

subscript_sequence = comma_sequence(subscript_arg, 'sub_list', allow_empty=True)

def lvalue(tokens):
    # lvalue is subject to stricter scrutiny, than an expression, since it
    # is used in the assignment statement.
    result = id_ref(tokens)
    with LockedIn(tokens):
        while True:
            if tokens.marker('('):
                seq = subscript_sequence(tokens)
                tokens.expect(')')
                result = tokens.produce('call', result, *seq[1:])
            if tokens.marker('%'):
                dependant = id_ref(tokens)
                result = tokens.produce('resolve', result, dependant)
            else:
                break
        return result

def prefix_op_handler(subglue, action, custom=False):
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
        return tokens.produce(action, next(tokens)[1])
    return literal_handle

def parens_expr_handler(tokens):
    tokens.advance()
    inner_expr = expr(tokens)
    if tokens.marker(','):
        imag_part = expr(tokens)
        tokens.expect(')')
        return tokens.produce('complex', inner_expr, imag_part)
    else:
        tokens.expect(')')
        return tokens.produce('parens', inner_expr)

def call_handler(tokens, lhs):
    tokens.advance()
    seq = subscript_sequence(tokens)
    tokens.expect(')')
    return tokens.produce('call', lhs, *seq[1:])

class ExpressionHandler:
    def __init__(self):
        prefix_ops = {
            "not":    prefix_op_handler( 50, 'not_'),
            "+":      prefix_op_handler(110, 'pos'),
            "-":      prefix_op_handler(110, 'neg'),
            "(":      parens_expr_handler,
            "(/":     inplace_array('(/', '/)'),
            "[":      inplace_array('[', ']')
            }
        infix_ops = {
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
            "%":      (140, infix_op_handler(141, 'resolve')),
            "(":      (140, call_handler),
            }

        # Fortran 90 operator aliases
        infix_ops["=="] = infix_ops["eq"]
        infix_ops["/="] = infix_ops["ne"]
        infix_ops["<="] = infix_ops["le"]
        infix_ops[">="] = infix_ops["ge"]
        infix_ops["<"]  = infix_ops["lt"]
        infix_ops[">"]  = infix_ops["gt"]

        prefix_cats = {
            lexer.CAT_STRING:     literal_handler('string'),
            lexer.CAT_FLOAT:      literal_handler('float'),
            lexer.CAT_INT:        literal_handler('int'),
            lexer.CAT_RADIX:      literal_handler('radix'),
            lexer.CAT_BOOLEAN:    literal_handler('bool'),
            lexer.CAT_CUSTOM_DOT: custom_unary_handler(120),
            lexer.CAT_WORD:       literal_handler('ref'),
            }
        infix_cats = {
            lexer.CAT_CUSTOM_DOT: (10, custom_binary_handler(11))
            }

        self._infix_ops = infix_ops
        self._infix_cats = infix_cats
        self._prefix_ops = prefix_ops
        self._prefix_cats = prefix_cats

    def get_prefix_handler(self, cat, token):
        try:
            if cat == lexer.CAT_SYMBOLIC_OP:
                return self._prefix_ops[token]
            elif cat == lexer.CAT_BUILTIN_DOT:
                return self._prefix_ops[token.lower()]
            else:
                return self._prefix_cats[cat]
        except KeyError:
            raise NoMatch()

    def get_infix_handler(self, cat, token):
        try:
            if cat == lexer.CAT_SYMBOLIC_OP:
                return self._infix_ops[token]
            elif cat == lexer.CAT_BUILTIN_DOT:
                return self._infix_ops[token.lower()]
            else:
                return self._infix_cats[cat]
        except KeyError:
            raise NoMatch()

EXPR_HANDLER = ExpressionHandler()

def expr(tokens, min_glue=0):
    # Get prefix
    handler = EXPR_HANDLER.get_prefix_handler(*tokens.peek())
    try:
        result = handler(tokens)

        # Cycle through appropriate infixes:
        while True:
            try:
                glue, handler = EXPR_HANDLER.get_infix_handler(*tokens.peek())
            except NoMatch:
                return result
            else:
                if glue < min_glue:
                    return result
                result = handler(tokens, result)
    except NoMatch:
        raise ParserError(tokens, "Invalid expression")

_optional_expr = optional(expr)

# -----------

@rule
def kind_selector(tokens):
    if tokens.marker('*'):
        with LockedIn(tokens):
            kind_ = int_(tokens)
    else:
        tokens.expect('(')
        with LockedIn(tokens):
            if tokens.marker('kind'):
                tokens.expect('=')
            kind_ = expr(tokens)
            tokens.expect(')')
    return tokens.produce('kind_sel', kind_)

@rule
def keyword_arg(tokens, choices=None):
    sel = tokens.expect_cat(lexer.CAT_WORD).lower()
    tokens.expect('=')
    if sel not in choices:
        raise NoMatch()
    return sel

@rule
def char_len(tokens):
    if tokens.marker('*'):
        return '*'
    if tokens.marker(':'):
        return ':'
    return expr(tokens)

@rule
def char_len_suffix(tokens):
    tokens.expect('*')
    with LockedIn(tokens):
        if tokens.marker('('):
            len_ = char_len(tokens)
            tokens.expect(')')
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
        tokens.expect('(')
        with LockedIn(tokens):
            sel = _optional_len_kind_kwd(tokens)
            if sel == 'len' or sel is None:
                len_ = char_len(tokens)
            else:
                kind = expr(tokens)

            if tokens.marker(','):
                sel = _optional_len_kind_kwd(tokens)
                print(sel)
                if sel is None:
                    sel = 'kind' if kind is None else 'len'
                if sel == 'len':
                    len_ = char_len(tokens)
                else:
                    kind = expr(tokens)

            tokens.expect(')')

    return tokens.produce('char_sel', len_, kind)

def _typename_handler(tokens):
    tokens.expect('type')
    tokens.expect('(')
    with LockedIn(tokens):
        typename = identifier(tokens)
        tokens.expect(')')
        return ('type', typename)

def _class_handler(tokens):
    tokens.expect('class')
    tokens.expect('(')
    with LockedIn(tokens):
        if tokens.marker('*'):
            typename = None
        else:
            typename = identifier(tokens)
        tokens.expect(')')
        return ('class_', typename)

def double_precision_type(tokens):
    tokens.expect('doubleprecision')
    return tokens.produce('real_type', 'double')

def double_type(tokens):
    tokens.expect('double')
    if tokens.marker('precision'):
        return tokens.produce('real_type', 'double')
    else:
        tokens.expect('complex')
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

@rule
def lower_bound(tokens):
    lower = _optional_expr(tokens)
    tokens.expect(':')
    return lower

optional_lower_bound = optional(lower_bound)

@rule
def dim_spec(tokens):
    lower = optional_lower_bound(tokens)
    if tokens.marker('*'):
        # Implied dimension
        return tokens.produce('implicit_dim', lower, '*')
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
    tokens.expect('(')
    dims = dimspec_sequence(tokens)
    tokens.expect(')')
    return dims

@rule
def intent(tokens):
    tokens.expect('intent')
    with LockedIn(tokens):
        tokens.expect('(')
        if tokens.marker('inout'):
            in_ = True
            out = True
        else:
            in_ = tokens.marker('in')
            out = tokens.marker('out')
            if not (in_ or out):
                raise ParserError("expecting in, out, or inout as intent.")
        tokens.expect(')')
        return tokens.produce('intent', in_, out)

_ENTITY_ATTR_HANDLERS = {
    'parameter':   tag('parameter', 'parameter'),
    'public':      tag('public', 'public'),
    'private':     tag('private', 'private'),
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
    tokens.expect(':')
    tokens.expect(':')

optional_double_colon = optional(double_colon)

def attribute_sequence(attr_rule, production_tag):
    @rule
    def attribute_sequence_rule(tokens):
        attrs = []
        while tokens.marker(','):
            attrs.append(attr_rule(tokens))
        try:
            double_colon(tokens)
        except NoMatch:
            if attrs: raise ParserError("Expecting ::")
        return tokens.produce(production_tag, *attrs)

    return attribute_sequence_rule

entity_attrs = attribute_sequence(entity_attr, 'entity_attrs')

def init_assign(tokens):
    tokens.expect('=')
    with LockedIn(tokens):
        init = expr(tokens)
        return tokens.produce('init_assign', init)

def init_point(tokens):
    tokens.expect('=>')
    with LockedIn(tokens):
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

    # Flatten out entity list for simplicity of handling:
    for e in entities[1:]:
        return tokens.produce('entity_decl', type_, attrs_, *e[1:])

@rule
def entity_ref(tokens):
    name = identifier(tokens)
    shape_ = optional_shape(tokens)
    return tokens.produce('entity_ref', name, shape_)

@rule
def bind_c(tokens):
    tokens.expect('bind')
    tokens.expect('(')
    tokens.expect('c')
    if tokens.marker(','):
        tokens.expect('name')
        tokens.expect('=')
        name = expr(tokens)
    else:
        name = None
    tokens.expect(')')
    return tokens.produce('bind_c', name)

def extends(tokens):
    tokens.expect('extends')
    with LockedIn(tokens):
        tokens.expect('(')
        name = identifier(tokens)
        tokens.expect(')')
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
    return ('preproc_stmt', tokens.expect_cat(lexer.CAT_PREPROC))

_BLOCK_DELIM = { 'end', 'else', 'contains', 'case' }

def block(rule, production_tag='block', fenced=True):
    # Fortran blocks are delimited by one of these words, so we can use
    # them in failing fast
    def block_rule(tokens, until_lineno=None):
        stmts = []
        while True:
            cat, token = tokens.peek()
            if cat == lexer.CAT_INT:
                tokens.advance()
                if int(token) == until_lineno:        # non-block do construct
                    break
            elif cat == lexer.CAT_EOS:
                tokens.advance()
            elif cat == lexer.CAT_PREPROC:
                stmts.append(preproc_stmt(tokens))
            elif fenced and token.lower() in _BLOCK_DELIM:
                break
            else:
                try:
                    stmts.append(rule(tokens))
                except NoMatch:
                    if fenced:
                        raise ParserError(tokens,
                                          "Expecting item or block delimiter.")
                    break
        return tokens.produce(production_tag, *stmts)

    return block_rule

component_block = block(entity_decl, 'component_block')

private_stmt = tag_stmt('private', 'private')

sequence_stmt = tag_stmt('sequence', 'sequence')

_TYPE_TAG_HANDLERS = {
    'private':     private_stmt,
    'sequence':    sequence_stmt,
    }

type_tag = prefixes(_TYPE_TAG_HANDLERS)

type_tag_block = block(type_tag, 'type_tags', fenced=False)

optional_identifier = optional(identifier)

def pass_attr(tokens):
    tokens.expect('pass')
    if tokens.marker('('):
        ident = identifier(tokens)
        tokens.expect(')')
    else:
        ident = None
    return tokens.produce('pass', identifier)

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
    if tokens.marker('=>'):
        ref = identifier(tokens)
    else:
        ref = None
    return tokens.produce('type_proc', name, ref)

type_proc_sequence = comma_sequence(type_proc, 'type_proc_list')

def type_proc_decl(tokens):
    tokens.expect('procedure')
    with LockedIn(tokens):
        if tokens.marker('('):
            iface_name = identifier(tokens)
            tokens.expect(')')
        else:
            iface_name = None
        attrs = type_proc_attrs(tokens)
        procs = type_proc_sequence(tokens)
        eos(tokens)
        # Flatten list
        for proc in procs[1:]:
            return tokens.produce('type_proc_decl', iface_name, attrs, *proc[1:])

def generic_decl(tokens):
    tokens.expect('generic')
    with LockedIn(tokens):
        attrs = type_proc_attrs(tokens)
        name = iface_name(tokens)
        tokens.expect('=>')
        refs = identifier_sequence(tokens)
        eos(tokens)
        return tokens.produce('generic_decl', name, attrs, refs)

def final_decl(tokens):
    tokens.expect('final')
    with LockedIn(tokens):
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
    if tokens.marker('contains'):
        private = optional_private_stmt(tokens)
        conts = type_contains_block(tokens)
        return ('type_bound_procedures', private, *conts[1:])
    else:
        return None

def end_stmt(objtype, require_type=False, name_type=None):
    comp = 'end' + objtype
    if name_type is None:
        name_type = identifier

    @rule
    def end_stmt_rule(tokens):
        if tokens.marker('end'):
            if not tokens.marker(objtype):
                if require_type:
                    raise NoMatch()
                eos(tokens)
                return None
        elif not tokens.marker(comp):
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
    tokens.expect('type')
    attrs = type_attrs(tokens)
    name = identifier(tokens)
    with LockedIn(tokens):
        eos(tokens)
        tags = type_tag_block(tokens)
        decls = component_block(tokens)
        proc = optional_procedures_block(tokens)
        end_type_stmt(tokens)
        return tokens.produce('type_decl', name, attrs, tags, decls, proc)

@rule
def rename(tokens):
    alias = identifier(tokens)
    tokens.expect('=>')
    name = identifier(tokens)
    return tokens.produce('rename', alias, name)

rename_sequence = comma_sequence(rename, 'rename_list')

@rule
def oper_spec(tokens):
    if tokens.marker('assignment'):
        tokens.expect('(')
        tokens.expect('=')
        tokens.expect(')')
        return tokens.produce('oper_spec', '=')
    else:
        tokens.expect('operator')
        try:
            # It is impossible for the lexer to disambiguate between an empty
            # in-place array (//) and bracketed slashes, so we handle it here:
            oper = tokens.expect_cat(lexer.CAT_BRACKETED_SLASH)
        except NoMatch:
            tokens.expect('(')
            cat, token = next(tokens)
            if cat == lexer.CAT_CUSTOM_DOT:
                oper = tokens.produce('custom_op', token)
            elif cat == lexer.CAT_SYMBOLIC_OP or lexer.CAT_BUILTIN_DOT:
                oper = token
            else:
                raise NoMatch()
            tokens.expect(')')
        return tokens.produce('oper_spec', oper)

@rule
def only(tokens):
    try:
        return oper_spec(tokens)
    except NoMatch:
        name = identifier(tokens)
        if tokens.marker('=>'):
            target = identifier(tokens)
            return tokens.produce('rename', name, target)
        else:
            return name

only_sequence = comma_sequence(only, 'only_list')

@rule
def use_stmt(tokens):
    tokens.expect('use')
    with LockedIn(tokens):
        name = identifier(tokens)
        clauses = tokens.produce('rename_list')   # default empty rename list
        is_only = False
        if tokens.marker(','):
            if tokens.marker('only'):
                is_only = True
                tokens.expect(':')
                clauses = only_sequence(tokens)
            else:
                clauses = rename_sequence(tokens)
        eos(tokens)
        return tokens.produce('use_stmt', name, clauses)

_letter_re = re.compile(r'^[a-zA-Z]$')

@rule
def letter_range(tokens):
    def letter():
        cand = next(tokens)[1]
        if _letter_re.match(cand):
            return cand.lower()
        else:
            raise NoMatch()

    start = letter()
    end = start
    if tokens.marker('-'):
        end = letter()
    return tokens.produce('letter_range', start, end)

letter_range_sequence = comma_sequence(letter_range, 'letter_range_list')

@rule
def implicit_spec(tokens):
    type_ = type_spec(tokens)
    tokens.expect('(')
    ranges = letter_range_sequence(tokens)
    tokens.expect(')')
    return tokens.produce('implicit_spec', type_, ranges)

implicit_spec_sequence = comma_sequence(implicit_spec, 'implicit_decl')

@rule
def implicit_stmt(tokens):
    tokens.expect('implicit')
    with LockedIn(tokens):
        if tokens.marker('none'):
            eos(tokens)
            return tokens.produce('implicit_none')
        else:
            specs = implicit_spec_sequence(tokens)
            eos(tokens)
            return specs

@rule
def dummy_arg(tokens):
    if tokens.marker('*'):
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
    if tokens.marker('contains'):
        with LockedIn(tokens):
            eos(tokens)
            return contained_block(tokens)
    else:
        return tokens.produce('contained_block')

end_subroutine_stmt = end_stmt('subroutine', require_type=False)

@rule
def subroutine_decl(tokens):
    # Header
    prefixes = sub_prefix_sequence(tokens)
    tokens.expect('subroutine')
    with LockedIn(tokens):
        name = identifier(tokens)
        if tokens.marker('('):
            args = dummy_arg_sequence(tokens)
            tokens.expect(')')
        else:
            args = tokens.produce('arg_list')   # empty args
        bind_ = optional_bind_c(tokens)
        eos(tokens)

        # Body
        declarations_ = declaration_part(tokens)
        execution_part(tokens)
        optional_contained_part(tokens)

        # Footer
        end_subroutine_stmt(tokens)
        return tokens.produce('subroutine_decl', name, prefixes, args, bind_,
                              declarations_)


_FUNC_PREFIX_HANDLERS = {
    'elemental': tag('elemental', 'elemental'),
    'impure':    tag('impure', 'impure'),
    'pure':      tag('pure', 'pure'),
    'recursive': tag('recursive', 'recursive'),
    }

func_modifier = prefixes(_FUNC_PREFIX_HANDLERS)

@rule
def func_prefix(tokens):
    try:
        return func_modifier(tokens)
    except NoMatch:
        return type_spec(tokens)

func_prefix_sequence = ws_sequence(func_prefix, 'func_prefix_list')

@rule
def result(tokens):
    tokens.expect('result')
    tokens.expect('(')
    res = identifier(tokens)
    tokens.expect(')')
    return ('result', res)

@rule
def func_suffix(tokens):
    try:
        return result(tokens)
    except NoMatch:
        return bind_c(tokens)

func_suffix_sequence = ws_sequence(func_suffix, 'func_suffix_list')

func_arg_sequence = comma_sequence(identifier, 'arg_list', allow_empty=True)

end_function_stmt = end_stmt('function', require_type=False)

@rule
def function_decl(tokens):
    # Header
    prefixes = func_prefix_sequence(tokens)
    tokens.expect('function')
    with LockedIn(tokens):
        name = identifier(tokens)
        tokens.expect('(')
        args = func_arg_sequence(tokens)
        tokens.expect(')')
        suffixes = func_suffix_sequence(tokens)
        eos(tokens)

        # Body
        declarations_ = declaration_part(tokens)
        execution_part(tokens)
        optional_contained_part(tokens)

        # Footer
        end_function_stmt(tokens)
        return tokens.produce('function_decl', name, prefixes, args, suffixes,
                              declarations_)

@rule
def subprogram_decl(tokens):
    try:
        return subroutine_decl(tokens)
    except NoMatch:
        return function_decl(tokens)

contained_block = block(subprogram_decl, 'contained_block')

@rule
def iface_name(tokens):
    try:
        return oper_spec(tokens)
    except NoMatch:
        return identifier(tokens)

optional_iface_name = optional(iface_name)

identifier_sequence = comma_sequence(identifier, 'identifier_list')

@rule
def module_proc_stmt(tokens):
    tokens.expect('module')
    tokens.expect('procedure')
    with LockedIn(tokens):
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
    tokens.expect('interface')
    with LockedIn(tokens):
        name = optional_iface_name(tokens)
        eos(tokens)
        decls = interface_body_block(tokens)
        end_interface_stmt(tokens)
        return tokens.produce('interface_decl', name, decls)

@rule
def abstract_interface_decl(tokens):
    tokens.expect('abstract')
    tokens.expect('interface')
    with LockedIn(tokens):
        eos(tokens)
        decls = interface_body_block(tokens)
        end_interface_stmt(tokens)
        return tokens.produce('abstract_interface_decl', decls)

def imbue_stmt(prefix_rule, object_rule):
    object_sequence = comma_sequence(object_rule, None)
    def imbue_stmt_rule(tokens):
        prefix = prefix_rule(tokens)
        with LockedIn(tokens):
            optional_double_colon(tokens)
            vars = object_sequence(tokens)
            eos(tokens)
            return tokens.produce('imbue', prefix, *vars[1:])
    return imbue_stmt_rule

# TODO: one can also save common blocks
imbue_save_stmt = imbue_stmt(tag('save', 'save'), identifier)

@rule
def save_all_stmt(tokens):
    tokens.expect('save')
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
    tokens.expect('parameter')
    with LockedIn(tokens):
        tokens.expect('(')
        seq = param_init_sequence(tokens)
        tokens.expect(')')
        eos(tokens)
    return seq

equivalence_object_sequence = comma_sequence(lvalue, 'equivalence_set')

@rule
def equivalence_set(tokens):
    tokens.expect('(')
    seq = equivalence_object_sequence(tokens)
    tokens.expect(')')
    return seq

equivalence_set_sequence = comma_sequence(equivalence_set, 'equivalence_stmt')

@rule
def equivalence_stmt(tokens):
    tokens.expect('equivalence')
    with LockedIn(tokens):
        seq = equivalence_set_sequence(tokens)
        eos(tokens)
        return seq

private_imbue_stmt = imbue_stmt(tag('private', 'private'), iface_name)

def private_or_imbue_stmt(tokens):
    try:
        # As a common extension, many compilers allow 'private' statements
        # as part of a module.
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
    tokens.expect('procedure')
    tokens.expect('(')
    iface = optional_identifier(tokens)
    tokens.expect(')')
    attrs = proc_attrs(tokens)
    procs = procedure_sequence(tokens)
    return tokens.produce('procedure_decl', iface, attrs, procs)

# TODO: some imbue statements are missing here.
_DECLARATION_HANDLERS = {
    'use':         use_stmt,
    'implicit':    implicit_stmt,
    'abstract':    abstract_interface_decl,
    'interface':   interface_decl,
    'equivalence': equivalence_stmt,
    'procedure':   procedure_decl,

    'public':      imbue_stmt(tag('public', 'public'), iface_name),
    'private':     private_or_imbue_stmt,
    'parameter':   parameter_stmt,
    'external':    imbue_stmt(tag('external', 'external'), identifier),
    'intent':      imbue_stmt(intent, identifier),
    'intrinsic':   imbue_stmt(tag('intrinsic', 'intrinsic'), identifier),
    'optional':    imbue_stmt(tag('optional', 'optional'), identifier),
    'save':        save_stmt,
    }

prefixed_declaration_stmt = prefixes(_DECLARATION_HANDLERS)

@rule
def declaration_stmt(tokens):
    try:
        return prefixed_declaration_stmt(tokens)
    except NoMatch:
        try:
            return type_decl(tokens)
        except NoMatch:
            return entity_decl(tokens)

declaration_part = block(declaration_stmt, 'declaration_block', fenced=False)

fenced_declaration_part = block(declaration_stmt, 'declaration_block', fenced=True)

@rule
def construct_tag(tokens):
    tokens.expect_cat(lexer.CAT_WORD)
    tokens.expect(':')
    cat, token = tokens.peek()
    return token.lower()

optional_construct_tag = optional(construct_tag)

@rule
def if_clause(tokens):
    tokens.expect('if')
    tokens.expect('(')
    expr(tokens)
    tokens.expect(')')

else_if = composite('else', 'if')

@rule
def else_if_block(tokens):
    else_if(tokens)
    tokens.expect('(')
    expr(tokens)
    tokens.expect(')')
    optional_identifier(tokens)
    eos(tokens)
    with LockedIn(tokens):
        execution_part(tokens)

else_if_block_sequence = ws_sequence(else_if_block, 'else_if_sequence')

@rule
def else_block(tokens):
    tokens.expect('else')
    optional_identifier(tokens)
    eos(tokens)
    execution_part(tokens)

optional_else_block = optional(else_block)

def ignore_stmt(tokens):
    while True:
        cat, token = next(tokens)
        if cat == lexer.CAT_EOS:
            return

end_if_stmt = end_stmt('if')

@rule
def if_construct(tokens):
    optional_construct_tag(tokens)
    if_clause(tokens)
    with LockedIn(tokens):
        if tokens.marker('then'):
            eos(tokens)
            execution_part(tokens)
            else_if_block_sequence(tokens)
            optional_else_block(tokens)
            end_if_stmt(tokens)
        else:
            ignore_stmt(tokens)

@rule
def while_ctrl(tokens):
    tokens.expect('while')
    tokens.expect('(')
    expr(tokens)
    tokens.expect(')')

def loop_ctrl(tokens):
    try:
        while_ctrl(tokens)
    except NoMatch:
        do_ctrl(tokens)

optional_loop_ctrl = optional(loop_ctrl)

end_do_stmt = end_stmt('do')

@rule
def do_construct(tokens):
    optional_construct_tag(tokens)
    tokens.expect('do')
    with LockedIn(tokens):
        try:
            until_lineno = int(int_(tokens)[1])
        except NoMatch:
            # BLOCK DO CONSTRUCT
            tokens.marker(',')
            optional_loop_ctrl(tokens)
            eos(tokens)
            execution_part(tokens)
            end_do_stmt(tokens)
        else:
            # NONBLOCK DO CONSTRUCT
            # TODO: nested non-block do constructs with shared end label
            tokens.marker(',')
            optional_loop_ctrl(tokens)
            eos(tokens)
            execution_part(tokens, until_lineno)
            try:
                end_do_stmt(tokens)
            except NoMatch:
                execution_stmt(tokens)

@rule
def case_slice(tokens):
    _optional_expr(tokens)
    tokens.expect(':')
    _optional_expr(tokens)

def case_range(tokens):
    try:
        case_slice(tokens)
    except NoMatch:
        expr(tokens)

case_range_sequence = comma_sequence(case_range, 'case_range_list')

@rule
def select_case(tokens):
    tokens.expect('case')
    if tokens.marker('default'):
        pass
    else:
        tokens.expect('(')
        case_range_sequence(tokens)
        tokens.expect(')')
    eos(tokens)
    execution_part(tokens)

select_case_sequence = block(select_case, 'select_case_list', fenced=False)

end_select_stmt = end_stmt('select')

select_case_tag = composite('select', 'case')

@rule
def select_case_construct(tokens):
    optional_construct_tag(tokens)
    select_case_tag(tokens)
    with LockedIn(tokens):
        tokens.expect('(')
        expr(tokens)
        tokens.expect(')')
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
    tokens.expect('=')
    expr(tokens)
    tokens.expect(':')
    expr(tokens)
    if tokens.marker(':'):
        expr(tokens)

@rule
def forall_clause(tokens):
    tokens.expect('forall')
    with LockedIn(tokens):
        tokens.expect('(')
        optional_type_prefix(tokens)
        forall_select(tokens)
        while tokens.marker(','):
            try:
                forall_select(tokens)
            except NoMatch:
                expr(tokens)
                break
        tokens.expect(')')

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
    tokens.expect('where')
    tokens.expect('(')
    expr(tokens)
    tokens.expect(')')

end_where_stmt = end_stmt('where')

else_where = composite('else', 'where')

@rule
def where_construct(tokens):
    optional_construct_tag(tokens)
    where_clause(tokens)
    with LockedIn(tokens):
        try:
            eos(tokens)
        except NoMatch:
            # WHERE STMT
            ignore_stmt(tokens)
        else:
            # WHERE BLOCK
            execution_part(tokens)
            while else_where(tokens):
                if tokens.marker('('):
                    expr(tokens)
                    tokens.expect(')')
                optional_identifier(tokens)
                execution_part(tokens)
            end_where_stmt(tokens)

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
    'data':       ignore_stmt,
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
    }
STMT_HANDLERS.update(CONSTRUCT_HANDLERS)

prefixed_stmt = prefixes(STMT_HANDLERS)

@rule
def assignment_stmt(tokens):
    lvalue(tokens)
    _, oper = next(tokens)
    if oper != '=' and oper != '=>':
        raise NoMatch()
    ignore_stmt(tokens)
    #with LockedIn(tokens):
    #    expr(tokens)
    #eos(tokens)

@rule
def format_stmt(tokens):
    tokens.expect_cat(lexer.CAT_FORMAT)

@rule
def execution_stmt(tokens):
    try:
        prefixed_stmt(tokens)
    except NoMatch:
        try:
            assignment_stmt(tokens)
        except NoMatch:
            try:
                # This is the least likely, so it moved here.
                construct_tag(tokens)
                construct(tokens)
            except NoMatch:
                format_stmt(tokens)

# FIXME: even though this incurs a runtime penalty, we cannot use a simple
#        fence here, since it is technically allowed to cause maximum confusion
#        by naming a variable 'end'.
execution_part = block(execution_stmt, 'execution_block', fenced=False)

end_module_stmt = end_stmt('module', require_type=False)

@rule
def module_decl(tokens):
    tokens.expect('module')
    with LockedIn(tokens):
        name = identifier(tokens)
        eos(tokens)
        decls = fenced_declaration_part(tokens)
        cont = optional_contained_part(tokens)
        end_module_stmt(tokens)
        return tokens.produce('module_decl', name, decls, cont)

end_program_stmt = end_stmt('program', require_type=False)

@rule
def program_decl(tokens):
    tokens.expect('program')
    with LockedIn(tokens):
        name = identifier(tokens)
        eos(tokens)
        decls = declaration_part(tokens)
        execution_part(tokens)
        cont = optional_contained_part(tokens)
        end_program_stmt(tokens)
        return tokens.produce('program_decl', name, decls, cont)

@rule
def program_unit(tokens):
    try:
        return program_decl(tokens)
    except NoMatch: pass
    try:
        return module_decl(tokens)
    except NoMatch:
        return subprogram_decl(tokens)
    # TODO: block_data
    # TODO: make this faster

program_unit_sequence = block(program_unit, 'program_unit_list', fenced=False)

@rule
def compilation_unit(tokens, filename=None):
    units = program_unit_sequence(tokens)
    tokens.expect_cat(lexer.CAT_DOLLAR)
    version = tokens.produce('ast_version', '0', '1')
    fname = tokens.produce('filename', filename)
    return tokens.produce('compilation_unit', version, fname, *units[1:])
