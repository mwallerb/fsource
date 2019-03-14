#!/usr/bin/env python
from __future__ import print_function
import lexer
import re

class NoMatch(Exception):
    pass

# CONTEXT

class TokenStream:
    def __init__(self, tokens, pos=0):
        self.tokens = tokens
        self.pos = pos
        self.stack = []

    def peek(self):
        try:
            return self.tokens[self.pos]
        except IndexError:
            if self.pos > len(self.tokens) + 1:  # TODO
                raise StopIteration()
            return lexer.CAT_DOLLAR, '<$>'

    def __next__(self):
        value = self.peek()
        self.pos += 1
        return value

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
        next(self)

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
            next(self)
            return True
        else:
            return False

def expect_eos(tokens):
    # TODO rewrite as rule?
    cat, token = tokens.peek()
    if cat != lexer.CAT_EOS and cat != lexer.CAT_DOLLAR and token != ';':
        raise NoMatch()
    return next(tokens)

def comma_sequence(rule):
    def comma_sequence_rule(tokens):
        vals = []
        vals.append(rule(tokens))
        while tokens.marker(','):
            vals.append(rule(tokens))
        return vals

    return comma_sequence_rule

def ws_sequence(rule):
    def ws_sequence_rule(tokens):
        items = []
        try:
            while True:
                items.append(rule(tokens))
        except NoMatch:
            return items

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
        return production_tag

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

class Rule:
    def __init__(self, tokens):
        self.tokens = tokens

    def __enter__(self):
        self.tokens.push()

    def __exit__(self, exc_type, value, traceback):
        if exc_type is not None:
            tokens.backtrack()
            return False
        else:
            tokens.commit()
            return True


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

class LockedIn:
    def __init__(self, tokens):
        self.tokens = tokens

    def __enter__(self): pass

    def __exit__(self, exc_type, exc_val, traceback):
        if exc_type is NoMatch:
            print(self.tokens.tokens[self.tokens.pos:self.tokens.pos+10])
            raise ValueError("Parsing failure")

@rule
def int_(tokens):
    return tokens.produce('int', tokens.expect_cat(lexer.CAT_INT))

@rule
def float_(tokens):
    return tokens.produce('float', tokens.expect_cat(lexer.CAT_FLOAT))

@rule
def string_(tokens):
    return tokens.produce('string', tokens.expect_cat(lexer.CAT_STRING))

@rule
def bool_(tokens):
    return tokens.produce('bool', tokens.expect_cat(lexer.CAT_BOOLEAN))

@rule
def radix(tokens):
    return tokens.produce('radix', tokens.expect_cat(lexer.CAT_RADIX))

@rule
def identifier(tokens):
    return tokens.produce('identifier', tokens.expect_cat(lexer.CAT_WORD).lower())

@rule
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

@rule
def inplace_array(tokens):
    seq = []
    tokens.expect('(/')
    with LockedIn(tokens):
        if tokens.marker('/)'):
            return tokens.produce('array')
        while True:
            try:
                seq.append(implied_do(tokens))
            except NoMatch:
                seq.append(expr(tokens))
            if tokens.marker('/)'):
                return tokens.produce('array', *seq)
            tokens.expect(',')

@rule
def parens_expr(tokens):
    tokens.expect('(')
    inner_expr = expr(tokens)
    tokens.expect(')')
    return tokens.produce('parens', inner_expr)


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


class _PrefixHandler:
    def __init__(self, subglue, action):
        self.subglue = subglue
        self.action = action

    def __call__(self, tokens):
        next(tokens)
        operand = expr(tokens, self.subglue)
        return (self.action, operand)


class _CustomUnary(_PrefixHandler):
    def __call__(self, tokens):
        operator = custom_op(tokens)
        operand = expr(tokens, self.subglue)
        return (self.action, operator, operand)


class _InfixHandler:
    def __init__(self, glue, assoc, action):
        if assoc not in ('left', 'right'):
            raise ValueError("Associativity must be either left or right")
        self.glue = glue
        self.subglue = glue + (0 if assoc == 'right' else 1)
        self.action = action

    def __call__(self, tokens, lhs):
        next(tokens)
        rhs = expr(tokens, self.subglue)
        return (self.action, lhs, rhs)


class _CustomBinary(_InfixHandler):
    def __call__(self, tokens, lhs):
        operator = custom_op(tokens)
        rhs = expr(tokens, self.subglue)
        return (self.action, operator, lhs, rhs)


class _SubscriptHandler:
    def __init__(self, glue):
        self.glue = glue

    def __call__(self, tokens, lhs):
        next(tokens)
        seq = []
        if tokens.marker(')'):
            return tokens.produce('call', lhs)
        while True:
            try:
                seq.append(slice_(tokens))
            except NoMatch:
                seq.append(expr(tokens))
            if tokens.marker(')'):
                return tokens.produce('call', lhs, *seq)
            tokens.expect(',')


class ExpressionHandler:
    def __init__(self):
        prefix_ops = {
            ".not.":  _PrefixHandler( 50, 'not_'),
            "+":      _PrefixHandler(110, 'pos'),
            "-":      _PrefixHandler(110, 'neg'),
            "(":      parens_expr,
            "(/":     inplace_array,
            }
        infix_ops = {
            ".eqv.":  _InfixHandler( 20, 'right', 'eqv'),
            ".neqv.": _InfixHandler( 20, 'right', 'neqv'),
            ".or.":   _InfixHandler( 30, 'right', 'or_'),
            ".and.":  _InfixHandler( 40, 'right', 'and_'),
            ".eq.":   _InfixHandler( 60, 'left',  'eq'),
            ".ne.":   _InfixHandler( 60, 'left',  'ne'),
            ".le.":   _InfixHandler( 60, 'left',  'le'),
            ".lt.":   _InfixHandler( 60, 'left',  'lt'),
            ".ge.":   _InfixHandler( 60, 'left',  'ge'),
            ".gt.":   _InfixHandler( 60, 'left',  'gt'),
            "//":     _InfixHandler( 70, 'left',  'concat'),
            "+":      _InfixHandler( 80, 'left',  'plus'),
            "-":      _InfixHandler( 80, 'left',  'minus'),
            "*":      _InfixHandler( 90, 'left',  'mul'),
            "/":      _InfixHandler( 90, 'left',  'div'),
            "**":     _InfixHandler(100, 'right', 'pow'),
            "%":      _InfixHandler(130, 'left',  'resolve'),
            "_":      _InfixHandler(130, 'left',  'kind'),
            "(":      _SubscriptHandler(140),
            }

        # Fortran 90 operator aliases
        infix_ops["=="] = infix_ops[".eq."]
        infix_ops["/="] = infix_ops[".ne."]
        infix_ops["<="] = infix_ops[".le."]
        infix_ops[">="] = infix_ops[".ge."]
        infix_ops["<"]  = infix_ops[".lt."]
        infix_ops[">"]  = infix_ops[".gt."]

        prefix_cats = {
            lexer.CAT_STRING:     string_,
            lexer.CAT_FLOAT:      float_,
            lexer.CAT_INT:        int_,
            lexer.CAT_RADIX:      radix,
            lexer.CAT_BOOLEAN:    bool_,
            lexer.CAT_CUSTOM_DOT: _CustomUnary(120, 'unary'),
            lexer.CAT_WORD:       identifier,
            }
        infix_cats = {
            lexer.CAT_CUSTOM_DOT: _CustomBinary(10, 'left', 'binary')
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
def expr(tokens, min_glue=0):
    # Get prefix
    handler = EXPR_HANDLER.get_prefix_handler(*tokens.peek())
    result = handler(tokens)

    # Cycle through appropriate infixes:
    while True:
        try:
            handler = EXPR_HANDLER.get_infix_handler(*tokens.peek())
        except NoMatch:
            return result
        else:
            if handler.glue < min_glue:
                return result
            result = handler(tokens, result)

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
    sel = tokens.expect_cat(lexer.CAT_WORD)
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
        sel = _optional_len_kind_kwd(tokens)
        if sel == 'len' or sel is None:
            len_ = char_len(tokens)
        else:
            kind = expr(tokens)

        if tokens.marker(','):
            sel = _optional_len_kind_kwd(tokens)
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

def double_precision(tokens):
    tokens.expect('double')
    tokens.expect('precision')
    return ('double_precision',)

_TYPE_SPEC_HANDLERS = {
    'integer':   prefix('integer', optional(kind_selector), 'integer_type'),
    'real':      prefix('real', optional(kind_selector), 'real_type'),
    'double':    double_precision,
    'complex':   prefix('complex', optional(kind_selector), 'complex_type'),
    'character': prefix('character', optional(char_selector), 'character_type'),
    'logical':   prefix('logical', optional(kind_selector), 'logical_type'),
    'type':      _typename_handler
    }

type_spec = prefixes(_TYPE_SPEC_HANDLERS)

@rule
def dim_spec(tokens):
    try:
        lower = _optional_expr(tokens)
        tokens.expect(':')
    except NoMatch:
        pass
    if tokens.marker('*'):
        upper = '*'
    else:
        upper = _optional_expr(tokens)
    return tokens.produce('dim_spec', lower, upper)

dimspec_sequence = comma_sequence(dim_spec)

@rule
def shape(tokens):
    tokens.expect('(')
    dims = dimspec_sequence(tokens)
    tokens.expect(')')
    return tokens.produce('shape', *dims)

_INTENT_STRINGS = {
    'in':    (True, False),
    'inout': (True, True),
    'out':   (False, True),
    }

@rule
def intent(tokens):
    tokens.expect('intent')
    with LockedIn(tokens):
        tokens.expect('(')
        string = tokens.expect_cat(lexer.CAT_WORD).lower()
        try:
            in_, out = _INTENT_STRINGS[string]
        except KeyError:
            raise NoMatch()
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

@rule
def entity_attrs(tokens):
    handler_dict = _ENTITY_ATTR_HANDLERS
    attrs = []
    while tokens.marker(','):
        attrs.append(entity_attr(tokens))
    try:
        double_colon(tokens)
    except NoMatch:
        if attrs: raise NoMatch()
    return tokens.produce('entity_attrs', *attrs)

@rule
def initializer(tokens):
    if tokens.marker('='):
        init = expr(tokens)
        return tokens.produce('init_assign', init)
    else:
        tokens.expect('=>')
        init = expr(tokens)
        return tokens.produce('init_point', init)

optional_char_len_suffix = optional(char_len_suffix)

optional_shape = optional(shape)

optional_initializer = optional(initializer)

@rule
def entity(tokens):
    name = identifier(tokens)
    len_ = optional_char_len_suffix(tokens)
    shape_ = optional_shape(tokens)
    init = optional_initializer(tokens)
    return tokens.produce('entity', name, len_, shape_, init)

entity_sequence = comma_sequence(entity)

@rule
def entity_stmt(tokens):
    type_ = type_spec(tokens)
    attrs_ = entity_attrs(tokens)
    entities = entity_sequence(tokens)
    expect_eos(tokens)
    return tokens.produce('entity_stmt', type_, attrs_, *entities)

@rule
def entity_ref(tokens):
    name = identifier(tokens)
    shape_ = optional_shape(tokens)
    return tokens.produce('entity_ref', name, shape_)

_TYPE_ATTR_HANDLERS = {
    'public':      tag('public', 'public'),
    'private':     tag('private', 'private'),
    }

type_attr = prefixes(_TYPE_ATTR_HANDLERS)

@rule
def type_attrs(tokens):
    attrs = []
    while tokens.marker(','):
        attrs.append(type_attr(tokens))
    try:
        double_colon(tokens)
    except NoMatch:
        if attrs: raise ValueError("Expecting ::")
    return tokens.produce('type_attrs', *attrs)

@rule
def lineno(tokens):
    no = tokens.expect_cat(lexer.CAT_LINENO)
    # Make sure we actually label something.
    cat = tokens.peek()[0]
    if cat in (lexer.CAT_EOS, lexer.CAT_DOLLAR):
        raise NoMatch()
    return tokens.produce('lineno', no)

optional_lineno = optional(lineno)

def preproc_stmt(tokens):
    return ('preproc_stmt', tokens.expect_cat(lexer.CAT_PREPROC))

_BLOCK_DELIM = { 'end', 'else', 'elsewhere', 'contains', 'case' }

def block(rule, fenced=True):
    # Fortran blocks are delimited by one of these words, so we can use
    # them in failing fast
    def block_rule(tokens):
        stmts = []
        while True:
            cat, token = tokens.peek()
            if cat == lexer.CAT_EOS:
                next(tokens)
            elif cat == lexer.CAT_PREPROC:
                stmts.append(preproc_stmt(tokens))
            elif token.lower() in _BLOCK_DELIM:
                break
            else:
                try:
                    stmts.append(rule(tokens))
                except NoMatch:
                    if fenced:
                        print(tokens.tokens[tokens.pos:tokens.pos+10])
                        raise ValueError("Expecting item.")
                    break
            print (stmts[-1:])
        return stmts

    return block_rule

entity_block = block(entity_stmt)

@rule
def type_tags(tokens):
    private_ = False
    sequence_ = False
    while True:
        optional_lineno(tokens)
        if tokens.marker('private'):
            private_ = True
        elif tokens.marker('sequence'):
            sequence_ = True
        else:
            break
        expect_eos(tokens)
    return tokens.produce('type_tags', private_, sequence_)

optional_identifier = optional(identifier)

@rule
def type_decl(tokens):
    tokens.expect('type')
    attrs = type_attrs(tokens)
    name = identifier(tokens)
    with LockedIn(tokens):
        expect_eos(tokens)
        tags = type_tags(tokens)
        decls = entity_block(tokens)
        tokens.expect('end')
        if tokens.marker('type'):
            optional_identifier(tokens)
        expect_eos(tokens)
        return tokens.produce('type_decl', name, attrs, tags, *decls)

@rule
def rename(tokens):
    alias = identifier(tokens)
    tokens.expect('=>')
    name = identifier(tokens)
    return tokens.produce('rename', alias, name)

rename_sequence = comma_sequence(rename)

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
            elif cat == lexer.CAT_OP:
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

only_sequence = comma_sequence(only)

@rule
def use_stmt(tokens):
    tokens.expect('use')
    with LockedIn(tokens):
        name = identifier(tokens)
        clauses = []
        is_only = False
        if tokens.marker(','):
            if tokens.marker('only'):
                is_only = True
                tokens.expect(':')
                clauses = only_sequence(tokens)
            else:
                clauses = rename_sequence(tokens)
        expect_eos(tokens)
        return tokens.produce('use_stmt', name, is_only, *clauses)

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

letter_range_sequence = comma_sequence(letter_range)

@rule
def implicit_spec(tokens):
    type_ = type_spec(tokens)
    tokens.expect('(')
    ranges = letter_range_sequence(tokens)
    tokens.expect(')')
    return tokens.produce('implicit_spec', type_, *ranges)

implicit_spec_sequence = comma_sequence(implicit_spec)

@rule
def implicit_stmt(tokens):
    tokens.expect('implicit')
    with LockedIn(tokens):
        if tokens.marker('none'):
            expect_eos(tokens)
            return tokens.produce('implicit_none_stmt', )
        else:
            specs = implicit_spec_sequence(tokens)
            expect_eos(tokens)
            return tokens.produce('implicit_stmt', *specs)

@rule
def dummy_arg(tokens):
    if tokens.marker('*'):
        return '*'
    else:
        return identifier(tokens)

dummy_arg_sequence = optional(comma_sequence(dummy_arg))

_SUB_PREFIX_HANDLERS = {
    'impure':    tag('impure', 'impure'),
    'pure':      tag('pure', 'pure'),
    'recursive': tag('recursive', 'recursive'),
    }

sub_prefix = prefixes(_SUB_PREFIX_HANDLERS)

sub_prefix_sequence = ws_sequence(sub_prefix)

@rule
def bind_c(tokens):
    tokens.expect('bind')
    tokens.expect('(')
    tokens.expect('c')
    if tokens.marker(','):
        tokens.expect('name')
        tokens.expect('=')
        name = expr()
    else:
        name = None
    tokens.expect(')')
    return tokens.produce('bind_c', name)

optional_bind_c = optional(bind_c)

@rule
def contained_part(tokens):
    # contains statement
    tokens.expect('contains')
    with LockedIn(tokens):
        expect_eos(tokens)
        vals = subprogram_block(tokens)
        return tokens.produce('contains', *vals)

optional_contained_part = optional(contained_part)

@rule
def subroutine_decl(tokens):
    # Header
    prefixes = tokens.produce('sub_prefixes', sub_prefix_sequence(tokens))
    tokens.expect('subroutine')
    with LockedIn(tokens):
        name = identifier(tokens)
        if tokens.marker('('):
            args = tokens.produce('sub_args', dummy_arg_sequence(tokens))
            tokens.expect(')')
        else:
            args = tokens.produce('sub_args')
        bind_ = optional_bind_c(tokens)
        expect_eos(tokens)

        # Body
        declarations_ = declaration_part(tokens)
        execs_ = execution_part(tokens)
        contained_ = optional_contained_part(tokens)
        print ("DECLARATIONS:", declarations_)

        # Footer
        tokens.expect('end')
        if tokens.marker('subroutine'):
            optional_identifier(tokens)
        expect_eos(tokens)
        return tokens.produce('subroutine_decl', name, prefixes, args, bind_,
                            declarations_, execs_, contained_)


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

func_prefix_sequence = ws_sequence(func_prefix)

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

func_suffix_sequence = ws_sequence(func_suffix)

func_arg_sequence = optional(comma_sequence(identifier))

@rule
def function_decl(tokens):
    # Header
    prefixes = tokens.produce('func_prefixes', func_prefix_sequence(tokens))
    tokens.expect('function')
    with LockedIn(tokens):
        name = identifier(tokens)
        tokens.expect('(')
        args = tokens.produce('func_args', func_arg_sequence(tokens))
        tokens.expect(')')
        suffixes = tokens.produce('func_suffixes', func_suffix_sequence(tokens))
        expect_eos(tokens)

        # Body
        declarations_ = declaration_part(tokens)
        execs_ = execution_part(tokens)
        contained_ = optional_contained_part(tokens)

        # Footer
        tokens.expect('end')
        if tokens.marker('function'):
            optional_identifier(tokens)
        expect_eos(tokens)
        return tokens.produce('function_decl', name, prefixes, args, suffixes,
                            declarations_, execs_, contained_)

@rule
def subprogram_decl(tokens):
    try:
        return subroutine_decl(tokens)
    except NoMatch:
        return function_decl(tokens)

subprogram_block = block(subprogram_decl)

@rule
def iface_name(tokens):
    try:
        return oper_spec(tokens)
    except NoMatch:
        return identifier(tokens)

optional_iface_name = optional(iface_name)

identifier_sequence = comma_sequence(identifier)

@rule
def module_proc_stmt(tokens):
    tokens.expect('module')
    tokens.expect('procedure')
    with LockedIn(tokens):
        procs = func_arg_sequence(tokens)
        expect_eos(tokens)
        return tokens.produce('module_proc_stmt', *procs)

def interface_body_stmt(tokens):
    try:
        return module_proc_stmt(tokens)
    except NoMatch:
        return subprogram_decl(tokens)

interface_body_block = block(interface_body_stmt)

@rule
def interface_decl(tokens):
    tokens.expect('interface')
    with LockedIn(tokens):
        name = optional_iface_name(tokens)
        expect_eos(tokens)
        decls = interface_body_block(tokens)
        tokens.expect('end')
        if tokens.marker('interface'):
            optional_iface_name(tokens)
        expect_eos(tokens)
        return tokens.produce('interface_decl', name, decls)

@rule
def declaration_stmt(tokens):
    try:
        return use_stmt(tokens)
    except NoMatch: pass
    try:
        return implicit_stmt(tokens)
    except NoMatch: pass
    try:
        return interface_decl(tokens)
    except NoMatch: pass
    try:
        return type_decl(tokens)
    except NoMatch:
        return entity_stmt(tokens)
    # TODO: imbue statements are missing
    # TODO: make this faster

declaration_part = block(declaration_stmt, fenced=False)

fenced_declaration_part = block(declaration_stmt, fenced=True)

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
    cond = expr(tokens)
    tokens.expect(')')

@rule
def else_if_block(tokens):
    tokens.expect('else')
    if_clause(tokens)
    tokens.expect('then')
    optional_identifier(tokens)
    expect_eos(tokens)
    execution_part(tokens)

else_if_block_sequence = ws_sequence(else_if_block)

@rule
def else_block(tokens):
    tokens.expect('else')
    optional_identifier(tokens)
    expect_eos(tokens)
    execution_part(tokens)

optional_else_block = optional(else_block)

def ignore_stmt(tokens):
    while True:
        cat, token = next(tokens)
        if cat == lexer.CAT_EOS or cat == lexer.CAT_DOLLAR:
            return

@rule
def if_construct(tokens):
    optional_construct_tag(tokens)
    if_clause(tokens)
    with LockedIn(tokens):
        if tokens.marker('then'):
            expect_eos(tokens)
            execution_part(tokens)
            else_if_block_sequence(tokens)
            optional_else_block(tokens)
            tokens.expect('end')
            if tokens.marker('if'):
                optional_identifier(tokens)
            expect_eos(tokens)
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

@rule
def do_construct(tokens):
    optional_construct_tag(tokens)
    tokens.expect('do')
    with LockedIn(tokens):
        # TODO: non-block do
        loop_ctrl(tokens)
        expect_eos(tokens)
        execution_part(tokens)
        tokens.expect('end')
        if tokens.marker('do'):
            optional_identifier(tokens)
        expect_eos(tokens)

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

case_range_sequence = comma_sequence(case_range)

@rule
def select_case(tokens):
    tokens.expect('case')
    if tokens.marker('default'):
        pass
    else:
        tokens.expect('(')
        case_range_sequence(tokens)
        tokens.expect(')')
    expect_eos(tokens)
    execution_part(tokens)

select_case_sequence = ws_sequence(select_case)

@rule
def select_case_construct(tokens):
    optional_construct_tag(tokens)
    tokens.expect('select')
    with LockedIn(tokens):
        tokens.expect('case')
        tokens.expect('(')
        expr(tokens)
        tokens.expect(')')
        expect_eos(tokens)
        select_case_sequence(tokens)
        tokens.expect('end')
        if tokens.marker('select'):
            optional_identifier(tokens)
        expect_eos(tokens)

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

@rule
def forall_construct(tokens):
    optional_construct_tag(tokens)
    forall_clause(tokens)
    try:
        expect_eos(tokens)
    except NoMatch:
        # FORALL STMT
        ignore_stmt(tokens)
    else:
        # FORALL BLOCK
        execution_part(tokens)
        tokens.expect('end')
        if tokens.marker('forall'):
            optional_identifier(tokens)
        expect_eos(tokens)

@rule
def where_clause(tokens):
    tokens.expect('where')
    tokens.expect('(')
    expr(tokens)
    tokens.expect(')')

@rule
def where_construct(tokens):
    optional_construct_tag(tokens)
    where_clause(tokens)
    with LockedIn(tokens):
        try:
            expect_eos(tokens)
        except NoMatch:
            # WHERE STMT
            ignore_stmt(tokens)
        else:
            # WHERE BLOCK
            execution_part(tokens)
            while tokens.marker('elsewhere'):
                if tokens.marker('('):
                    expr(tokens)
                    tokens.expect(')')
                optional_identifier(tokens)
                execution_part(tokens)
            tokens.expect('end')
            if tokens.marker('where'):
                optional_identifier(tokens)
            expect_eos(tokens)

CONSTRUCT_HANDLERS = {
    'if':         if_construct,
    'do':         do_construct,
    'select':     select_case_construct,
    'forall':     forall_construct,
    'where':      where_construct
    }
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
    'format':     ignore_stmt,
    'go':         ignore_stmt,
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
def execution_stmt(tokens):
    optional_construct_tag(tokens)
    try:
        prefixed_stmt(tokens)
    except NoMatch:
        ident = tokens.expect_cat(lexer.CAT_WORD)
        if ident in _BLOCK_DELIM:
            raise NoMatch()
        ignore_stmt(tokens)

execution_part = block(execution_stmt)

@rule
def module_decl(tokens):
    tokens.expect('module')
    with LockedIn(tokens):
        name = identifier(tokens)
        expect_eos(tokens)
        decls = fenced_declaration_part(tokens)
        cont = optional_contained_part(tokens)
        tokens.expect('end')
        if tokens.marker('module'):
            optional_identifier(tokens)
        return tokens.produce('module_decl', name, decls, cont)

@rule
def program_decl(tokens):
    tokens.expect('program')
    with LockedIn(tokens):
        name = identifier(tokens)
        expect_eos(tokens)
        decls = declaration_part(tokens)
        exec_ = execution_part(tokens)
        cont = optional_contained_part(tokens)
        tokens.expect('end')
        if tokens.marker('program'):
            optional_identifier(tokens)
        return tokens.produce('program_decl', name, decls, exec_, cont)

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

program_unit_sequence = block(program_unit, fenced=False)

@rule
def compilation_unit(tokens):
    units = program_unit_sequence(tokens)
    tokens.expect_cat(lexer.CAT_DOLLAR)
    return units


if __name__ == '__main__':
    import sys
    lexre = lexer.LEXER_REGEX
    for fname in sys.argv[1:]:
        program = open(fname).read()
        slexer = lexer.tokenize_regex(lexre, program)
        tokens = TokenStream(list(slexer))
        print (compilation_unit(tokens))


#lexre = lexer.LEXER_REGEX

##program = """x(3:1, 4, 5::2) * &   ! something
##&  (3 + 5)"""
#program = "+1 + 3 * x(::1, 2:3) * (/ /) * 4 ** (5 .mybinary. 1) ** sin(.true., 1) + (/ 1, 2, (i, i=1,5), .myunary. 3 /)"
#slexer = lexer.tokenize_regex(lexre, program)
#tokens = TokenStream(list(slexer))
#print (expr(tokens))

#program = "character(kind=4, :)"
#slexer = lexer.tokenize_regex(lexre, program)
#tokens = TokenStream(list(slexer))
#print (type_spec(tokens))

#program = "(1:, :3, 1:4, 1:*)"
#slexer = lexer.tokenize_regex(lexre, program)
#tokens = TokenStream(list(slexer))
#print (shape(tokens))

#program = "dimension (1:, :3, 1:4, 1:*)"
#slexer = lexer.tokenize_regex(lexre, program)
#tokens = TokenStream(list(slexer))
#print (entity_attr(tokens))

#program = "operator(.mysomething.)"
#slexer = lexer.tokenize_regex(lexre, program)
#tokens = TokenStream(list(slexer))
#print (oper_spec(tokens))

#program = "character, value, intent('in') :: x*4(:,:) = 3\n"
#slexer = lexer.tokenize_regex(lexre, program)
#tokens = TokenStream(list(slexer))
#print (entity_stmt(tokens))

#program = "use ifort_module, only: a => b, c, operator(=)\n"
#slexer = lexer.tokenize_regex(lexre, program)
#tokens = TokenStream(list(slexer))
#print (use_stmt(tokens))

#program = "implicit integer (a-x), real*4 (c, f)\n"
#slexer = lexer.tokenize_regex(lexre, program)
#tokens = TokenStream(list(slexer))
#print (implicit_stmt(tokens))

#program = """type, public :: my_type
    #sequence
    #private

    #integer :: x(:) = (/ 3, 5, 9 /)

#end type
#"""
#slexer = lexer.tokenize_regex(lexre, program)
#tokens = TokenStream(list(slexer))
#print (type_decl(tokens))

#program = """pure subroutine abc(x, y, *)
    #use something
    #implicit none
    #integer :: x
#contains
    #pure function b() result(gaga)
    #end function
#end subroutine

#function x(r) result(a)
    #integer, dimension(:,:) :: r

    #if (something) then
        #call something_else(a, b, c)
        #m = n
    #else if (something == 3) then
    #end if
#end function
#"""
#slexer = lexer.tokenize_regex(lexre, program)
#tokens = TokenStream(list(slexer))
#print (compilation_unit(tokens))


#print (expr(tokens))
#parser = BlockParser(DefaultActions())
#print (parser.block(tokens))

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

# fbridge.parser.rules
