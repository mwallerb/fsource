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
            if self.pos > len(self.tokens):
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
    tokens.expect_cat(lexer.CAT_EOS)

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

def choice(*rules):
    def choice_rule(tokens):
        for current in rules:
            try:
                return current(tokens)
            except NoMatch:
                pass
        raise NoMatch()

    return choice_rule

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
        kind_ = int_(tokens)
    else:
        tokens.expect('(')
        if tokens.marker('kind'):
            tokens.expect('=')
        kind_ = int_(tokens)
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
            len_ = char_len()
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
    tokens.expect('(')
    typename = identifier(tokens)
    tokens.expect(')')
    return typename

_TYPE_SPEC_HANDLERS = {
    'integer':   kind_selector,
    'real':      kind_selector,
    'double':    lambda t: t.expect('precision'),
    'complex':   kind_selector,
    'character': char_selector,
    'logical':   kind_selector,
    'type':      _typename_handler
    }

@rule
def type_spec(tokens):
    prefix = tokens.expect_cat(lexer.CAT_WORD)
    try:
        contd = _TYPE_SPEC_HANDLERS[prefix]
    except KeyError:
        raise NoMatch()
    try:
        arg = contd(tokens)
    except NoMatch:
        arg = None
    return tokens.produce('type_spec', prefix, arg)

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
    tokens.expect('(')
    string = lexer.parse_string(tokens.expect_cat(lexer.CAT_STRING)).lower()
    try:
        in_, out = _INTENT_STRINGS[string]
    except KeyError:
        raise NoMatch()
    tokens.expect(')')
    return tokens.produce('intent', in_, out)

_ENTITY_ATTR_HANDLERS = {
    'parameter':   lambda tokens: ('parameter'),
    'public':      lambda tokens: tokens.produce('visible', True),
    'private':     lambda tokens: tokens.produce('visible', False),
    'allocatable': lambda tokens: tokens.produce('allocatable'),
    'dimension':   lambda tokens: shape(tokens),
    'external':    lambda tokens: tokens.produce('external'),
    'intent':      lambda tokens: intent(tokens),
    'intrinsic':   lambda tokens: tokens.produce('intrinsic'),
    'optional':    lambda tokens: tokens.produce('optional'),
    'pointer':     lambda tokens: tokens.produce('pointer'),
    'save':        lambda tokens: tokens.produce('save'),
    'target':      lambda tokens: tokens.produce('target'),
    'value':       lambda tokens: tokens.produce('value'),
    'volatile':    lambda tokens: tokens.produce('volatile'),
    }

@rule
def attribute(tokens, handler_dict):
    prefix = tokens.expect_cat(lexer.CAT_WORD)
    try:
        handler = handler_dict[prefix]
    except KeyError:
        raise NoMatch()
    else:
        return handler(tokens)

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
        attrs.append(attribute(tokens, handler_dict))
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
    print (tokens.peek())
    entities = entity_sequence(tokens)
    expect_eos(tokens)
    return tokens.produce('entity_stmt', type_, attrs_, *entities)

@rule
def entity_ref(tokens):
    name = identifier(tokens)
    shape_ = optional_shape(tokens)
    return tokens.produce('entity_ref', name, shape_)

_TYPE_ATTR_HANDLERS = {
    'public':      lambda tokens: tokens.produce('visible', True),
    'private':     lambda tokens: tokens.produce('visible', False),
    }

@rule
def type_attrs(tokens):
    handler_dict = _TYPE_ATTR_HANDLERS
    attrs = []
    while tokens.marker(','):
        attrs.append(attribute(tokens, handler_dict))
    try:
        double_colon(tokens)
    except NoMatch:
        if attrs: raise NoMatch()
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

def block(rule):
    # Fortran blocks are delimited by one of these words, so we can use
    # them in failing fast
    block_delim = { 'end', 'else', 'elsewhere', 'contains' }
    def block_rule(tokens):
        stmts = []
        while True:
            cat, token = tokens.peek()
            if cat == lexer.CAT_EOS:
                next(tokens)
            elif cat == lexer.CAT_PREPROC:
                stmts.append(preproc_stmt(tokens))
            elif token.lower() in block_delim:
                break
            else:
                try:
                    stmts.append(rule(tokens))
                except NoMatch:
                    break
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

dummy_arg_sequence = comma_sequence(dummy_arg)

_SUB_PREFIX_HANDLERS = {
    'impure':    lambda tokens: tokens.produce('pure', False),
    'pure':      lambda tokens: tokens.produce('pure', True),
    'recursive': lambda tokens: tokens.produce('recursive'),
    }

@rule
def sub_prefix(tokens):
    cat, token = next(tokens)
    try:
        handler = _SUB_PREFIX_HANDLERS[token.lower()]
    except KeyError:
        raise NoMatch()
    else:
        return handler(tokens)

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
    expect_eos(tokens)
    vals = subprogram_block(tokens)
    return tokens.produce('contains', *vals)

optional_contained_part = optional(contained_part)

@rule
def subroutine_decl(tokens):
    # Header
    prefixes = tokens.produce('sub_prefixes', sub_prefix_sequence(tokens))
    tokens.expect('subroutine')
    name = identifier(tokens)
    tokens.expect('(')
    args = tokens.produce('sub_args', dummy_arg_sequence(tokens))
    tokens.expect(')')
    bind_ = optional_bind_c(tokens)
    expect_eos(tokens)

    # Body
    declarations_ = declaration_part(tokens)
    execs_ = execution_part(tokens)
    contained_ = optional_contained_part(tokens)

    # Footer
    tokens.expect('end')
    if tokens.marker('subroutine'):
        optional_identifier(tokens)
    return tokens.produce('subroutine_decl', name, prefixes, args, bind_,
                          declarations_, execs_, contained_)

@rule
def function_decl(tokens):
    raise NotImplementedError()

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
    procs = identifier_sequence(tokens)
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
    name = iface_name(tokens)
    expect_eos(tokens)
    decls = interface_body_block(tokens)
    tokens.expect('end')
    if tokens.marker('interface'):
        optional_iface_name(tokens)
    return tokens.produce('interface_decl', name)

@rule
def declaration_stmt(tokens):
    try:
        return use_stmt(tokens)
    except NoMatch: pass
    try:
        return implicit_stmt(tokens)
    except NoMatch: pass
    try:
        return type_decl(tokens)
    except NoMatch: pass
    try:
        return entity_stmt(tokens)
    except NoMatch: pass
    # TODO: imbue statements are missing
    # TODO: make this faster

declaration_part = block(declaration_stmt)

@rule
def execution_stmt(tokens):
    # FIXME
    raise NoMatch()

execution_part = block(execution_stmt)

lexre = lexer.LEXER_REGEX

#program = """x(3:1, 4, 5::2) * &   ! something
#&  (3 + 5)"""
program = "+1 + 3 * x(::1, 2:3) * (/ /) * 4 ** (5 .mybinary. 1) ** sin(.true., 1) + (/ 1, 2, (i, i=1,5), .myunary. 3 /)"
slexer = lexer.tokenize_regex(lexre, program)
tokens = TokenStream(list(slexer))
print (expr(tokens))

program = "character(kind=4, :)"
slexer = lexer.tokenize_regex(lexre, program)
tokens = TokenStream(list(slexer))
print (type_spec(tokens))

program = "(1:, :3, 1:4, 1:*)"
slexer = lexer.tokenize_regex(lexre, program)
tokens = TokenStream(list(slexer))
print (shape(tokens))

program = "dimension (1:, :3, 1:4, 1:*)"
slexer = lexer.tokenize_regex(lexre, program)
tokens = TokenStream(list(slexer))
print (attribute(tokens, _ENTITY_ATTR_HANDLERS))

program = "operator(.mysomething.)"
slexer = lexer.tokenize_regex(lexre, program)
tokens = TokenStream(list(slexer))
print (oper_spec(tokens))

program = "character, value, intent('in') :: x*4(:,:) = 3\n"
slexer = lexer.tokenize_regex(lexre, program)
tokens = TokenStream(list(slexer))
print (entity_stmt(tokens))

program = "use ifort_module, only: a => b, c, operator(=)\n"
slexer = lexer.tokenize_regex(lexre, program)
tokens = TokenStream(list(slexer))
print (use_stmt(tokens))

program = "implicit integer (a-x), real*4 (c, f)\n"
slexer = lexer.tokenize_regex(lexre, program)
tokens = TokenStream(list(slexer))
print (implicit_stmt(tokens))

program = """type, public :: my_type
    sequence
    private

    integer :: x(:) = (/ 3, 5, 9 /)

end type
"""
slexer = lexer.tokenize_regex(lexre, program)
tokens = TokenStream(list(slexer))
print (type_decl(tokens))

program = """pure subroutine abc(x, y, *)
    use something
    implicit none
    integer :: x
contains
end subroutine
"""
slexer = lexer.tokenize_regex(lexre, program)
tokens = TokenStream(list(slexer))
print (subroutine_decl(tokens))

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
