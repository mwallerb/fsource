from . import common

NoMatch = common.NoMatch


class ExprParser:
    @classmethod
    def create(cls, sub_expr, *pattern_groups):
        if sub_expr is None:
            sub_expr = ExprParser([], [])
        full_expr = ExprParser([], [])
        for pattern_group in pattern_groups:
            self_expr = ExprParser.inherit(sub_expr)
            for pattern in pattern_group:
                handler = pattern.make_handler(full_expr, self_expr, sub_expr)
                pattern.register(self_expr, handler)
            sub_expr = self_expr

        full_expr.reseat(self_expr)
        return full_expr

    @classmethod
    def inherit(cls, inner):
        return cls([None if d is None else d.copy() for d in inner._head],
                   [None if d is None else d.copy() for d in inner._tail])

    def __init__(self, head, tail):
        self._head = head
        self._tail = tail

    def reseat(self, other):
        self._head = other._head
        self._tail = other._tail

    def register(self, bind, cat, cat_dispatch, token, handler):
        if cat >= len(self._head):
            padding = [None] * (cat + 1 - len(self._head))
            self._head += padding
            self._tail += padding

        if bind == 'head':
            target = self._head
        elif bind == 'tail':
            target = self._tail
        else:
            raise ValueError("Invalid bind target")

        DispatchType = DISPATCH_REGISTRY[cat_dispatch]
        if target[cat] is None:
            target[cat] = DispatchType()
        elif not isinstance(target[cat], DispatchType):
            raise ValueError("Inconsistent dispatch type")
        target[cat].register(token, handler)

    def __call__(self, tokens):
        _, _, cat, token = tokens.peek()
        try:
            handler = self._head[cat](token)
        except:
            raise NoMatch()
        try:
            result = handler(tokens)

            # cycle through appropriate infixes
            while True:
                _, _, cat, token = tokens.peek()
                try:
                    handler = self._tail[cat](token)
                except:
                    return result
                result = handler(tokens, result)
        except NoMatch:
            raise ParserError(tokens, "Invalid expression")


class Pattern:
    def register(self, parser, handler):
        raise NotImplementedError()

    def make_handler(self, full_expr, self_expr, sub_expr):
        raise NotImplementedError()


class InfixOperator(Pattern):
    def __init__(self, cat, token, assoc, tag):
        self.cat = cat
        self.token = token
        self.assoc = assoc
        self.tag = tag

    def register(self, parser, handler):
        parser.register('tail', self.cat, 'literal', self.token, handler)

    def make_handler(self, full_expr, self_expr, sub_expr):
        tag = self.tag
        rhs_expr = sub_expr if self.assoc == 'left' else self_expr

        def handle_infix_op(tokens, lhs):
            tokens.advance()
            rhs = rhs_expr(tokens)
            return tokens.produce(tag, lhs, rhs)
        return handle_infix_op


class Literal(Pattern):
    def __init__(self, cat, tag):
        self.cat = cat
        self.tokens = None
        self.token_sel = 'none'
        self.bind = 'head'

        self.tag = tag

    def register(self, parser, handler):
        parser.register('head', self.cat, 'none', None, handler)

    def make_handler(self, full_expr, self_expr, sub_expr):
        tag = self.tag
        def handle_literal(tokens):
            return tokens.produce(tag, next(tokens)[3])

        return handle_literal


class SelfDispatch:
    def __init__(self, handler=None):
        self._handler = handler

    def copy(self):
        return SelfDispatch(self._handler)

    def register(self, token, handler):
        if token is not None:
            raise ValueError("token must be None")
        self._handler = handler

    def __call__(self, token):
        return self._handler


class LiteralDispatch:
    def __init__(self, tokens={}):
        self._tokens = dict(tokens)

    def copy(self):
        return LiteralDispatch(self._tokens)

    def register(self, token, handler):
        self._tokens[token] = handler

    def __call__(self, token):
        return self._tokens[token]


class CaseInsensitiveDispatch:
    def __init__(self, tokens={}):
        self._tokens = dict(tokens)

    def copy(self):
        return CaseInsensitiveDispatch(self._tokens)

    def register(self, token, handler):
        self._tokens[token.lower()] = handler

    def __call__(self, token):
        return self._tokens[token.lower()]


DISPATCH_REGISTRY = {
    'none': SelfDispatch,
    'literal': LiteralDispatch,
    'nocase': CaseInsensitiveDispatch,
    }
