from .common import NoMatch, ParsingError


class ExprGrammar:
    """Expression grammar suitable for top-down operator precendence parsing.

    An expression grammar is characterized by a set of precedence levels,
    which we index from highest (0) to lowest.  At each precendence level,
    there is a list of operations, which is either a single symbol, and
    operator in conjunction with one or more subexpressions of higher
    precendence, or a enclosed expression or arbitrary precedence.  Such
    a grammar can be efficiently parsed by an `ExprParser`.
    """
    def __init__(self, *rulesets, max_cat=None):
        """Create a new expression grammar for some rulesets.

        Each item in `ruleset` corresponds to one precendence level, beginning
        with the highest precendence.  Each item in `ruleset` must itself be
        a list of rules corresponding to the valid operations at that level.

        As an example, assuming that tokens category 1 are either `+`, `-`, `*`
        or `/` and tokens category 2 are integer literal, the following is a
        grammar for simple arithmetic:

            grammar = ExprGrammar(
                [ Literal(2, 'int') ],
                [ Infix(1, '*', 'left', 'mul'), Infix(1, '/', 'left', 'div') ],
                [ Infix(1, '+', 'left', 'add'), Infix(1, '-', 'left', 'sub') ]
                )
        """
        # First, figure out how many category code there are.
        if max_cat is None:
            max_cat = max(pred.cat for ruleset in rulesets
                          for rule in ruleset for pred in rule.predicates)
        ncat = max_cat + 1

        # There is a chicken-and-egg problem here:  parser at all levels are
        # needed to be called by the handlers (e.g. for bracketed expressions),
        # yet the dispatch table in the expression parsers contain pointers to
        # those handlers.  We solve this by creating "empty" parsers.
        parsers = tuple(ExprParser(None, None) for _ in rulesets)

        # Next we iterate through the rulesets, where each ruleset corresponds
        # to a single precedence level and thus has its own parser.  Each
        # handler must have access to the parser at the current level, one
        # level lower (sub_expr) and highest level (full_expr).
        full_expr = parsers[-1]
        sub_table = {
            'head': [_NeverDispatcher()] * ncat,
            'tail': [_NeverDispatcher()] * ncat,
            }
        sub_expr = ExprParser(
            tuple(disp.compile() for disp in sub_table['head']),
            tuple(disp.compile() for disp in sub_table['tail'])
            )
        for self_expr, ruleset in zip(parsers, rulesets):
            self_table = {}
            for rule in ruleset:
                handler = rule.handler(full_expr, self_expr, sub_expr)
                for pred in rule.predicates:
                    try:
                        place_table = self_table[pred.place]
                    except KeyError:
                        place_table = [disp.clone() for disp
                                       in sub_table[pred.place]]
                        self_table[pred.place] = place_table

                    disp = place_table[pred.cat]
                    try:
                        disp.add(pred, handler)
                    except _NeverDispatcher.CannotAdd:
                        disp = pred.dispatcher()
                        place_table[pred.cat] = disp
                        disp.add(pred, handler)

            # Fill the expression parsers with the compiled dispatch table.
            # Here, we reuse the compiled tables from the lower levels to
            # reduce memory use and thus lower cache pressure.
            def compiled_table(place):
                try:
                    place_table = self_table[place]
                except KeyError:
                    self_table[place] = sub_table[place]
                    return getattr(sub_expr, place)
                else:
                    return tuple(disp.compile() for disp in place_table)

            self_expr.head = compiled_table('head')
            self_expr.tail = compiled_table('tail')

            # Ascend by one precedence level
            sub_expr = self_expr
            sub_table = self_table

        self.rulesets = tuple(rulesets)
        self.levels = parsers
        self.dispatch_table = self_table

    @property
    def parser(self):
        """Expression parser for this grammar."""
        return self.levels[-1]


class Rule:
    """Production rule for part of an expression.

    Unlike in recursive descent parser, top-down operator precedence rules have
    two parts: a set of `predciates`, which describes the matching conditions,
    and a handler, which is triggered once a predicate matches.
    """
    @property
    def predicates(self):
        """Sequence of `Predicate` instances, any one of which may match"""
        raise NotImplementedError()

    def handler(self, full_expr, self_expr, sub_expr):
        """Return a handler function.

        Handler functions are a core part of a top-down expression parser.
        Their signature typically is:

            def handler_function(tokens, lhs=None):
                return (...)

        `tokens` is the token stream parsed, with the cursor at the token on
        which one of the predicates matched.  If we are in the middle of a
        subexpression (`place == "tail"`), then `lhs` is the result of the
        previous handler, otherwise (`place == "head"`), `lhs` is absent.
        """
        raise NotImplementedError()


class Infix(Rule):
    """Rule for infix operation `lhs (op) rhs`"""
    def __init__(self, cat, token, assoc, tag, ignore_case=False):
        self.cat = cat
        self.token = token
        self.assoc = assoc
        self.tag = tag
        self.ignore_case = ignore_case

    @property
    def predicates(self):
        P = CaseInsensitivePredicate if self.ignore_case else LiteralPredicate
        return P('tail', self.cat, self.token),

    def handler(self, full_expr, self_expr, sub_expr):
        tag = self.tag
        rhs_expr = sub_expr if self.assoc == 'left' else self_expr
        def handle_infix_op(tokens, lhs):
            tokens.advance()
            rhs = rhs_expr(tokens)
            return tag, lhs, rhs
        return handle_infix_op


class Prefix(Rule):
    """Rule for prefix operation `(op) rhs`"""
    def __init__(self, cat, token, tag, ignore_case=False):
        self.cat = cat
        self.token = token
        self.tag = tag
        self.ignore_case = ignore_case

    @property
    def predicates(self):
        P = CaseInsensitivePredicate if self.ignore_case else LiteralPredicate
        return P('head', self.cat, self.token),

    def handler(self, full_expr, self_expr, sub_expr):
        tag = self.tag
        def handle_infix_op(tokens):
            tokens.advance()
            rhs = self_expr(tokens)
            return tag, rhs
        return handle_infix_op


class Parenthesized(Rule):
    def __init__(self, cat, begin_token, end_token):
        self.cat = cat
        self.begin_token = begin_token
        self.end_token = end_token

    @property
    def predicates(self):
        return LiteralPredicate('head', self.cat, self.begin_token),

    def handler(self, full_expr, self_expr, sub_expr):
        end_token = self.end_token
        def handle_parenthesized(tokens, lhs=None):
            tokens.advance()
            inner = full_expr(tokens)
            if next(tokens)[3] != end_token:
                raise NoMatch()
            return inner
        return handle_parenthesized


class Literal(Rule):
    """Rule for a literal (number, identifier, etc.)"""
    def __init__(self, cat, tag):
        self.cat = cat
        self.tag = tag

    @property
    def predicates(self):
       return CategoryPredicate('head', self.cat),

    def handler(self, full_expr, self_expr, sub_expr):
        tag = self.tag
        def handle_literal(tokens):
            return tag, next(tokens)[3]
        return handle_literal


class Predicate:
    """Matching part of a production rule.

    A predicate is a matching rule for tokens, consisting of three parts:

      1. `place`: place in the subexpression, either at the `head` (first
         token) or `tail` (subsequent token).
      2. `cat`: category code of the tokens
      3. `dispatcher`: for a given place and category, predicate for the
         token text themselves
    """
    def __init__(self, place, cat):
        if place not in {'head', 'tail'}:
            raise ValueError("place must be either 'head' or 'tail'")
        if cat != int(cat) or cat <= 0:
            raise ValueError("cat must be a positive integer")
        self.place = place
        self.cat = cat

    def dispatcher(self):
        return self.Dispatcher()

    class Dispatcher:
        def __init__(self):
            raise NotImplementedError()

        def clone(self):
            raise NotImplementedError()

        def add(self, predicate, handler):
            raise NotImplementedError()

        def compile(self):
            raise NotImplementedError()


class _NeverDispatcher:
    class CannotAdd(Exception): pass

    def clone(self):
        return self

    def add(self, predicate, handler):
        raise _NeverDispatcher.CannotAdd()

    def compile(self):
        def unknown_cat_code(tokens):
            raise NoMatch()
        return unknown_cat_code


class CategoryPredicate(Predicate):
    """A predicate that matches a full category code.

    This is useful for, e.g., integer literals, where the rule does not depend
    on the token text itself, but only on its category.
    """
    class Dispatcher:
        def __init__(self):
            self.handler = None
            self._compiled = None

        def clone(self):
            result = self.__class__()
            result.handler = self.handler
            result._compiled = self._compiled
            return result

        def add(self, predicate, handler):
            if not isinstance(predicate, CategoryPredicate):
                raise ValueError("Incompatible dispatch to category")
            if self.handler is not None:
                raise ValueError("only one dispatch allowed per cat")
            self.handler = handler

        def compile(self):
            handler = self.handler
            if handler is None:
                raise ValueError("handler is not yet set")
            if self._compiled is None:
                self._compiled = lambda _: handler
            return self._compiled


class LiteralPredicate(Predicate):
    """Predicate which matches a category and token text.

    This is useful for, e.g., symbolic operators and textual operators in
    case-sensitive languages.
    """
    def __init__(self, place, cat, token):
        super().__init__(place, cat)
        self.token = token

    class Dispatcher:
        def __init__(self):
            self.tokens = {}
            self._compiled = None

        def clone(self):
            result = self.__class__()
            result.tokens = dict(self.tokens)
            result._compiled = self._compiled
            return result

        def add(self, predicate, handler):
            if not isinstance(predicate, LiteralPredicate):
                raise ValueError("incompatible dispatch to literal")

            token = predicate.token
            if token in self.tokens:
                raise ValueError("already in there")
            self.tokens[token] = handler
            self._compiled = None      # invalidate cache

        def compile(self):
            tokens_get = self.tokens.__getitem__
            def literal_dispatch(token):
                try:
                    return tokens_get(token)
                except KeyError:
                    raise NoMatch()
            if self._compiled is None:
                self._compiled = literal_dispatch
            return self._compiled


class CaseInsensitivePredicate(Predicate):
    """Predicate which matches a category and token text (case insensitive).

    This is useful for textual operators in case-insensitive languages.
    """
    def __init__(self, place, cat, token):
        super().__init__(place, cat)
        self.token = token.lower()

    class Dispatcher:
        def __init__(self):
            self.tokens = {}
            self._compiled = None

        def clone(self):
            result = self.__class__()
            result.tokens = self.tokens
            result._compiled = self._compiled
            return result

        def add(self, predicate, handler):
            if not isinstance(predicate, CaseInsensitivePredicate):
                raise ValueError("incompatible dispatch to case-insensitive")

            token = predicate.token
            if token in self.tokens:
                raise ValueError("already in there")
            self.tokens[token] = handler
            self._compiled = None

        def compile(self):
            tokens_get = self.tokens.__getitem__
            def case_insensitive_dispatch(token):
                try:
                    return tokens_get(token.lower())
                except KeyError:
                    raise NoMatch()
            if self._compiled is None:
                self._compiled = case_insensitive_dispatch
            return self._compiled


class ExprParser:
    """Top-down operator precedence parser.

    This is the actual workhorse of the parsing process.  It is a top-down
    operator precendence parser [1], which exploits the special "nested"
    structure of expressions to do away with the need for backtracking in
    recursive descent parsers.  It thus guarantees linear runtime.

    Like the LL/LR family of parsers, the parser is a finite state transducer:
    it is in the "head" state at the start of the expression and in the "tail"
    state otherwise.  In each state, it peeks at the next token and uses the
    respective dispatch table (`head` or `tail`) to decide which handler to
    call.  The handler then handles the token and any arguments.

    [1]: https://doi.org/10.1145/512927.512931
    """
    __slots__ = 'head', 'tail'

    def __init__(self, head, tail):
        """Construct an expression parser from dispatch tables.

        This function is only for advanced users; use `ExprGrammar.parser()`
        for a more friendly way to construct these parsers.
        """
        self.head = head
        self.tail = tail

    def __call__(self, tokens):
        """Try matching an expression for given `tokens` sequence."""
        # Parser is in "head" state.  Peek at next token and get associated
        # handler.  NoMatch exceptions are forwarded, signifying that this
        # is not an expression.
        lineno, colno, cat, token = tokens.peek()
        handler = self.head[cat](token)
        try:
            result = handler(tokens)

            # Parser is in "tail" state.  We are now trying to find an
            # appropriate infix or suffix symbol, and call the associated
            # handler.  This is repeated to get, e.g., chains of same-priority
            # operations.
            while True:
                lineno, colno, cat, token = tokens.peek()
                try:
                    handler = self.tail[cat](token)
                except NoMatch:
                    return result
                result = handler(tokens, result)
        except NoMatch:
            raise ParsingError(tokens.fname, lineno, colno, colno,
                               tokens.current_line(), "Invalid expression")
