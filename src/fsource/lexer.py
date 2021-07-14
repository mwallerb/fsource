"""
Lexical analysis for free-form Fortran, up to version 2008.

Uses regular expressions to split up a Fortran source file into a sequence
of tokens, where each token is has a category attached to it signifying its
type (literal, identifier, comment, etc.).

Generate pairs of the form `(cat, token)`, where `cat` is a category code
between 0 and 12 signifying the type of token returned, and `token` is the
portion of source code matching the token.  Whitespace is generally
ignored.

Lexical analysis must deal with three ambiguities in the Fortran grammar:

 1. The string '::' can mean an empty slice or a separator token.  The
    lexer always returns single ':', which means one gets '::' as a
    sequence of two ':' and cannot detect whitespace within a seperator.

 2. '(//)' can mean an empty inplace array or an overloaded '//' operator,
    and '(/)' is ambiguous for a similar reason.  The lexer acts greedy in
    this case, returning tokens '(/' and '/)' if found.  The application must
    then disambiguate.

 3. The 'FORMAT' statement is a bit of an oddball, as it allows tokens that
    are illegal everywhere else, e.g., 3I6 or ES13.2.  The lexer works
    around this by returning a special `CAT_FORMAT` token for such specifiers,
    and the parser must ensure they appear in the format statement.

Copyright 2019 Markus Wallerberger
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
from __future__ import print_function
import sys
import re

from . import common
from . import splicer

# Python 2/3 compatibility
if sys.version_info >= (3,):
    _str_maketrans = str
else:
    import string
    _str_maketrans = string

_maketrans = _str_maketrans.maketrans


class LexerError(common.ParsingError):
    @property
    def error_type(self): return "lexer error"



class RegexLexer:
    """Lexical analysis using regular expressions.

    Lexical analysis splits a text into a sequence of non-overlapping lexical
    units called tokens.  Each token is characterized by a token category code
    (an integer starting from 1) and the token text, i.e., the substring
    corresponding to that token.  Whitespace between tokens is silently
    discarded.  Token category 0 can be used to mark the end of file.

    For each category, one specifies a name and regular expression.  The
    lexer than moves through the text, attempting to match the regular
    expressions for each category, starting from 1.  After matching, it
    restarts the match after the token found.
    """
    @classmethod
    def create(cls, *categories, whitespace=r'', re_opts=re.X):
        cat_names, cat_regex = zip(*categories)
        group_re = "|".join("(%s)" % cat for cat in cat_regex)
        full_re = "(?:%s)(?:%s|[^\t ]+)" % (whitespace, group_re)
        full_re = re.compile(full_re, re_opts)
        return cls(full_re, cat_names)

    def __init__(self, full_regex, cat_names):
        cat_names = tuple(cat_names)
        if full_regex.groups != len(cat_names):
            raise ValueError("Number of capturing groups is inconsistent "
                             "with number of categories.")

        self.cat_name = 'END_OF_FILE', cat_names
        self.cat_code = {name: code for (code, name)
                         in enumerate(cat_names, start=1)}
        self._full_regex = full_regex
        self._finditer = full_regex.finditer

    def line_tokens(self, line, lineno=None):
        """Tokenizes text using the groups in the regex specified

        Iterates through all matches of the regex on `text`, returning the
        highest matching category with the associated token (group text).
        """
        try:
            for match in self._finditer(line):
                cat = match.lastindex
                yield lineno, match.start(cat), cat, match.group(cat)
        except IndexError:
            raise LexerError(None, lineno, match.start(), match.end(), line,
                             "invalid token")

    def transform(self, text, actions):
        """Splits text into tokens and transforms them"""
        try:
            for match in self._finditer(text):
                cat = match.lastindex
                yield actions[cat](match.group(cat))
        except IndexError:
            if len(actions) <= cat:
                raise ValueError("No action registered for category %d" % cat)
            raise LexerError(None, None, match.start(), match.end(), text,
                             "invalid token")



def get_lexer_regex():
    """Return regular expression for parsing free-form Fortran 2008"""
    endline = r"""(?:\n|\r\n?)"""
    comment = r"""(?:![^\r\n]*)"""
    skip_ws = r"""[\t ]*"""
    postq = r"""(?!['"\w])"""
    dq_string = r""""(?:""|[^"\r\n])*"{postq}""".format(postq=postq)
    sq_string = r"""'(?:''|[^'\r\n])*'{postq}""".format(postq=postq)
    postnum = r"""(?= [^.'"0-9A-Za-z]
                    | \.\s*[a-zA-Z]+\s*\.
                    | $
                    )"""
    integer = r"""\d+{postnum}""".format(postnum=postnum)
    decimal = r"""(?:\d+\.\d*|\.\d+)"""
    exponent = r"""(?:[dDeE][-+]?\d+)"""
    real = r"""(?:{decimal}{exponent}?|\d+{exponent}){postnum}""" \
           .format(decimal=decimal, exponent=exponent, postnum=postnum)
    binary = r"""[Bb](?:'[01]+'|"[01]+"){postq}""".format(postq=postq)
    octal = r"""[Oo](?:'[0-7]+'|"[0-7]+"){postq}""".format(postq=postq)
    hexadec = r"""[Zz](?:'[0-9A-Fa-f]+'|"[0-9A-Fa-f]+"){postq}""" \
              .format(postq=postq)
    operator = r"""\(/?|\)|[-+,:_%\[\]]|=[=>]?|\*\*?|\/[\/=)]?|[<>]=?"""
    builtin_dot = r"""(?:eq|ne|l[te]|g[te]|n?eqv|not|and|or)"""
    dotop = r"""[A-Za-z]+"""
    word = r"""[A-Za-z][A-Za-z0-9_]*(?![A-Za-z0-9_'"])"""
    formattok = r"""\d*(?: [IBOZ] \d+ (?: \.\d+)?
                         | [FD]   \d+     \.\d+
                         | E[NS]? \d+     \.\d+  (?: E\d+)?
                         | G      \d+ (?: \.\d+  (?: E\d+)?)?
                         | L      \d+
                         | A      \d*
                         | X
                         )(?=\s*[:/,)])
                  | (?:\d+P)      (?=\s*,?)"""  # Matches scale factor P
                                                # followed by optional comma
    fortran_token = re.compile(r"""(?ix)
          {skipws}(?:
            ({word})                            #  1 word
          | ({operator})                        #  2 symbolic operator
          | (; | {comment}?{endline})           #  3 end of statement
          | ({int})                             #  4 ints
          | ({real})                            #  5 real
          | \.\s* (?:
              ( true | false )                  #  6 boolean
            | ( {builtin_dot} )                 #  7 built-in dot operator
            | ( {dotop} )                       #  8 custom dot operator
            ) \s*\.
          | ({sqstring} | {dqstring})           #  9 strings
          | ({binary} | {octal} | {hex})        # 10 radix literals
          | ({format})                          # 11 format specifier
          | [^ \t]+                             #    invalid token
          )
        """.format(
                skipws=skip_ws, endline=endline, comment=comment,
                sqstring=sq_string, dqstring=dq_string,
                real=real, int=integer, binary=binary, octal=octal,
                hex=hexadec, operator=operator, builtin_dot=builtin_dot,
                dotop=dotop, word=word, format=formattok
                ))
    cat_names = ('word', 'symop', 'eos', 'int', 'float',
                 'bool', 'dotop', 'custom_dotop', 'string', 'radix',
                 'format')
    return RegexLexer(fortran_token, cat_names)


CAT_DOLLAR = 0
CAT_WORD = 1
CAT_SYMBOLIC_OP = 2
CAT_EOS = 3
CAT_INT = 4
CAT_FLOAT = 5
CAT_BOOLEAN = 6
CAT_BUILTIN_DOT = 7
CAT_CUSTOM_DOT = 8
CAT_STRING = 9
CAT_RADIX = 10
CAT_FORMAT = 11
CAT_PREPROC = 12
CAT_MAX = 12

CAT_NAMES = ('eof', 'word', 'symop', 'eos', 'int', 'float',
             'bool', 'dotop', 'custom_dotop', 'string', 'radix',
             'format', 'preproc')


STRING_LEXER = {
    "'": RegexLexer.create(("NORMAL", "[^']+|^'|'$"), ("ESCAPED",  "''")),
    '"': RegexLexer.create(("NORMAL", '[^"]+|^"|"$'), ("ESCAPED",  '""'))
    }
STRING_ACTIONS = (
    None,
    lambda tok: tok,
    lambda tok: tok[1],
    )


def parse_string(tok):
    """Translates a Fortran string literal to a Python string"""
    lexer = STRING_LEXER[tok[0]]
    return lexer.transform(STRING_ACTIONS)


CHANGE_D_TO_E = _maketrans('dD', 'eE')


def parse_float(tok):
    """Translates a Fortran real literal to a Python float"""
    try:
        return float(tok), False
    except ValueError:
        # DOUBLE PRECISION have D instead of E for precision
        tok = tok.translate(CHANGE_D_TO_E)
        return float(tok), True


def parse_bool(tok):
    """Translates a Fortran boolean literal to a Python boolean"""
    return {'true': True, 'false': False}[tok.lower()]


def parse_radix(tok):
    """Parses a F03-style x'***' literal"""
    base = {'b': 2, 'o': 8, 'z': 16}[tok[0].lower()]
    return int(tok[2:-1], base)


def lex_buffer(mybuffer, form=None):
    """Perform lexical analysis for an opened free-form Fortran file."""
    # check for buffer
    if isinstance(mybuffer, str):
        raise ValueError("Expect open file or other sequence of lines")
    if form is None:
        form, _ = common.guess_form(mybuffer.name)

    lexer_regex = get_lexer_regex()
    linecat_preproc = splicer.LINECAT_PREPROC
    lines_iter = splicer.get_splicer(form)

    fname = mybuffer.name
    lineno = 0
    for lineno, linecat, line in lines_iter(mybuffer):
        if linecat == linecat_preproc:
            yield lineno, 0, CAT_PREPROC, line
        else:
            try:
                for token_tuple in lexer_regex.line_tokens(line, lineno):
                    yield token_tuple
            except LexerError as e:
                e.fname = fname
                raise e

    # Make sure last line is terminated, then yield terminal token
    yield lineno+1, 0, CAT_EOS, '\n'
    yield lineno+1, 0, CAT_DOLLAR, '<$>'


def lex_snippet(fstring):
    """Perform lexical analysis of parts of a line"""
    return tuple(get_lexer_regex().line_tokens(fstring)) \
                 + ((None, len(fstring), CAT_DOLLAR, ''),)
