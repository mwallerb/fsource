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


def tokenize_regex(regex, text, lineno=None):
    """Tokenizes text using the groups in the regex specified

    Expects a `regex`, where different capturing groups correspond to different
    categories of tokens.  Iterates through all matches of the regex on `text`,
    returning the highest matching category with the associated token (group
    text).
    """
    try:
        for match in regex.finditer(text):
            cat = match.lastindex
            yield lineno, match.start(cat), cat, match.group(cat)
    except IndexError:
        raise LexerError(None, lineno, match.start(), match.end(), text,
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
                         | [XP]
                         )(?=\s*[:/,)])"""

    fortran_token = r"""(?ix)
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
                )
    return re.compile(fortran_token)


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

CAT_NAMES = ('eof', 'word', 'symop', 'eos', 'int', 'float',
             'bool', 'dotop', 'custom_dotop', 'string', 'radix',
             'format', 'preproc')


def _string_lexer_regex(quote):
    pattern = r"""(?x) ({quote}{quote}) | ([^{quote}]+)""".format(quote=quote)
    return re.compile(pattern)


def _string_lexer_actions():
    return (None,
            lambda tok: tok[0],
            lambda tok: tok,
            )


STRING_LEXER_REGEX = {
    "'": _string_lexer_regex("'"),
    '"': _string_lexer_regex('"'),
    }
STRING_LEXER_ACTIONS = _string_lexer_actions()


def parse_string(tok):
    """Translates a Fortran string literal to a Python string"""
    actions = STRING_LEXER_ACTIONS
    return "".join(actions[cat](token) for (_, _, cat, token)
                   in tokenize_regex(STRING_LEXER_REGEX[tok[0]], tok[1:-1]))


CHANGE_D_TO_E = _maketrans('dD', 'eE')


def parse_float(tok):
    """Translates a Fortran real literal to a Python float"""
    return float(tok.translate(CHANGE_D_TO_E))


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
                for token_tuple in tokenize_regex(lexer_regex, line, lineno):
                    yield token_tuple
            except LexerError as e:
                e.fname = fname
                raise e

    # Make sure last line is terminated, then yield terminal token
    yield lineno+1, 0, CAT_EOS, '\n'
    yield lineno+1, 0, CAT_DOLLAR, '<$>'


def lex_snippet(fstring):
    """Perform lexical analysis of parts of a line"""
    return tuple(tokenize_regex(get_lexer_regex(), fstring)) \
           + ((None, len(fstring), CAT_DOLLAR, ''),)
