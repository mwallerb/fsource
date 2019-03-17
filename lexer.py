#!/usr/bin/env python
"""
Lexical analysis for free-form Fortran, up to version 2008.

Uses regular expressions to split up a Fortran source file into a sequence
of tokens, where each token is has a category attached to it signifying its
type (literal, identifier, comment, etc.).

Generate pairs of the form `(cat, token)`, where `cat` is a category code
between 0 and 12 signifying the type of token returned, and `token` is the
portion of source code matching the token.  Whitespace is generally
ignored.

Lexical analysis must deal with two ambiguities in the Fortran grammar:

 1. The string '::' can mean an empty slice or a separator token.  The
    lexer always returns single ':', which means one gets '::' as a
    sequence of two ':' and cannot detect whitespace within a seperator.

 2. '(//)' can mean an empty inplace array or an overloaded '//' operator,
    and '(/)', which is ambiguous for a similar reason.  To work around
    this lexer will return a token of category `CAT_BRACKETED_SLASH`, and
    the application must disambiguate.

Author: Markus Wallerberger
"""
from __future__ import print_function
import sys
import string
import re

class LexerError(RuntimeError):
    def __init__(self, text, pos):
        self.text = text
        self.pos = pos
        RuntimeError.__init__(self,
            "Lexer error at character %d:\n%s" % (pos, text[pos:pos+70]))

def tokenize_regex(regex, text):
    """Tokenizes text using the groups in the regex specified

    Expects a `regex`, where different capturing groups correspond to different
    categories of tokens.  Iterates through all matches of the regex on `text`,
    returning the highest matching category with the associated token (group
    text).
    """
    try:
        for match in regex.finditer(text):
            category = match.lastindex
            yield category, match.group(category)
    except (TypeError, IndexError) as e:
        raise LexerError(text, match.start())

def _stub_regex():
    """Return regular expression for stub part of continuation"""
    endline = r"""(?:\r\n?|\n)$"""
    comment = r"""(?:![^\r\n]*)"""
    skip_ws = r"""[\t ]*"""
    stub = r"""(?x)^
                (?: '(?:''|[^'\r\n])*(?=&{skipws}{endline})
                  | "(?:""|[^"\r\n])*(?=&{skipws}{endline})
                  | [^\s&]*(?=&{skipws}{comment}?{endline})
                  )
            """.format(skipws=skip_ws, comment=comment, endline=endline)
    return re.compile(stub)

def _spill_regex():
    """Return regular expression, where group 1 matches continuation spill"""
    return re.compile("""(?xs)^[ \t]*&?(.*)$""")

def _lexer_regex():
    """Return regular expression for parsing free-form Fortran 2008"""
    endline = r"""(?:\r\n?|\n)$"""
    comment = r"""(?:![^\r\n]*)"""
    skip_ws = r"""[\t ]*"""
    postquote = r"""(?!['"\w])"""
    sq_string = r"""'(?:''|[^'\r\n])*'{postquote}""".format(postquote=postquote)
    dq_string = r""""(?:""|[^"\r\n])*"{postquote}""".format(postquote=postquote)
    sq_trunc = r"""'(?:''|[^'\r\n])*&{skipws}{endline}""" \
                    .format(skipws=skip_ws, endline=endline)
    dq_trunc = r""""(?:""|[^"\r\n])*&{skipws}{endline}""" \
                    .format(skipws=skip_ws, endline=endline)
    contd = r"""(?:[^\s&]*&{skipws}{comment}?{endline})""" \
                    .format(skipws=skip_ws, comment=comment, endline=endline)
    postnum = r"""(?!['"&0-9A-Za-z]|\.[0-9])"""
    integer = r"""\d+{postnum}""".format(postnum=postnum)
    decimal = r"""(?:\d+\.\d*|\.\d+)"""
    exponent = r"""(?:[dDeE][-+]?\d+)"""
    real = r"""(?:{decimal}{exponent}?|\d+{exponent}){postnum}""" \
                .format(decimal=decimal, exponent=exponent, postnum=postnum)
    binary = r"""[Bb](?:'[01]+'|"[01]+"){postq}""".format(postq=postquote)
    octal = r"""[Oo](?:'[0-7]+'|"[0-7]+"){postq}""".format(postq=postquote)
    hexadec = r"""[Zz](?:'[0-9A-Fa-f]+'|"[0-9A-Fa-f]+"){postq}""" \
                .format(postq=postquote)
    operator = r"""\(/?|\)|[-+,;:_%]|=[=>]?|\*\*?|\/[\/=)]?|[<>]=?"""
    builtin_dot = r"""
          \.(?:eq|ne|l[te]|g[te]|n?eqv|not|and|or)\.
          """
    dotop = r"""\.[A-Za-z]+\."""
    preproc = r"""(?:\#|include[ \t])[^\r\n]+{endline}""".format(endline=endline)
    word = r"""[A-Za-z][A-Za-z0-9_]*(?![A-Za-z0-9_&])"""
    linestart = r"""(?<=[\r\n])"""
    compound = r"""
          (?: block(?=(?:data)\W)
            | double(?=(?:precision)\W)
            | else(?=(?:if|where)\W)
            | end(?=(?:associate|block|blockdata|critical|do|enum|file|forall
                      |function|if|interface|module|procedure|program|select
                      |submodule|subroutine|type|where)\W)
            | in(?=(?:out)\W)
            | go(?=(?:to)\W)
            | select(?=(?:case|type)\W)
            )
          """
    fortran_token = r"""(?ix)
          ^{skipws}(\d{{1,5}})(?=\s)            #  1 line number
        | ^({skipws}{preproc})                  #  2 preprocessor stmt
        | {skipws}(?:
            ({comment}?{endline})               #  3 endline
          | ({sqstring} | {dqstring})           #  4 strings
          | ({real})                            #  5 real
          | ({int})                             #  6 ints
          | (\.true\. | \.false\.)              #  7 booleans
          | ({binary} | {octal} | {hex})        #  8 radix literals
          | \( {skipws} (//?) {skipws} \)       #  9 bracketed slashes
          | ({operator} | {builtin_dot})        # 10 symbolic/dot operator
          | ({dotop})                           # 11 custom dot operator
          | ({compound} | {word})               # 12 word
          | ({contd} | {sqtrunc} | {dqtrunc})   # (13 continuation)
          | (?=.)
          )
        """.format(
                skipws=skip_ws, endline=endline, linestart=linestart,
                comment=comment, preproc=preproc,
                sqstring=sq_string, dqstring=dq_string,
                real=real, int=integer, binary=binary, octal=octal,
                hex=hexadec, operator=operator, builtin_dot=builtin_dot,
                dotop=dotop, compound=compound, word=word,
                contd=contd, sqtrunc=sq_trunc, dqtrunc=dq_trunc
                )

    return re.compile(fortran_token)

CAT_DOLLAR = 0
CAT_LINENO = 1
CAT_PREPROC = 2
CAT_EOS = 3
CAT_STRING = 4
CAT_FLOAT = 5
CAT_INT = 6
CAT_BOOLEAN = 7
CAT_RADIX = 8
CAT_BRACKETED_SLASH = 9
CAT_OP = 10
CAT_CUSTOM_DOT = 11
CAT_WORD = 12

CAT_NAMES = ('eof', 'lineno', 'preproc', 'eos', 'string', 'float',
             'int', 'boolean', 'radix', 'bracketed_slash', 'op',
             'custom_dot', 'word')

LEXER_REGEX = _lexer_regex()
STUB_REGEX = _stub_regex()
SPILL_REGEX = _spill_regex()

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
    return "".join(actions[cat](token) for (cat, token)
                   in tokenize_regex(STRING_LEXER_REGEX[tok[0]], tok[1:-1]))

if sys.version_info >= (3,):
    CHANGE_D_TO_E = str.maketrans('dD', 'eE')
else:
    CHANGE_D_TO_E = string.maketrans('dD', 'eE')

def parse_float(tok):
    """Translates a Fortran real literal to a Python float"""
    return float(tok.translate(CHANGE_D_TO_E))

def parse_bool(tok):
    """Translates a Fortran boolean literal to a Python boolean"""
    return {'.true.': True, '.false.': False }[tok.lower()]

def parse_radix(tok):
    """Parses a F03-style x'***' literal"""
    base = {'b': 2, 'o': 8, 'z': 16}[tok[0].lower()]
    return int(tok[2:-1], base)

def lex_free_form(buffer):
    """Perform lexical analysis for an opened free-form Fortran file."""
    # For speed of access
    lexer_regex = LEXER_REGEX
    stub_regex = STUB_REGEX
    spill_regex = SPILL_REGEX

    # Iterate through lines of the file
    for line in buffer:
        tokens = list(tokenize_regex(lexer_regex, line))

        # Continuation lines:
        # Fortran allows complicated line continuations with '&', optional
        # comments, and an optional '&' on the following line.  Worse, it also
        # allows to split any token across multiple lines, which requires the
        # lexer to re-read parts of the previous line.
        while tokens[-1][0] == 13:
            stub = stub_regex.match(tokens.pop()[1]).group(0)
            spill = spill_regex.match(next(buffer)).group(1)
            tokens += list(tokenize_regex(lexer_regex, stub + spill))

        # Yield that stuff
        for token_pair in tokens:
            yield token_pair

    yield (CAT_EOS, '\n')
    yield (CAT_DOLLAR, '<$>')
    yield (CAT_DOLLAR, '<$>')

FIXED_CONTD_REGEX = re.compile(r'^[ ]{5}[^ 0]')

def mend_fixed_form_lines(buffer, margin=72):
    """Handle line continuation in fixed form fortran"""
    # For speed of access
    fixed_contd_regex = FIXED_CONTD_REGEX

    # Iterate through lines of the file.  Fortran allows to split tokens
    # across lines, which is why we build up the whole line before giving
    # it to the tokenizer.
    logical_line = None
    for line in buffer:
        line = line[:margin].rstrip()
        if fixed_contd_regex.match(line):
            # This is a continuation.
            logical_line += line[6:]
        else:
            # This is a new line
            if logical_line is not None:
                yield logical_line + "\n"
            logical_line = line

    # Handle the last line
    if logical_line is not None:
        yield logical_line + "\n"

def lex_fixed_form(buffer, margin=72):
    """Perform lexical analysis for an opened fixed-form Fortran file."""
    # For speed of access
    lexer_regex = LEXER_REGEX

    for line in mend_fixed_form_lines(buffer, margin):
        if line[0] in 'cC*':
            # Comment line
            yield (CAT_EOS, line)
        else:
            # Handle normal lines
            for token_pair in tokenize_regex(lexer_regex, line):
                yield token_pair

    yield (CAT_DOLLAR, '<$>')
    yield (CAT_DOLLAR, '<$>')

def pprint(lexer, out):
    """Make nicely formatted JSON output from lexer output"""
    out.write("[\n")
    out.write("['lex_version', '1.0'],\n")
    for cat, token in lexer:
        out.write("['%s',%s]" % (CAT_NAMES[cat], repr(token)))
        if cat == CAT_EOS or cat == CAT_PREPROC:
            out.write(',\n')
        else:
            out.write(', ')
    out.write("['eof', '<$>']\n]\n")

def get_lexer(form='free'):
    if form == 'free':
        return lex_free_form
    else:
        return lex_fixed_form

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Lexer for free-form Fortran')
    parser.add_argument('files', metavar='FILE', type=str, nargs='+',
                        help='files to lex')
    parser.add_argument('--fixed-form', dest='form', action='store_const',
                        const='fixed', default='free', help='Fixed form input')
    parser.add_argument('--free-form', dest='form', action='store_const',
                        const='free', help='Free form input')
    parser.add_argument('--no-output', dest='output', action='store_false',
                        default=True,
                        help='perform lexing but do not print result')
    args = parser.parse_args()
    lexer = get_lexer(args.form)

    for fname in args.files:
        contents = open(fname)
        if args.output:
            pprint(lexer(contents), sys.stdout)
        else:
            for _ in lexer(contents): pass
