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

Lexical analysis must deal with three ambiguities in the Fortran grammar:

 1. The string '::' can mean an empty slice or a separator token.  The
    lexer always returns single ':', which means one gets '::' as a
    sequence of two ':' and cannot detect whitespace within a seperator.

 2. '(//)' can mean an empty inplace array or an overloaded '//' operator,
    and '(/)', which is ambiguous for a similar reason.  To work around
    this lexer will return a token of category `CAT_BRACKETED_SLASH`, and
    the application must disambiguate.

 3. The 'FORMAT' statement is a bit of an oddball, as it allows tokens that
    are illegal everywhere else, e.g., 3I6 or ES13.2.  The lexer works
    around this by returning the format line as single token of category
    `CAT_FORMAT`.

Author: Markus Wallerberger
"""
from __future__ import print_function
import sys
import re

# Python 2/3 compatibility
if sys.version_info >= (3,):
    _string_like_types = str, bytes,
    _maketrans = str.maketrans
else:
    import string
    _string_like_types = basestring,
    _maketrans = string.maketrans

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
    operator = r"""\(/?|\)|[-+,:_%]|=[=>]?|\*\*?|\/[\/=)]?|[<>]=?"""
    builtin_dot = r"""
          \.(?:eq|ne|l[te]|g[te]|n?eqv|not|and|or)\.
          """
    # HACK: The spaces are to work around wacky F77 code
    dotop = r"""\.\s*[A-Za-z]+\s*\."""
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
    format = r"""format\s*\([^\r\n]+"""
    fortran_token = r"""(?ix)
          ^({skipws}{preproc})                  #  1 preprocessor stmt
        | {skipws}(?:
            (;|{comment}?{endline})             #  2 end of statement
          | ({sqstring} | {dqstring})           #  3 strings
          | ({real})                            #  4 real
          | ({int})                             #  5 ints
          | (\.true\. | \.false\.)              #  6 booleans
          | ({binary} | {octal} | {hex})        #  7 radix literals
          | \( {skipws} (//?) {skipws} \)       #  8 bracketed slashes
          | ({operator} | {builtin_dot})        #  9 symbolic/dot operator
          | ({dotop})                           # 10 custom dot operator
          | ({format})                          # 11 format line
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
                dotop=dotop, compound=compound, word=word, format=format,
                contd=contd, sqtrunc=sq_trunc, dqtrunc=dq_trunc
                )

    return re.compile(fortran_token)

CAT_DOLLAR = 0
#CAT_LINENO = 1
CAT_PREPROC = 1
CAT_EOS = 2
CAT_STRING = 3
CAT_FLOAT = 4
CAT_INT = 5
CAT_BOOLEAN = 6
CAT_RADIX = 7
CAT_BRACKETED_SLASH = 8
CAT_OP = 9
CAT_CUSTOM_DOT = 10
CAT_FORMAT = 11
CAT_WORD = 12
_CAT_CONTINUATION = 13

CAT_NAMES = ('eof', 'preproc', 'eos', 'string', 'float', 'int', 'boolean',
             'radix', 'bracketed_slash', 'op', 'custom_dot', 'format', 'word')

LEXER_REGEX = _lexer_regex()
STUB_REGEX = _stub_regex()
SPILL_REGEX = re.compile("""(?xs)^[ \t]*&?(.*)$""")
COMMENT_LINE_REGEX = re.compile("""(?xs)^[ \t]*!(.*)$""")

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

CHANGE_D_TO_E = _maketrans('dD', 'eE')

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
    # check for buffer
    if isinstance(buffer, _string_like_types):
        raise ValueError("Expect open file or other sequence of lines")
    buffer = iter(buffer)

    # For speed of access
    lexer_regex = LEXER_REGEX
    stub_regex = STUB_REGEX
    spill_regex = SPILL_REGEX
    comment_line_regex = COMMENT_LINE_REGEX
    cat_continuation = _CAT_CONTINUATION

    # Iterate through lines of the file
    for line in buffer:
        tokens = list(tokenize_regex(lexer_regex, line))

        # Continuation lines:
        # Fortran allows complicated line continuations with '&', optional
        # comments, and an optional '&' on the following line.  Worse, it also
        # allows to split any token across multiple lines, which requires the
        # lexer to re-read parts of the previous line.
        while tokens[-1][0] == cat_continuation:
            spill_line = next(buffer)
            if comment_line_regex.match(spill_line):
                continue
            stub = stub_regex.match(tokens.pop()[1]).group(0)
            spill = spill_regex.match(spill_line).group(1)
            tokens += list(tokenize_regex(lexer_regex, stub + spill))

        # Yield that stuff
        for token_pair in tokens:
            yield token_pair

    # Make sure last line is terminated, then yield terminal token
    yield (CAT_EOS, '\n')
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
    # check for buffer
    if isinstance(buffer, _string_like_types):
        raise ValueError("Expect open file or other sequence of lines")
    buffer = iter(buffer)

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

    # Yield terminal token
    yield (CAT_DOLLAR, '<$>')

def lex_snippet(fstring):
    """Perform lexical analysis of parts of a line"""
    return tuple(tokenize_regex(LEXER_REGEX, fstring)) + ((CAT_DOLLAR, ''),)

def pprint(lexer, out, filename=None):
    """Make nicely formatted JSON output from lexer output"""
    from json.encoder import encode_basestring
    out.write('[\n')
    out.write('["lex_version", "1.0"],\n')
    out.write('["filename", %s],\n' % encode_basestring(filename))
    for cat, token in lexer:
        out.write('["%s",%s]' % (CAT_NAMES[cat], encode_basestring(token)))
        if cat == CAT_EOS or cat == CAT_PREPROC:
            out.write(',\n')
        elif cat == CAT_DOLLAR:
            out.write('\n]\n')
        else:
            out.write(', ')

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
            pprint(lexer(contents), sys.stdout, fname)
        else:
            for _ in lexer(contents): pass
