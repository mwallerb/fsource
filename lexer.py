#!/usr/bin/env python
from __future__ import print_function
import sys
import string
import re
import itertools

class LexerError(RuntimeError):
    pass

def tokenize_regex(regex, text, actions=None):
    """Tokenizes text using the groups in the regex specified

    Expects a `regex`, where different capturing groups correspond to different
    categories of tokens.  Iterates through all matches of the regex on `text`,
    returning the highest matching category with the associated token (group
    text).
    """
    try:
        if actions is None:
            for match in regex.finditer(text):
                category = match.lastindex
                yield category, match.group(category)
        else:
            for match in regex.finditer(text):
                category = match.lastindex
                yield actions[category](match.group(category))
    except (TypeError, IndexError) as e:
        raise LexerError(
            "Lexer error at character %d:\n%s" %
            (match.start(), text[match.start():match.start()+100]))

def _lexer_regex():
    """Return regular expression for parsing free-form Fortran 2008"""
    newline = r"""(?:\r\n?|\n)"""
    comment = r"""(?:![^\r\n]*)"""
    skip_ws = r"""[\t ]*"""
    contn = r"""(?:&{skipws}{comment}?{newline}{skipws}(?:&{skipws})?)""" \
                .format(skipws=skip_ws, comment=comment, newline=newline)
    postquote = r"""(?!['"\w])"""
    sq_string = r"""'(?:''|&{skipws}{newline}|[^'\r\n])*'{postquote}""" \
                .format(skipws=skip_ws, newline=newline, postquote=postquote)
    dq_string = r""""(?:""|&{skipws}{newline}|[^"\r\n])*"{postquote}""" \
                .format(skipws=skip_ws, newline=newline, postquote=postquote)
    postnum = r"""(?!['"0-9A-Za-z]|\.[0-9])"""
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
    preproc = r"""(?:\#[^\r\n]+){newline}""".format(newline=newline)
    word = r"""[A-Za-z][A-Za-z0-9_]*"""
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
          {linestart}{skipws}(\d{{1,5}})(?=\s)  #  1 line number
        | {linestart}({skipws}{preproc})        #  2 preprocessor stmt
        | {skipws}{contd}?(?:
            ({comment}?(?:{newline}|$))         #  3 newline or end
          | ({sqstring} | {dqstring})           #  4 strings
          | ({real})                            #  5 real
          | ({int})                             #  6 ints
          | (\.true\. | \.false\.)              #  7 booleans
          | ({binary} | {octal} | {hex})        #  8 radix literals
          | \( {skipws} (//?) {skipws} \)       #  9 bracketed slashes
          | ({operator} | {builtin_dot})        # 10 symbolic/dot operator
          | ({dotop})                           # 11 custom dot operator
          | ({compound} | {word})               # 12 word
          | (?=.)
          )
        """.format(
                skipws=skip_ws, newline=newline, linestart=linestart,
                comment=comment, contd=contn, preproc=preproc,
                sqstring=sq_string, dqstring=dq_string,
                real=real, int=integer, binary=binary, octal=octal,
                hex=hexadec, operator=operator, builtin_dot=builtin_dot,
                dotop=dotop, compound=compound, word=word
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

LEXER_REGEX = _lexer_regex()

def _string_lexer_regex(quote):
    skipws = r"""[\t ]*"""
    newline = r"""(?:\r\n?|\n)"""
    pattern = r"""(?x)
          ({quote}{quote})
        | (&{skipws}{newline}(?:{skipws}&)?)
        | (&)
        | ([^{quote}&\r\n]+)
        """.format(newline=newline, skipws=skipws, quote=quote)
    return re.compile(pattern)

def _string_lexer_actions():
    return (None,
        lambda tok: tok[0],
        lambda tok: "\n",
        lambda tok: "&",
        lambda tok: tok,
        )

STRING_LEXER_REGEX = {
    "'": _string_lexer_regex("'"),
    '"': _string_lexer_regex('"'),
    }
STRING_LEXER_ACTIONS = _string_lexer_actions()

def parse_string(tok):
    """Translates a Fortran string literal to a Python string"""
    return "".join(tokenize_regex(STRING_LEXER_REGEX[tok[0]],
                                  tok[1:-1], STRING_LEXER_ACTIONS))

if sys.version_info >= (3,):
    CHANGE_D_TO_E = str.maketrans('dD', 'eE')
else:
    CHANGE_D_TO_E = string.maketrans('dD', 'eE')

def parse_float(tok):
    """Translates a Fortran real literal to a Python float"""
    return float(tok.translate(CHANGE_D_TO_E))

def parse_bool(tok):
    return {'.true.': True, '.false.': False }[tok.lower()]

def parse_radix(tok):
    """Parses a F03-style x'***' literal"""
    base = {'b': 2, 'o': 8, 'z': 16}[tok[0].lower()]
    return int(tok[2:-1], base)


class LexerError(RuntimeError):
    pass

def lexer_print_actions():
    return (None,
            lambda tok: 'lineno:%s' % tok,
            lambda tok: 'preproc:%s' % tok,
            lambda tok: 'eos:%s\n' % repr(tok.rstrip()),
            lambda tok: 'string:%s' % repr(parse_string(tok)),
            lambda tok: 'float:%s' % repr(parse_float(tok)),
            lambda tok: 'int:%d' % int(tok),
            lambda tok: 'bool:%s' % repr(parse_bool(tok)),
            lambda tok: 'radix:%d' % parse_radix(tok),
            lambda tok: 'bracketed_slashes:%s' % tok[1:-1],
            lambda tok: 'op:%s' % tok,
            lambda tok: 'custom_dot:%s' % tok[1:-1],
            lambda tok: 'word:%s' % tok
            )

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Lexer for free-form Fortran')
    parser.add_argument('files', metavar='FILE', type=str, nargs='+',
                        help='files to lex')
    parser.add_argument('--dump', dest='dump', action='store_true', default=False,
                        help='dump the tokens to stdout')
    args = parser.parse_args()

    for fname in args.files:
        contents = "\n" + open(fname).read()
        if args.dump:
            actions = lexer_print_actions()
            print (" ".join(tokenize_regex(LEXER_REGEX, contents, actions)))
        else:
            for _ in tokenize_regex(LEXER_REGEX, contents): pass
