import re
import itertools

def _get_lexer_regex():
    """Return regular expression for parsing free-form Fortran 2008"""
    newline = r"""(?:\r\n?|\n)"""
    comment = r"""(?:![^\r\n]*)"""
    skip_ws = r"""[\t ]*"""
    continuation = r"""&{skipws}{comment}?{newline}{skipws}&?""" \
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
    operator = r"""\(/?|\)|[-+,;:_%]|=>?|\*\*?|\/[\/=)]?|[<>]=?"""
    builtin_dot = r"""
          \.(?:eq|ne|l[te]|g[te]|n?eqv|not|and|or|true|false)\.
          """
    dotop = r"""\.[A-Za-z]+\."""
    preproc = r"""(?:\#[^\r\n]+){newline}""".format(newline=newline)
    word = r"""[A-Za-z][A-Za-z0-9_]*"""
    linestart = r"""(?<=[\r\n])"""
    compound = r"""
          (?: go(?=to)
            | else(?=if|where)
            | end(?=if|where|function|subroutine|program|do|while|block)
            )
          """
    fortran_token = r"""(?ix)
          {linestart}{skipws}(\d{{1,5}})(?=\s)  #  1 line number
        | {linestart}({skipws}{preproc})        #  2 preprocessor stmt
        | {skipws}(?:
            ({newline}|$)                       #  3 newline or end
          | ({contd})                           #  4 contd
          | ({comment})                         #  5 comment
          | ({sqstring} | {dqstring})           #  6 strings
          | ({real})                            #  7 real
          | ({int})                             #  8 ints
          | ({binary} | {octal} | {hex})        #  9 radix literals
          | \( {skipws} (//?) {skipws} \)       # 10 bracketed slashes
          | ({operator} | {builtin_dot})        # 11 symbolic/dot operator
          | ({dotop})                           # 12 custom dot operator
          | ({compound} | {word})               # 13 word
          | (?=.)
          )
        """.format(
                skipws=skip_ws, newline=newline, linestart=linestart,
                comment=comment, contd=continuation, preproc=preproc,
                sqstring=sq_string, dqstring=dq_string,
                real=real, int=integer, binary=binary, octal=octal,
                hex=hexadec, operator=operator, builtin_dot=builtin_dot,
                dotop=dotop, compound=compound, word=word
                )

    return re.compile(fortran_token)

def _get_stringlexer_regex(quote):
    skipws = r"""[\t ]*"""
    newline = r"""(?:\r\n?|\n)"""
    pattern = r"""(?x)
          ({quote}{quote})
        | (&{skipws}{newline}(?:{skipws}&)?)
        | (&)
        | ([^{quote}&\r\n]+)
        """.format(newline=newline, skipws=skipws, quote=quote)
    return re.compile(pattern)

def _get_stringlexer_actions():
    return (None,
        lambda tok: tok[0],
        lambda tok: "\n",
        lambda tok: "&",
        lambda tok: tok,
        )

LEXER_REGEX = _get_lexer_regex()
STRING_LEXER_REGEX = {
    "'": _get_stringlexer_regex("'"),
    '"': _get_stringlexer_regex('"'),
    }
STRING_LEXER_ACTIONS = _get_stringlexer_actions()

def _sublex_string_body(tok):
    quote = tok[0]
    body = tok[1:-1]
    action = STRING_LEXER_ACTIONS
    for match in STRING_LEXER_REGEX[quote].finditer(body):
        type_ = match.lastindex
        yield action[type_](match.group(type_))

def parse_string(tok):
    """Translates a Fortran string literal to a Python string"""
    return "".join(_sublex_string_body(tok))

def parse_float(tok):
    """Translates a Fortran real literal to a Python float"""
    change_d_to_e = {100: 101, 68: 69}
    return float(tok.translate(change_d_to_e))

def parse_radix(tok):
    """Parses a F03-style x'***' literal"""
    base = {'b': 2, 'o': 8, 'z': 16}[tok[0].lower()]
    return int(tok[2:-1], base)

class LexerError(RuntimeError): pass

def run_lexer(text):
    postproc = (None,
         lambda tok: '\\line(%s)' % tok,
         lambda tok: '\\preproc(%s)' % repr(tok),
         lambda tok: '\\eos\n',
         lambda tok: '\\contd',
         lambda tok: '\\comment(%s)' % repr(tok),
         lambda tok: '\\string(%s)'% repr(parse_string(tok)),
         lambda tok: '\\real(%s)' % repr(parse_float(tok)),
         lambda tok: '\\int(%s)' % repr(int(tok)),
         lambda tok: '\\radix(%s)' % repr(parse_radix(tok)),
         lambda tok: '\\brslashes(%s)' % tok,
         lambda tok: tok,   # symbolic op/dotop
         lambda tok: '\\dotop(%s)' % tok[1:-1],
         lambda tok: '\\word(%s)' % tok,   # word
        )

    try:
        for match in LEXER_REGEX.finditer(text):
            type_ = match.lastindex
            yield postproc[type_](match.group(type_))
    except TypeError:
        raise LexerError(
            "Lexer error at character %d:\n%s" %
            (match.start(), text[match.start():match.start()+100]))

if __name__ == '__main__':
    import sys
    fname = sys.argv[1]
    contents = "\n" + open(fname).read()
    tokens = list(run_lexer(contents))
    if len(sys.argv) > 2:
        print(" ".join(map(str, tokens)))
