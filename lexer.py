import re
import itertools

def _get_lexer_regex():
    """Return regular expression for parsing free-form Fortran 2008"""
    newline = r"""[\t ]*(?:\r\n?|\n)"""
    comment = r"""(?:![^\r\n]*)"""
    skip_ws = r"""[\t ]*"""
    continuation = r"""&{skipws}{comment}?{newline}{skipws}&?""" \
            .format(skipws=skip_ws, comment=comment, newline=newline)
    postquote = r"""(?!['"\w])"""
    sq_string = r"""'(?:''|&{newline}|[^'\r\n])*'{postquote}""" \
                .format(newline=newline, postquote=postquote)
    dq_string = r""""(?:""|&{newline}|[^"\r\n])*"{postquote}""" \
                .format(newline=newline, postquote=postquote)
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
    preproc = r"""(?:\#[^\r\n]+)"""
    word = r"""[A-Za-z][A-Za-z0-9_]*"""
    compound = r"""
          (?: go(?=to)
            | else(?=if|where)
            | end(?=if|where|function|subroutine|program|do|while|block)
            )
          """
    fortran_token = r"""(?ix) {skipws}(?:
          ({newline})(?:{skipws}({pp}))?    #  1 newline, 2 preproc
        | ({contd})                         #  3 contd
        | ({comment})                       #  4 comment
        | ({sqstring} | {dqstring})         #  5 strings
        | ({real})                          #  6 real
        | ({int})                           #  7 ints
        | ({binary} | {octal} | {hex})      #  8 radix literals
        | \( {skipws} (//?) {skipws} \)     #  9 bracketed slashes
        | ({operator} | {builtin_dot})      # 10 symbolic/dot operator
        | ({dotop})                         # 11 custom dot operator
        | ({compound} | {word})             # 12 word
        | (?=.)
        )""".format(skipws=skip_ws, newline=newline, comment=comment,
                    contd=continuation, pp=preproc,
                    sqstring=sq_string, dqstring=dq_string,
                    real=real, int=integer, binary=binary, octal=octal,
                    hex=hexadec, operator=operator, builtin_dot=builtin_dot,
                    dotop=dotop, compound=compound, word=word)

    return re.compile(fortran_token)

def _get_string_body_regex():
    newline = r"""[\t ]*(?:\r\n?|\n)"""
    sq_body = r"""(?x)(''|&{newline}(?:\s*&)|[^'\r\n]+)""" \
                .format(newline=newline)
    dq_body = r"""(?x)(""|&{newline}(?:\s*&)|[^"\r\n]+)""" \
                .format(newline=newline)

    return {"'": re.compile(sq_body),
            '"': re.compile(dq_body)
            }

LEXER_REGEX = _get_lexer_regex()
STRING_BODY_REGEX = _get_string_body_regex()

def _sublex_string_body(tok):
    quote = tok[0]
    body = tok[1:-1]
    for subtok in STRING_BODY_REGEX[quote].findall(body):
        if subtok[0] == quote:
            yield quote
        elif subtok[0] == '&':
            yield '\n'
        else:
            yield subtok

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

def run_lexer(text):
    postproc = [None,
         lambda tok: 'EOS\n',
         lambda tok: 'PREPROC(%s)' % repr(tok),
         lambda tok: 'CONTD',
         lambda tok: 'COMMENT(%s)' % repr(tok),
         lambda tok: 'STRING(%s)'% repr(parse_string(tok)),
         lambda tok: 'REAL(%s)' % repr(parse_float(tok)),
         lambda tok: 'INT(%s)' % repr(int(tok)),
         lambda tok: 'RADIX(%s)' % repr(parse_radix(tok)),
         lambda tok: 'BSL(%s)' % tok,
         lambda tok: tok,   # symbolic op/dotop
         lambda tok: 'DOTOP(%s)' % tok[1:-1],
         lambda tok: tok,   # word
        ]
    for match in LEXER_REGEX.finditer(text):
        type_ = match.lastindex
        yield postproc[type_](match.group(type_))

if __name__ == '__main__':
    import sys
    fname = sys.argv[1]
    contents = "\n" + open(fname).read() + "\n"
    tokens = list(run_lexer(contents))
    if len(sys.argv) > 2:
        print(" ".join(map(str, tokens)))
