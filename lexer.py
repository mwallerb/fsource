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
    bracketed_slashes = r"""\({skipws}//?{skipws}\)""".format(skipws=skip_ws)
    operator = r"""\(/?|\)|[-+,;:_%]|=>?|\*\*?|\/[\/=)]?|[<>]=?"""
    dotop = r"""\.[A-Za-z]+\."""
    preproc = r"""(?:\#[^\r\n]+)"""
    word = r"""[A-Za-z][A-Za-z0-9_]*"""
    fortran_token = r"""(?x) {skipws}(
          {newline}(?:{skipws}{pp})?
        | {contd}
        | {comment}
        | {sqstring}
        | {dqstring}
        | {real}
        | {int}
        | {binary}
        | {octal}
        | {hex}
        | {brslash}
        | {operator}
        | {dotop}
        | {word}
        | (?=.)
        )""".format(skipws=skip_ws, newline=newline, comment=comment,
                    contd=continuation, pp=preproc,
                    sqstring=sq_string, dqstring=dq_string,
                    real=real, int=integer, binary=binary, octal=octal,
                    hex=hexadec, brslash=bracketed_slashes, operator=operator,
                    dotop=dotop, word=word)

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

def parse_base_literal(tok):
    """Parses a F03-style x'***' literal"""
    base = {'b': 2, 'o': 8, 'z': 16}[tok[0].lower()]
    return int(tok[2:-1], base)


class Symbol:
    def __init__(self, tok):
        self.token = tok

    def __str__(self):
        return "Symbol %s" % self.token

class Invalid:
    def __init__(self, tok): pass

    def __str__(self):
        return "INVALID"

class EOS:
    def __init__(self, tok):
        self.token = tok

    def __str__(self):
        return "EOS"

class PreprocStmt:
    def __init__(self, tok):
        self.token = tok

    def __str__(self):
        return "PreprocStmt %s" % repr(self.token)

class Comment:
    def __init__(self, tok):
        self.token = tok
        self.value = tok[1:]

    def __str__(self):
        return "Comment %s" % repr(self.value)

class String:
    def __init__(self, tok):
        self.tok = tok
        self.value = parse_string(tok)

    def __str__(self):
        return "String %s" % repr(self.value)

class Integer(Symbol):
    def __init__(self, tok, value=None):
        if value is None: value = int(tok)
        self.token = tok
        self.value = value

    def __str__(self):
        return "Integer %s" % repr(self.value)

class Float(Symbol):
    def __init__(self, tok):
        self.token = tok
        self.value = float(tok.translate({100: 101, 68: 69}))

    def __str__(self):
        return "Float %s" % repr(self.value)

class DotOp:
    def __init__(self, tok):
        self.token = tok
        self.value = tok[1:-1]

    def __str__(self):
        return "DotOp %s" % self.value

class Keyword:
    def __init__(self, tok):
        self.token = tok

    def __str__(self):
        return "Keyword %s" % self.token

class Name:
    def __init__(self, tok):
        self.token = tok
        self.value = tok.lower()

    def __str__(self):
        return "Name %s" % self.value

MARKERS = {
    '\r': EOS,
    '\n': EOS,
    '':   Invalid,
    }

SYMBOLS = [
    '(', ')', '(/',  '/)', '+', '-', '*', '**', '/', '//', '%', '_', ',', ';',
    '<', '>', '<=', '>=', '==', '/=', '=', '=>', ':'
    ]

for symbol in SYMBOLS:
    MARKERS[symbol] = Symbol

def handle_dot_prefix(tok):
    if tok[1].isdigit():
        return Float(tok)
    else:
        return DotOp(tok)

def handle_eos(tok):
    try:
        eos_part, pp_part = tok.split("#")
    except ValueError:
        return EOS(tok)
    else:
        return [EOS(eos_part), PreprocStmt("#" + pp_part)]

def handle_bracketed_slash(tok):
    if "//" in tok:
        return [Symbol("("), Symbol("//"), Symbol(")")]
    else:
        return [Symbol("("), Symbol("/"), Symbol(")")]

PREFIX = {
    '"': String,
    "'": String,
    '!': Comment,
    '.': handle_dot_prefix,
    "\r": handle_eos,
    "\n": handle_eos,
    "(": handle_bracketed_slash,
    "&": lambda tok: [],
}

def handle_number(tok):
    try:
        num = int(tok)
    except ValueError:
        return Float(tok)
    else:
        return Integer(tok, num)

for digit in range(10):
    PREFIX[str(digit)] = handle_number

ENDABLE = ['if', 'where', 'program', 'type', 'subroutine' 'function', 'file',
           'do', 'while', 'block']

KEYWORDS = ENDABLE + [
    'else',  'program',  'implicit',  'none',  'end', 'integer',  'double',
    'precision',  'complex',  'character',  'logical',  'type',
    'operator',  'assignment',  'common',  'data',  'equivalence',  'namelist',
    'subroutine',   'function',   'recursive', 'parameter', 'entry', 'result',
    'optional',  'intent', 'dimension',  'external',  'internal',  'intrinsic',
    'public',  'private',  'sequence', 'interface', 'module', 'use', 'only',
    'contains', 'allocatable',  'pointer',  'save',  'allocate',  'deallocate',
    'cycle', 'exit', 'nullify', 'call',  'continue',  'pause',  'return',
    'stop', 'format',  'backspace', 'close',  'inquire',  'open',  'print',
    'read',  'write',  'rewind', 'where', 'assign',  'to', 'select', 'case',  'default', 'go'
    ]

def handle_word(tok):
    ltok = tok.lower()
    if ltok in KEYWORDS:
        return Keyword(ltok)
    elif ltok.startswith('end') and ltok[3:] in ENDABLE:
        return [Keyword('end'), Keyword(ltok[3:])]
    elif ltok.startswith('else') and ltok[4:] in ('if', 'where'):
        return [Keyword('else'), Keyword(ltok[4:])]
    elif ltok == 'goto':
        return [Keyword('go'), Keyword('to')]
    else:
        return Name(tok)

for letter in 'acdefghijklmnpqrstuvwxy':
    PREFIX[letter] = handle_word
    PREFIX[letter.upper()] = handle_word

def handle_word_or_pattern(tok):
    if tok[-1] in "'\"":
        val = parse_base_literal(tok)
        return Integer(tok, val)
    else:
        return handle_word(tok)

for letter in 'boz':
    PREFIX[letter] = handle_word_or_pattern
    PREFIX[letter.upper()] = handle_word_or_pattern

def postproc(token):
    global MARKERS
    try:
        fn = MARKERS[token]
    except KeyError:
        fn = PREFIX[token[0]]

    val = fn(token)
    if not isinstance(val, list):
        val = [val]
    return val

if __name__ == '__main__':
    import sys
    fname = sys.argv[1]
    contents = "\n" + open(fname).read() + "\n"
    tokens = LEXER_REGEX.findall(contents)
    tokens = list(itertools.chain(*map(postproc, tokens)))

    if not all(tokens):
        print("ERROR IN LEXING")
    else:
        #print("\n".join(map(str, tokens)))
        print("SUCCESS")
