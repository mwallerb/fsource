import re
import itertools

NEWLINE_P = r"""[\t ]*(?:\r\n?|\n)"""
COMMENT_P = r"""(?:![^\r\n]*)"""
SKIPWS_P = r"""[\t ]*"""
CONTINUE_P = r"""&{skipws}{comment}?{newline}{skipws}&?""" \
             .format(skipws=SKIPWS_P, comment=COMMENT_P, newline=NEWLINE_P)

POSTQUOTE_P = r"""(?!['"\w])"""
SQSTRING_P = r"""'(?:''|&{newline}|[^'\r\n])*'{postquote}""" \
             .format(newline=NEWLINE_P, postquote=POSTQUOTE_P)
DQSTRING_P = r""""(?:""|&{newline}|[^"\r\n])*"{postquote}""" \
             .format(newline=NEWLINE_P, postquote=POSTQUOTE_P)

POSTNUM_P = r"""(?!['"0-9A-Za-z]|\.[0-9])"""
INT_P = r"""\d+{postnum}""".format(postnum=POSTNUM_P)
DECIMAL_P = r"""(?:\d+\.\d*|\.\d+)"""
EXPONENT_P = r"""(?:[dDeE][-+]?\d+)"""
REAL_P = r"""(?:{decimal}{exponent}?|\d+{exponent}){postnum}""" \
         .format(decimal=DECIMAL_P, exponent=EXPONENT_P, postnum=POSTNUM_P)

BINARY_P = r"""[Bb](?:'[01]+'|"[01]+"){postq}""".format(postq=POSTQUOTE_P)
OCTAL_P = r"""[Oo](?:'[0-7]+'|"[0-7]+"){postq}""".format(postq=POSTQUOTE_P)
HEX_P = r"""[Zz](?:'[0-9A-Fa-f]+'|"[0-9A-Fa-f]+"){postq}""".format(postq=POSTQUOTE_P)

BRSLASH_P = r"""\({skipws}//?{skipws}\)""".format(skipws=SKIPWS_P)
OPERATOR_P = r"""\(/?|\)|[-+,;:_%]|=>?|\*\*?|\/[\/=)]?|[<>]=?"""
DOTOP_P = r"""\.[A-Za-z]+\."""
PP_P = r"""(?:\#[^\r\n]+)"""

WORD_P = r"""[A-Za-z][A-Za-z0-9_]*"""

FORTRAN_P = r"""(?x) {skipws}(
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
     )""".format(skipws=SKIPWS_P, newline=NEWLINE_P, comment=COMMENT_P,
                 contd=CONTINUE_P, pp=PP_P,
                 sqstring=SQSTRING_P, dqstring=DQSTRING_P,
                 real=REAL_P, int=INT_P, binary=BINARY_P, octal=OCTAL_P,
                 hex=HEX_P, brslash=BRSLASH_P, operator=OPERATOR_P,
                 dotop=DOTOP_P, word=WORD_P)

fortran_lex_re = re.compile(FORTRAN_P)


class Symbol:
    def __init__(self, tok):
        self.token = tok

    def __str__(self):
        return "Symbol %s" % repr(self.token)

class Invalid:
    def __init__(self, tok): pass

    def __str__(self):
        return "INVALID"

class EOS:
    def __init__(self, tok):
        self.token = tok

    def __str__(self):
        return "EOS %s" % repr(self.token)

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

SQBODY_P = r"""(?x)(''|&{newline}(?:\s*&)|[^'\r\n]+)""".format(newline=NEWLINE_P)
DQBODY_P = r"""(?x)(""|&{newline}(?:\s*&)|[^"\r\n]+)""".format(newline=NEWLINE_P)

sqbody_lex_re = re.compile(SQBODY_P)
dqbody_lex_re = re.compile(DQBODY_P)

class String:
    @classmethod
    def parse_value(cls, tok):
        quote = tok[0]
        body = tok[1:-1]
        lexre = sqbody_lex_re if quote == "'" else dqbody_lex_re
        for subtok in lexre.findall(body):
            if subtok[0] == quote:
                yield quote
            elif subtok[0] == '&':
                yield '\n'
            else:
                yield subtok

    def __init__(self, tok):
        self.tok = tok
        self.value = ''.join(self.parse_value(tok))

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
        return "Keyword %s" % repr(self.token)

class Name:
    def __init__(self, tok):
        self.token = tok
        self.value = tok.lower()

    def __str__(self):
        return "Name %s" % repr(self.value)

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
        base = {'b': 2, 'o': 8, 'z': 16}[tok[0].lower()]
        val = int(tok[2:-1], base)
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
    contents = open(fname).read()
    tokens = fortran_lex_re.findall(contents)
    tokens = list(itertools.chain(*map(postproc, tokens)))

    if not all(tokens):
        print("ERROR IN LEXING")
    else:
        print("\n".join(map(str, tokens)))
        print("\n")
