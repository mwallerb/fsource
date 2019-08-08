"""
Command line interface (CLI) for fsource.

Copyright 2019 Markus Wallerberger
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
from __future__ import print_function
import argparse
import time
import sys
import json

from . import __version__
from . import splicer
from . import lexer
from . import parser

class Stopwatch:
    def __init__(self, time=time.time):
        self.time = time
        self.previous = time()
        self.initial = self.previous

    def click(self):
        elapsed = self.time() - self.previous
        self.previous += elapsed
        return elapsed

    def total(self):
        return self.time() - self.initial

def get_parser():
    p = argparse.ArgumentParser(prog='fsource',
                                description='Fortran analysis tool')
    p.add_argument('command', choices=('splice','lex','parse'),
                   help='command to execute')
    p.add_argument('files', metavar='FILE', type=str, nargs='+',
                   help='files to parse')
    p.add_argument('--fixed-form', dest='form', action='store_const',
                   const='fixed', default='free', help='Fixed form input')
    p.add_argument('--free-form', dest='form', action='store_const',
                   const='free', help='Free form input')
    p.add_argument('--time', dest='output', action='store_const',
                   const='time', default='json',
                   help='perform lexing but do not print result')
    p.add_argument('--no-output', dest='output', action='store_const',
                   const='no', help='do not output anything')
    return p

def cmd_splice(args):
    lines = splicer.get_splicer(args.form)
    for fname in args.files:
        contents = open(fname)
        if args.output == 'json':
            for cat, line in lines(contents):
                print("%s: %s" % (splicer.LINECAT_NAMES[cat], line), end='')
        else:
            for _ in lines(contents): pass

def pprint_lex(mylexer, out, filename=None):
    """Make nicely formatted JSON output from lexer output"""
    encode_basestring = json.encoder.encode_basestring

    out.write('[\n')
    out.write('["lex_version", "1.0"],\n')
    out.write('["filename", %s],\n' % encode_basestring(filename))
    for cat, token in mylexer:
        out.write('["%s",%s]' % (lexer.CAT_NAMES[cat], encode_basestring(token)))
        if cat == lexer.CAT_EOS or cat == lexer.CAT_PREPROC:
            out.write(',\n')
        elif cat == lexer.CAT_DOLLAR:
            out.write('\n]\n')
        else:
            out.write(', ')

def cmd_lex(args):
    if args.output == 'json':
        for fname in args.files:
            pprint_lex(lexer.lex_buffer(open(fname), args.form), sys.stdout,
                       fname)
    else:
        for fname in args.files:
            for _ in lexer.lex_buffer(open(fname), args.form): pass

def pprint_parser(ast, out, level=0):
    """Make nicely formatted JSON output from parser output"""
    encode_basestring = json.encoder.encode_basestring
    block_elems = {
        'compilation_unit',
        'subroutine_decl',
        'function_decl',
        'interface_body',
        'entity_decl',
        'component_block',
        'declaration_block',
        'contained_block',
        'execution_block',
        'type_bound_procedures'
        }
    repl = {
        True: 'true',
        False: 'false',
        None: 'null'
    }

    if isinstance(ast, tuple):
        out.write("[" + encode_basestring(ast[0]))
        if ast[0] in block_elems:
            for elem in ast[1:]:
                out.write(",\n" + "    " * (level + 1))
                pprint_parser(elem, out, level + 1)
            out.write("\n" + "    " * level + "]")
        else:
            for elem in ast[1:]:
                out.write(", ")
                pprint_parser(elem, out, level)
            out.write("]")
    else:
        try:
            val = repl[ast]
        except KeyError:
            val = encode_basestring(ast)
        out.write(val)

def cmd_parse(args):
    for fname in args.files:
        program = open(fname)
        slexer = lexer.lex_buffer(program, args.form)
        tokens = parser.TokenStream(slexer)
        ast = parser.compilation_unit(tokens, fname)
        if args.output == 'json':
            pprint_parser(ast, sys.stdout)
            print()

def main():
    p = get_parser()
    args = p.parse_args()
    rabbit = Stopwatch()
    if args.command == 'splice':
        cmd_splice(args)
    elif args.command == 'lex':
        cmd_lex(args)
    elif args.command == 'parse':
        cmd_parse(args)
    if args.output == 'time':
        sys.stderr.write("Elapsed: %g sec\n" % rabbit.total())

if __name__ == '__main__':
    main()
