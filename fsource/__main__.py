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
from . import common


class Stopwatch:
    """Keeps track of elapsed time since construction"""
    def __init__(self, time=time.time):
        """Initialize object and start stopwatch"""
        self.time = time
        self.previous = time()
        self.initial = self.previous

    def click(self):
        """Start new lap and return time of current lap"""
        elapsed = self.time() - self.previous
        self.previous += elapsed
        return elapsed

    def total(self):
        """Return total elapsed time"""
        return self.time() - self.initial


def get_parser():
    """Return argument parser"""
    p = argparse.ArgumentParser(prog='fsource',
                                description='Fortran analysis tool')
    p.add_argument('command', choices=('splice', 'lex', 'parse'),
                   help='command to execute')
    p.add_argument('files', metavar='FILE', type=str, nargs='+',
                   help='files to parse')
    p.add_argument('--fixed-form', dest='form', action='store_const',
                   const='fixed', default=None, help='Fixed form input')
    p.add_argument('--free-form', dest='form', action='store_const',
                   const='free', help='Free form input')
    p.add_argument('--time', dest='output', action='store_const',
                   const='time', default='json',
                   help='perform lexing but do not print result')
    p.add_argument('--no-output', dest='output', action='store_const',
                   const='no', help='do not output anything')
    return p


def get_form(fname, force_form):
    """Discern source form (fixed/free) from parameter and file name"""
    if force_form is None:
        is_free, _ = common.guess_form(fname)
        return is_free
    else:
        return force_form


def cmd_splice(args):
    """splice subcommand handler"""
    try:
        for fname in args.files:
            form = get_form(fname, args.form)
            lines = splicer.get_splicer(form)
            contents = open(fname)
            if args.output == 'json':
                for lineno, cat, line in lines(contents):
                    print("%d: %s: %s"
                          % (lineno, splicer.LINECAT_NAMES[cat], line), end='')
            else:
                for _ in lines(contents): pass
    except common.ParsingError as e:
        sys.stdout.flush()
        sys.stderr.write("\n" + e.errmsg())


def pprint_lex(mylexer, out, filename=None):
    """Make nicely formatted JSON output from lexer output"""
    encode_string = json.encoder.encode_basestring

    out.write('[\n')
    out.write('["lex_version", "1.0"],\n')
    out.write('["filename", %s],\n' % encode_string(filename))
    for _, _, cat, token in mylexer:
        out.write('["%s",%s]' % (lexer.CAT_NAMES[cat], encode_string(token)))
        if cat == lexer.CAT_EOS or cat == lexer.CAT_PREPROC:
            out.write(',\n')
        elif cat == lexer.CAT_DOLLAR:
            out.write('\n]\n')
        else:
            out.write(', ')


def cmd_lex(args):
    """lex subcommand handler"""
    try:
        for fname in args.files:
            form = get_form(fname, args.form)
            mylexer = lexer.lex_buffer(open(fname), form)
            if args.output == 'json':
                pprint_lex(mylexer, sys.stdout, fname)
            else:
                for _ in mylexer: pass
    except common.ParsingError as e:
        sys.stdout.flush()
        sys.stderr.write("\n\n" + e.errmsg())


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
    """parse subcommand handler"""
    try:
        for fname in args.files:
            form = get_form(fname, args.form)
            slexer = lexer.lex_buffer(open(fname), form)
            tokens = parser.TokenStream(slexer, fname=fname)
            ast = parser.compilation_unit(tokens, fname)
            if args.output == 'json':
                pprint_parser(ast, sys.stdout)
                print()
    except common.ParsingError as e:
        sys.stdout.flush()
        sys.stderr.write("\n\n" + e.errmsg())


def main():
    """main entry point"""
    p = get_parser()
    args = p.parse_args()
    rabbit = Stopwatch()
    try:
        if args.command == 'splice':
            cmd_splice(args)
        elif args.command == 'lex':
            cmd_lex(args)
        elif args.command == 'parse':
            cmd_parse(args)
        if args.output == 'time':
            sys.stderr.write("Elapsed: %g sec\n" % rabbit.total())
    except IOError as e:
        p.error(e)


if __name__ == '__main__':
    main()
