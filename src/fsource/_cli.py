"""
Command line interface (CLI) for fsource.

Copyright 2019 Markus Wallerberger
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
from __future__ import print_function
import argparse
import errno
import time
import sys
import json
import textwrap

from . import __version__, __version_tuple__
from . import splicer
from . import lexer
from . import parser
from . import analyzer
from . import common


class Stopwatch:
    """Keeps track of elapsed time since construction"""
    def __init__(self, mytime=time.time):
        """Initialize object and start stopwatch"""
        self.time = mytime
        self.previous = mytime()
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
    command_summary = textwrap.dedent("""\
        Fortran static analyis tool

        available commands:
          parse         construct abstract syntax tree for declarations
          lex           split source into tokens
          splice        split source into sequence of logical lines
          wrap          generate C wrappers for Fortran declarations
        """)

    p = argparse.ArgumentParser(
            prog='fsource',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=command_summary,
            usage="fsource COMMAND FILE [FILE ...]"
            )
    p.add_argument('command', metavar='COMMAND',
                   choices=('splice', 'lex', 'parse', 'wrap'),
                   help=argparse.SUPPRESS)
    p.add_argument('files', metavar='FILE', type=str, nargs='+',
                   help="Fortran file(s) to process")
    p.add_argument('--version', action='version',
                   version='%(prog)s ' + __version__)
    p.add_argument('--fixed-form', dest='form', action='store_const',
                   const='fixed', default=None, help='force fixed form input')
    p.add_argument('--free-form', dest='form', action='store_const',
                   const='free', help='force free form input')
    p.add_argument('--time', dest='output', action='store_const',
                   const='time', default='json',
                   help='process files but print timings only')
    p.add_argument('--dry-run', dest='output', action='store_const',
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
                          % (lineno+1, splicer.LINECAT_NAMES[cat], line), end='')
            else:
                for _ in lines(contents): pass
    except common.ParsingError as e:
        sys.stdout.flush()
        sys.stderr.write("\n" + e.errmsg())


def pprint_lex(mylexer, out, filename=None):
    """Make nicely formatted JSON output from lexer output"""
    encode_string = json.encoder.encode_basestring

    out.write('[\n')
    out.write('["lex_version", %s],\n' %
              ", ".join('"%s"' % v for v in __version_tuple__))
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
        'entity_list',
        'component_block',
        'declaration_block',
        'contained_block',
        'execution_block',
        'type_bound_procedures',
        'common_stmt',
        'common_block',
        'block_data_decl',
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
            ast = parser.compilation_unit(tokens)
            if args.output == 'json':
                pprint_parser(ast, sys.stdout)
                print()
    except common.ParsingError as e:
        sys.stdout.flush()
        sys.stderr.write("\n\n" + e.errmsg())


def cmd_wrap(args):
    """wrapping subcommand handler"""
    try:
        for fname in args.files:
            form = get_form(fname, args.form)
            slexer = lexer.lex_buffer(open(fname), form)
            tokens = parser.TokenStream(slexer, fname=fname)
            ast = parser.compilation_unit(tokens)
            asr = analyzer.TRANSFORMER(ast)
            asr.imbue()

            # TODO: order contexts by module dependencies
            asr.resolve(analyzer.Context())
            print (asr.cdecl().get())
    except common.ParsingError as e:
        sys.stdout.flush()
        sys.stderr.write("\n\n" + e.errmsg())
        sys.exit(1)


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
        elif args.command == 'wrap':
            cmd_wrap(args)
        if args.output == 'time':
            sys.stderr.write("Elapsed: %g sec\n" % rabbit.total())
    except IOError as e:
        if e.errno != errno.EPIPE:
            raise

if __name__ == '__main__':
    main()
