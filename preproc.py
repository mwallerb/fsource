#!/usr/bin/env python
"""
Line handling and preprocessing for free/fixed-form Fortran.

Uses regular expressions to split up a Fortran source file into a sequence
of logical lines together with line types.

Lexical analysis must deal with three ambiguities in the Fortran grammar:

 1. The 'FORMAT' statement is a bit of an oddball, as it allows tokens that
    are illegal everywhere else, e.g., 3I6 or ES13.2.  The lexer works
    around this by returning the format line as single token of category
    `CAT_FORMAT`.

Author: Markus Wallerberger
"""
from __future__ import print_function
import sys
import re

def _freeform_line_regex():
    """Discriminate line type"""
    ws = r"""[ \t]+"""
    anything = r"""[^\r\n]*"""
    comment = r"""(?:![^\r\n]*)"""
    lineno = r"""[0-9]{1,5}(?=[ \t])"""
    include = r"""include{ws}{anything}""".format(ws=ws, anything=anything)
    preproc = r"""\#{anything}""".format(anything=anything)
    atom = r"""(?: [^!&'"\r\n] | '(?:''|[^'\r\n])*' | "(?:""|[^"\r\n])*" )"""
    format = r"""{lineno}{ws}format{ws}\({anything}""".format(
                    ws=ws, anything=anything, lineno=lineno)
    truncstr = r"""(?: '(?:''|[^'\r\n])*
                     | "(?:""|[^"\r\n])*
                     )
            """
    line = r"""(?ix) ^[ \t]*
            (?: ( {preproc} )                   # 1 preprocessor stmt
              | ( {include} )                   # 2 include line
              | ( {format} )                    # 3 format stmt
              | ( {atom}* ) (?:                 # 4 whole line part
                  ( {comment}? )                # 5 full line end
                | ( & [ \t]* {comment}? )       # 6 truncated line end
                | ( {truncstr} ) &[ \t]*        # 7 truncated string line end
                )
              ) $
            """.format(preproc=preproc, include=include, format=format,
                       anything=anything, atom=atom, truncstr=truncstr,
                       comment=comment)

    return re.compile(line)

LINE_PREPROC = 1
LINE_INCLUDE = 2
LINE_FORMAT = 3
LINE_WHOLE_PART = 4
LINE_FULL_END = 5
LINE_TRUNC_END = 6
LINE_TRUNC_STRING_END = 7

LINE_NAMES = (None, 'preproc', 'include', 'format', 'line'
              'end', 'trunc', 'trunc_string')

FF_LINE_REGEX = _freeform_line_regex()

def _freeform_contd_regex():
    """Discriminate line type"""
    ws = r"""[ \t]+"""
    anything = r"""[^\r\n]+"""
    comment = r"""(?:![^\r\n]*)"""
    line = r"""(?x) ^[ \t]*
            (?:
                ( {comment} )              # 1 comment line (ignored)
              | &? [ \t]* ( {anything} )   # 2 spill
              ) $
            """.format(anything=anything, comment=comment)
    return re.compile(line)

CONTD_COMMENT = 1
CONTD_SPILL = 2

FF_CONTD_REGEX = _freeform_contd_regex()


LINECAT_NORMAL = 1
LINECAT_INCLUDE = 2
LINECAT_FORMAT = 3
LINECAT_PREPROC = 4

LINECAT_NAMES = (None, 'line', 'include', 'format', 'preproc')

def free_form_lines(buffer):
    line_regex = FF_LINE_REGEX
    contd_regex = FF_CONTD_REGEX

    # Iterate through lines of the file.  Fortran allows to split tokens
    # across lines, which is why we build up the whole line before giving
    # it to the tokenizer.
    stub = ''
    trunc_str = ''

    for line in buffer:
        # Handle truncated lines
        if stub:
            if trunc_str:
                line = trunc_str + line
                trunc_str = ''
            else:
                match = contd_regex.match(line)
                if match.lastindex == CONTD_COMMENT:
                    continue
                line = match.group(2)

        # Now parse current (potentially preprocessed) line
        match = line_regex.match(line)
        discr = match.lastindex
        if discr == LINE_FULL_END:
            stub += match.group(LINE_WHOLE_PART) + match.group(LINE_FULL_END)
            yield LINECAT_NORMAL, stub
            stub = ''
        elif discr >= LINE_TRUNC_END:
            stub += match.group(LINE_WHOLE_PART)
            if discr == LINE_TRUNC_STRING_END:
                trunc_str = match.group(LINE_TRUNC_STRING_END)
        elif discr == LINE_PREPROC:
            trunc_str += match.group(LINE_PREPROC)
            if trunc_str[-1] == '\\':
                trunc_str = trunc_str[:-1]
                stub = trunc_str
            else:
                yield LINECAT_PREPROC, trunc_str
                stub = ''
                trunc_str = ''
        elif discr == LINE_FORMAT:
            yield LINECAT_FORMAT, line
        else:
            yield LINECAT_INCLUDE, line

    if stub or trunc_str:
        raise RuntimeError("line continuation marker followed by end of file")


def get_lines(form='free'):
    if form == 'free':
        return free_form_lines
    else:
        return None

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Preproc for free-form Fortran')
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
    lines = get_lines(args.form)

    for fname in args.files:
        contents = open(fname)
        if args.output:
            for cat, line in lines(contents):
                print("%s: %s" % (LINECAT_NAMES[cat], line))
        else:
            for _ in lines(contents): pass
