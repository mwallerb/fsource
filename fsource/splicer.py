"""
Line handling and preprocessing for free/fixed-form Fortran.

Uses regular expressions to split up a Fortran source file into a sequence
of logical lines together with line types.

Lexical analysis must deal with three ambiguities in the Fortran grammar:

 1. The 'FORMAT' statement is a bit of an oddball, as it allows tokens that
    are illegal everywhere else, e.g., 3I6 or ES13.2.  The lexer works
    around this by returning the format line as single token of category
    `CAT_FORMAT`.

Copyright 2019 Markus Wallerberger
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
from __future__ import print_function
import sys
import re

from . import common


class SpliceError(common.ParsingError):
    """Error in line splicing"""
    @property
    def error_type(self): return "splice error"

    def __init__(self, fname, lineno, line, msg):
        common.ParsingError.__init__(self, fname, lineno, None, None, line,
                                     msg)


def get_freeform_line_regex():
    """Discriminate line type"""
    ws = r"""[ \t]+"""
    endline = r"""(?:\n|\r\n?)"""
    anything = r"""[^\r\n]*"""
    comment = r"""(?:![^\r\n]*)"""
    lineno = r"""[0-9]{1,5}(?=[ \t])"""
    include = r"""include{ws}{anything}""".format(ws=ws, anything=anything)
    preproc = r"""\#{anything}""".format(anything=anything)
    atom = r"""(?: [^!&'"\r\n] | '(?:''|[^'\r\n])*' | "(?:""|[^"\r\n])*" )"""
    format = r"""{lineno}{ws}format[ \t]*\({anything}""".format(
                    ws=ws, anything=anything, lineno=lineno)
    truncstr = r"""(?: '(?:''|[^'\r\n])*
                     | "(?:""|[^"\r\n])*
                     )
            """
    line = r"""(?ix) ^[ \t]*
        (?: ( {preproc} ) {endline}                  # 1 preprocessor stmt
            | ( {include} {endline} )                # 2 include line
            | ( {format} {endline} )                 # 3 format stmt
            | ( {atom}* ) (?:                        # 4 whole line part
                  ( {comment}? {endline} )             # 5 full line end
                | ( & [ \t]* {comment}? {endline} )    # 6 truncated line end
                | ( {truncstr} ) &[ \t]* {endline}     # 7 truncated string end
              )
            ) $
        """.format(preproc=preproc, include=include, format=format,
                   anything=anything, atom=atom, truncstr=truncstr,
                   comment=comment, endline=endline)

    return re.compile(line)


FREE_PREPROC = 1
FREE_INCLUDE = 2
FREE_FORMAT = 3
FREE_WHOLE_PART = 4
FREE_FULL_END = 5
FREE_TRUNC_END = 6
FREE_TRUNC_STRING_END = 7


def get_freeform_contd_regex():
    """Discriminate line type for free-form file"""
    ws = r"""[ \t]+"""
    anything = r"""[^\r\n]+"""
    endline = r"""(?:\n|\r\n?)"""
    comment = r"""(?:![^\r\n]*)"""
    line = r"""(?x) ^[ \t]*
            (?:
                ( {comment} {endline} )              # 1 comment line (ignored)
              | &? [ \t]* ( {anything} {endline} )   # 2 spill
              ) $
            """.format(anything=anything, comment=comment, endline=endline)
    return re.compile(line)


CONTD_COMMENT = 1
CONTD_SPILL = 2

LINECAT_NORMAL = 1
LINECAT_INCLUDE = 2
LINECAT_FORMAT = 3
LINECAT_PREPROC = 4

LINECAT_NAMES = (None, 'line', 'include', 'format', 'preproc', 'comment')


def splice_free_form(buffer):
    """Splice lines in free-form Fortran file"""
    # Iterate through lines of the file.  Fortran allows to split tokens
    # across lines, which is why we build up the whole line before giving
    # it to the tokenizer.
    stub = ''
    trunc_str = ''
    line_regex = get_freeform_line_regex()
    contd_regex = get_freeform_contd_regex()

    fname = buffer.name
    buffer_iter = enumerate(buffer)
    for lineno, line in buffer_iter:
        # Handle lines that have been truncated
        if trunc_str:
            line = trunc_str + line
            trunc_str = ''
        # Handle proper stub lines
        if stub:
            match = contd_regex.match(line)
            if match.lastindex == CONTD_COMMENT:
                continue
            line = match.group(2)

        # Now parse current (potentially preprocessed) line
        match = line_regex.match(line)
        discr = match.lastindex
        if discr == FREE_FULL_END:
            stub += match.group(FREE_WHOLE_PART) + match.group(FREE_FULL_END)
            yield lineno, LINECAT_NORMAL, stub
            stub = ''
        elif discr >= FREE_TRUNC_END:
            stub += match.group(FREE_WHOLE_PART)
            if discr == FREE_TRUNC_STRING_END:
                trunc_str = match.group(FREE_TRUNC_STRING_END)
        elif discr == FREE_PREPROC:
            ppstmt = match.group(FREE_PREPROC)
            if ppstmt[-1] == '\\':
                trunc_str = ppstmt[:-1]
            else:
                yield lineno, LINECAT_PREPROC, ppstmt + '\n'
        elif discr == FREE_FORMAT:
            yield lineno, LINECAT_FORMAT, line
        else:
            yield lineno, LINECAT_INCLUDE, line

    if stub or trunc_str:
        raise SpliceError(fname, lineno, line,
                          "File ends with line continuation marker")


def get_fixedform_line_regex():
    """Discriminate line type for fixed-form file"""
    line = r"""(?isx) ^
        (?: [cC*!](.*)                                      # 1: comment
            | [ ]{5}[^ 0] (.*)                              # 2: continuation
            | [ \t]* (\#.*)                                 # 3: preprocessor
            | [ ]{6} [ \t]* (include[ \t].*)                # 4: include line
            | ( [ ][\d ]{4}[ ] [ \t]* format[ \t]*\(.* )    # 5: format line
            | ( [ ][\d ]{4}[ ] [ \t]* .* | )                # 6: normal line
            ) $
            """
    return re.compile(line)


FIXED_COMMENT = 1
FIXED_CONTD = 2
FIXED_PREPROC = 3
FIXED_INCLUDE = 4
FIXED_FORMAT = 5
FIXED_OTHER = 6


def splice_fixed_form(buffer, margin=72):
    """Splice physical lines in fixed form fortran"""
    # The continuation markers at fixed-form lines are at the *following*
    # line, so we need to store the previous current line
    cat = None
    stub = None
    line_regex = get_fixedform_line_regex()

    fname = buffer.name
    buffer = iter(buffer)
    for lineno, line in enumerate(buffer):
        line = line[:margin].rstrip()
        match = line_regex.match(line)
        if not match:
            raise SpliceError(fname, lineno, line, "invalid fixed-form line")

        discr = match.lastindex

        if discr == FIXED_CONTD:
            if not stub:
                raise SpliceError(fname, lineno, line,
                            "continuation marker without line to continue")
            stub += match.group(FIXED_CONTD)
            continue
        else:
            if stub and discr != FIXED_COMMENT:
                yield lineno, cat, stub + "\n"
                stub = None

        if discr == FIXED_OTHER:
            cat = LINECAT_NORMAL
            stub = match.group(FIXED_OTHER)
        elif discr == FIXED_COMMENT:
            yield lineno, LINECAT_NORMAL, "!" + match.group(FIXED_COMMENT) + "\n"
        elif discr == FIXED_FORMAT:
            cat = LINECAT_FORMAT
            stub = match.group(FIXED_FORMAT)
        elif discr == FIXED_INCLUDE:
            yield lineno, LINECAT_INCLUDE, match.group(FIXED_INCLUDE) + "\n"
        else:  # discr == FIXED_PREPROC
            ppstmt = match.group(FIXED_PREPROC)
            if ppstmt[-1] == '\\':
                raise SpliceError(fname, lineno, line,
                                  "Preprocessor continuations not supported "
                                  "in fixed form")
            yield lineno, LINECAT_PREPROC, ppstmt + "\n"

    # Handle last line
    if stub is not None:
        yield lineno, cat, stub


def get_splicer(form='free'):
    """Get splice routine based on file form"""
    if form == 'free':
        return splice_free_form
    elif form == 'fixed':
        return splice_fixed_form
    else:
        raise ValueError("form must be either 'free' or 'fixed'")
