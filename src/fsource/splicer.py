"""
Line handling and preprocessing for free/fixed-form Fortran.

Uses regular expressions to split up a Fortran source file into a sequence
of logical lines together with line types.

Copyright 2019 Markus Wallerberger
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
from __future__ import print_function
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
    endline = r"""(?:\n|\r\n?)"""
    comment = r"""(?:!.*)"""
    atom = r"""(?: [^!&'"\r\n] | '(?:''|[^'\r\n])*' | "(?:""|[^"\r\n])*" )"""
    truncstr = r"""(?: '(?:''|[^'\r\n])*
                     | "(?:""|[^"\r\n])*
                     )
                """
    line = r"""(?x) ^[ \t]*
        (?: ( \# .* ) {endline}                      # 1 preprocessor stmt
            | ( {atom}* ) (?:                        # 2 whole line part
                  ( {comment}? {endline} )           # 3 full line end
                | ( & [ \t]* {comment}? {endline} )  # 4 truncated line end
                | ( {truncstr} ) &[ \t]* {endline}   # 5 truncated string end
              )
            ) $
        """.format(atom=atom, truncstr=truncstr, comment=comment,
                   endline=endline)

    return re.compile(line)


FREE_PREPROC = 1
FREE_WHOLE_PART = 2
FREE_FULL_END = 3
FREE_TRUNC_END = 4
FREE_TRUNC_STRING_END = 5


def get_freeform_contd_regex():
    """Discriminate line type for free-form file"""
    endline = r"""(?:\n|\r\n?)"""
    line = r"""(?x) ^[ \t]*
            (?:
                ( ! .* {endline} )              # 1 comment line (ignored)
              | &? [ \t]* ( .+ {endline} )      # 2 spill
              ) $
            """.format(endline=endline)
    return re.compile(line)


CONTD_COMMENT = 1
CONTD_SPILL = 2

LINECAT_NORMAL = 1
LINECAT_PREPROC = 2

LINECAT_NAMES = (None, 'line', 'preproc')


def splice_free_form(mybuffer):
    """Splice lines in free-form Fortran file"""
    # Iterate through lines of the file.  Fortran allows to split tokens
    # across lines, which is why we build up the whole line before giving
    # it to the tokenizer.
    stub = ''
    trunc_str = ''
    line_regex = get_freeform_line_regex()
    contd_regex = get_freeform_contd_regex()

    fname = mybuffer.name
    lineno = 0
    for lineno, line in enumerate(mybuffer):
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
        else: # discr == FREE_PREPROC:
            ppstmt = match.group(FREE_PREPROC)
            if ppstmt[-1] == '\\':
                trunc_str = ppstmt[:-1]
            else:
                yield lineno, LINECAT_PREPROC, ppstmt + '\n'

    if stub or trunc_str:
        raise SpliceError(fname, lineno, line,
                          "File ends with line continuation marker")


def get_fixedform_line_regex(margin):
    """Discriminate line type for fixed-form file"""
    line = r"""(?mx) ^
        (?: [cC*!] (.*)                                   # 1: comment
          | [ ]    [ ]{{4}}  [^ 0]  (.{{0,{body}}}) .*    # 2: continuation
          | [ ]*                    (\#.*)                # 3: preprocessor
          | (    [\d ]{{5}}  [ 0]    .{{0,{body}}}
            |    [ ]{{0,5}} $                     ) .*    # 4: normal line
          ) $
          """.format(body=margin-6)
    return re.compile(line)


FIXED_COMMENT = 1
FIXED_CONTD = 2
FIXED_PREPROC = 3
FIXED_OTHER = 4


def splice_fixed_form(mybuffer, margin=72):
    """Splice physical lines in fixed form fortran"""
    # The continuation markers at fixed-form lines are at the *following*
    # line, so we need to store the previous current line
    cat = None
    stub = None
    line_regex = get_fixedform_line_regex(margin)

    fname = mybuffer.name
    lineno = 0
    for lineno, line in enumerate(mybuffer):
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
        elif stub:
            # Discard comment lines in between continuations
            if discr == FIXED_COMMENT:
                continue
            yield lineno-1, cat, stub + "\n"
            stub = None

        if discr == FIXED_OTHER:
            cat = LINECAT_NORMAL
            stub = match.group(FIXED_OTHER)
        elif discr == FIXED_COMMENT:
            yield lineno, LINECAT_NORMAL, "!" + match.group(FIXED_COMMENT) + "\n"
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
