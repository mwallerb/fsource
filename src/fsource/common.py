"""
Common classes for fsource.

Copyright 2019 Markus Wallerberger
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
from __future__ import print_function
import re


class ParsingError(Exception):
    """Base exception class for parsing errors"""
    def __init__(self, fname, lineno, colbegin, colend, line, msg):
        """Construct new parsing exception"""
        Exception.__init__(self)
        self.fname = fname
        self.lineno = lineno
        self.colbegin = colbegin
        self.colend = colend
        self.line = line
        self.msg = msg

    @property
    def error_type(self):
        """Type of parsing error"""
        return "error"

    def errmsg(self):
        """Error message for fancy-printing in command line interface"""
        errstr = ""
        if self.fname is not None:
            errstr += self.fname + ":"
        if self.lineno is not None:
            errstr += str(self.lineno + 1) + ":"
        if self.colbegin is not None:
            errstr += str(self.colbegin + 1) + ":"
        errstr += " " + self.error_type
        if self.msg is not None:
            errstr += ": " + self.msg + "\n"
        if self.line is not None:
            errstr += "|\n"
            for line in self.line.splitlines():
                errstr += "|\t%s\n" % line
            if self.colbegin is not None:
                errstr += "|\t" + " " * self.colbegin + "^"
                if self.colend is not None:
                    errstr += "~" * (self.colend - self.colbegin - 1)
                errstr += "\n"
        return errstr

    def __str__(self):
        return "\n" + self.errmsg()   # TODO


def _extension_switch_re():
    # Please don't add other years here.
    exts = r"""(?: ( for | f77 | f )             # 1: fixed form, final
                 | ( FOR | F77 | F )             # 2: fixed form, preprocess
                 | ( f90 | f95 | f03 | f08 )     # 3: free form,  final
                 | ( F90 | F95 | F03 | F08 )     # 4: free form,  preprocess
                 )"""
    return re.compile(r"""(?x) \.{exts}$""".format(exts=exts))


def guess_form(fname):
    """Guess source form format from file name"""
    guesser = _extension_switch_re()
    match = guesser.search(fname)
    if not match:
        raise ValueError("Unable to guess whether file is fixed or free form")

    discr = match.lastindex - 1
    form = 'free' if discr & 2 else 'fixed'
    is_preproc = bool(discr & 1)
    return form, is_preproc
