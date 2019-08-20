"""
Common classes for fsource.

Copyright 2019 Markus Wallerberger
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
from __future__ import print_function

class ParsingError(Exception):
    def __init__(self, fname, lineno, colbegin, colend, line, msg):
        self.fname = fname
        self.lineno = lineno
        self.colbegin = colbegin
        self.colend = colend
        self.line = line
        self.msg = msg

    @property
    def error_type(self):
        return "error"

    def errmsg(self):
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
            errstr += "|\n|\t%s" % self.line
            if self.colbegin is not None:
                errstr += ("|\t" + " "*self.colbegin + "^"
                           + "~"*(self.colend - self.colbegin) + "\n")
        return errstr

    def __str__(self):
        return "\n" + self.errmsg()   # TODO

