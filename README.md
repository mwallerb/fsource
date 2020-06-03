fsource - Fortran static analysis tool
======================================
[![Tests]](https://travis-ci.org/mwallerb/fsource)
[![PyPI]](https://pypi.python.org/pypi/fsource)

fsource is a collection of tools allowing you to parse Fortran 77 through
Fortran 2008 programs.  It is written in pure Python and has no external
dependencies.

You install fsource via pip (you may want to append `--user` flag to install
it just for you or `--prefix=/install/path` to choose the installation location):

    $ pip install fsource

You can also simply download the [source], since there are no external dependencies.

    $ git clone github.com/mwallerb/fsource
    $ cd fsource

In this case you should run `bin/fsource` instead of `fsource`, which augments the
python path with the downloaded source files.

Command line interface
----------------------
fsource currently features a [command line interface]:

 - a [parser], which takes a Fortran file and outputs an abstract syntax tree
   (for the definitions) allowing you to extract modules, subprograms, derived
   types, parameters, etc.:

       $ fsource parse FILE.f90

 - a [wrapper], which builds on the Fortran parser to extract module variables,
   types and subroutines which can be interfaced with C and generates header
   files for them:

       $ fsource wrap FILE.f90

 - a [line splicer] and a [lexer], low-level tools which split a Fortran file
   into a set of logical lines and tokens, respectively.  This allows you to
   set up your parsing infrastructure on top of fsource:

       $ fsource splice FILE.f90
       $ fsource lex FILE.f90

[source]: https://github.com/mwallerb/fsource
[command line interface]: doc/cli.md
[line splicer]: doc/splicer.md
[lexer]: doc/lexer.md
[parser]: doc/parser.md
[wrapper]: doc/wrapper.md

[Tests]: https://travis-ci.org/mwallerb/fsource.svg?branch=master
[PyPI]: https://img.shields.io/pypi/v/fsource.svg?style=flat
