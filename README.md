fsource - Fortran static analysis tool
======================================

fsource is a collection of tools allowing you to parse Fortran 77 through
Fortran 2008 programs.  It is written in pure Python and has no external
dependencies.

You install fsource via pip:

    $ pip install fsource

or simply download the [source], since there are no external dependencies
(note that you should use `bin/fsource` instead of `fsource` in this case).

fsource currently features a [command line interface]:

 - a [parser], which takes a Fortran file and outputs an abstract syntax tree
   (for the definitions) allowing you to extract modules, subprograms, derived
   types, parameters, etc.:

       $ fsource parse FILE.f90

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
