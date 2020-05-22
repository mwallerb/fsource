Command line interface
======================

fsource includes a command line tool `fsource`, which allows analyzing
Fortran files.

    $ fsource COMMAND [OPTIONS] FILE.f90 ...

The following values are allowed for `COMMAND`:

  1. `splice`: reads a Fortran file and turns it into a sequence of
     logical lines and classifying the lines. See [splicer].

  2. `lex`: reads a Fortran file and turns it into a sequence
     of *tokens*, which are the smallest units of text understood by
     the parser. See [lexer].

  3. `parse`: reads a Fortran file, splices and lexes it, then matches
     the tokens against the grammatical rules of Fortran, generating a
     hierarchical structure of items called an *abstract syntax tree* (AST).
     See [parser].

  4. `wrap`: reads a Fortran file, creates an abstract syntax tree. Then
     generate a C header file from Fortran declarations that use `BIND(C)`.
     See [wrapper].

[splicer]: splicer.md
[lexer]: lexer.md
[parser]: parser.md
[wrapper]: wrapper.md

Fortran source form
-------------------

Fortran source files come in two variants: fixed form, which is the older
punchcard-inspired format, and free form, the more modern format introduced
with Fortran 90.

By default, `fsource` will try to guess free or fixed form files from the
file name extension, but you may override this by specifying a command line
flag.  The following table summarizes the behaviour:

  | source file  | flag            | extensions                      |
  |--------------|-----------------|---------------------------------|
  | fixed form   | `--fixed-form`  | `.f`, `.f77`, `.for`            |
  | free form    | `--free-form`   | `.f90`, `.f95`, `.f03`, `.f08`  |

