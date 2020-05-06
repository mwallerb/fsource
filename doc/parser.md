Parser
======
The parser takes the stream of tokens from the lexer and matches it against the
grammatical rules of Fortran.   Most of the rules generate a node, and since
rules form a hierarchy, the result is a tree structure called an _abstract
syntax tree_ (AST).  You can call the parser as follows:

    $ fsource parse [--fixed-form] FILENAME [FILENAME ...]

For fixed-form source files, one must specify the `--fixed-form` option.

The result is an abstract syntax tree, represented in JSON as an S-expression:
each node is either a terminal or non-terminal node.  A terminal node usually
represents a token and can either be `null`, `true`, `false`, or any string.
A non-terminal node is a list `[name, ...]`, where `name` is a string
describing the type of node and subsequent items (if any) are the nodes
children.

Fortran compatibility
---------------------
The fsource parser strives to be fully compliant with Fortran 2008 and with its
most common extensions.  We focus on modern Fortran: obscure or outdated
syntax like `a.b` for structure resolution is not supported.

Things not yet implemented are:

  - `submodule` blocks
  - coarrays, `codimension`
  - derived type parametrization

Preprocessor statements
-----------------------
The fsource parsing step handles preprocessor statements transparently by
emitting a node of type `preproc_stmt`.  This allows you to perform analysis
of the AST with preprocessor information such as `#pragma`s and `#ifdef`s
in place.

However, for this to work, the Fortran program with all preprocessor statements
removed must still be parsable by fsource.  For example, the following source
cannot not parsed without prior preprocessing:

    #ifdef MPI
        if (MPI_Get_rank(comm) == 0 .and. x == 0)
    #else
        if (x == 0)
    #endif
            call do_something
        endif

It is usually a good idea to avoid constructs like this and have preprocessor
statements respect the logic of the program (see [Linux Kernel Coding Style]
for a more detailed rationale).  For example, the above example can be
rewritten as:

    #ifndef MPI
        integer function MPI_Get_rank(comm)
            integer, intent(in) :: comm
            MPI_Get_rank = 0
        end function
    #endif
    ...
    if (MPI_Get_rank(comm) == 0 .and. x == 0)
        call do_something
    endif

Similary, if identifiers are computed from a macro function, it is useful to
define a temporary name for them.


[Linux Kernel Coding Style]: https://www.kernel.org/doc/html/latest/process/coding-style.html#conditional-compilation
