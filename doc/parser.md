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


