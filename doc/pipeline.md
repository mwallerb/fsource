Parsing Fortran
===============
fsource parses Fortran in three main steps:

  1. *Splicing*: reads the free/fixed format Fortran file and turns it
     into a sequence of logical lines.

  2. *Lexical analysis*: reads the sequence of lines and turns it
     into a sequence of *tokens*, which are the smallest units of text
     understood by the parser.

  3. *Parsing*: takes the stream of tokens from the lexer and matches it
     against the grammatical rules of Fortran, generating a hierarchical
     structure of items called an *abstract syntax tree* (AST).

fsource allows you to control each of these steps individually, look at the
output and modify them as needed.
