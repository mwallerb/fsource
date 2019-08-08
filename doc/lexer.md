Lexical analysis
================
The lexer or lexical analyser understands free-form and fixed-form Fortran
source files, from version 77 up to version 2008.  You can call the lexer on
free-form source files as follows:

    $ fsource lex [--fixed-form] FILENAME [FILENAME ...]

For fixed-form source files, one must specify the `--fixed-form` option.

The lexer uses regular expressions to split up a Fortran source file into a
sequence of *tokens* (a.k.a. terminal symbols), where each token has a
*category* attached to it signifying its type.  The lexer writes a JSON array
to stdout: it starts with a header, followed by a sequence of pairs of the form
`[cat, token]`, where `cat` is a category string (see later) and `token` is the
character corresponding to the token.

Token categories
----------------
Each token is associated with one category of the following:

 | code              | explanation              | example              |
 |-------------------|--------------------------|----------------------|
 | `eof`             | end of file marker       |                      |
 | `preproc`         | preprocessor statement   | `#ifdef INTEL\n`     |
 | `eos`             | end of statement marker  | `\n`                 |
 | `string`          | string literal           | `'Donald''s things'` |
 | `float`           | real number literal      | `1.0D0`              |
 | `int`             | integer number literal   | `4711`               |
 | `radix`           | hex/octal/binary literal | `b"0010001"`         |
 | `bool`            | boolean literal          | .`true`.             |
 | `dotop`           | dot-delimited operator   | .`eq`.               |
 | `custom_dotop`    | user-defined operator    | .`myoperator`.       |
 | `bracketed_slash` | bracketed slash(es)      | (`//`)               |
 | `symop`           | symbolic operator        | `**`                 |
 | `format`          | format statement         | `FORMAT (3I2)\n`     |
 | `word`            | identifier or keyword    | `counter`            |

**Note** that enclosing dots in dot-delimited operators as well as the
enclosing brackets for bracketed slashes are not included in the token string.

Lexical analysis must deal with three ambiguities in the Fortran grammar:

 1. The string `::` can mean an empty slice or a separator token.  The
    lexer always returns single `:`, which means one gets `::` as a
    sequence of two `:` and cannot detect whitespace within a seperator.

 2. `(//)` can mean an empty inplace array or an overloaded `//` operator,
    and `(/)`, which is ambiguous for a similar reason.  To work around
    this lexer will return a token of category `bracketed_slash`, and
    the application must disambiguate.

 3. The `FORMAT` statement is a bit of an oddball, as it allows tokens that
    are illegal everywhere else, e.g., `3I6` or `ES13.2`.  The lexer works
    around this by returning the format line as single token of category
    `format`.

