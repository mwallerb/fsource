Wrapper
=======
The wrapper allows wrapping Fortran statements in C declations

    $ fsource wrap [--fixed-form] FILENAME [FILENAME ...]

The result is a C header file.


Exact type maps
---------------

The following maps are exact:

| Fortran   | iso_c_binding KIND  | sizeof | C type             | numpy dtype |
|-----------|-----------------------|----|----------------------|-------------|
| LOGICAL   | c_bool                | 1  | _Bool                | bool_       |
| CHARACTER | c_char                | 1  | char                 | char        |
| INTEGER   | c_int                 |    | int                  | intc        |
| INTEGER   | c_short               |    | short                | short       |
| INTEGER   | c_long                |    | long                 | int_        |
| INTEGER   | c_long_long           |    | long long            | longlong    |
| INTEGER   | c_signed_char         | 1  | signed char          | byte        |
| INTEGER   | c_size_t              |    | ssize_t              | intp        |
| INTEGER   | c_int8_t              | 1  | int8_t               | int8        |
| INTEGER   | c_int16_t             | 2  | int16_t              | int16       |
| INTEGER   | c_int32_t             | 4  | int32_t              | int32       |
| INTEGER   | c_int64_t             | 8  | int64_t              | int64       |
| INTEGER   | c_intptr_t            |    | intptr_t             | intp        |
| INTEGER   | c_ptrdiff_t           |    | ptrdiff_t            | intp        |
| REAL      | c_float               | 4  | float                | float32     |
| REAL      | c_double              | 8  | double               | float64     |
| REAL      | c_long_double         |    | long double          | longdouble  |
| COMPLEX   | c_float_complex       | 8  | float _Complex       | complex64   |
| COMPLEX   | c_double_complex      | 16 | double _Complex      | complex128  |
| COMPLEX   | c_long_double_complex |    | long double _Complex | clongdouble |

The sizeof column indicates, e.g., that `INTEGER*4` is equivalent to `int32_t`,
which is true on major (all?) platforms.
