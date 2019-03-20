#!/usr/bin/env python
from __future__ import print_function

def ast_transformer(fn):
    "Make depth-first transformer of prefix-form abstract syntax tree"
    def ast_transform(ast):
        if isinstance(ast, tuple):
            return fn(ast[0], *map(ast_transform, ast[1:]))
        else:
            return ast
    return ast_transform

def ast_dispatcher(type_map, fallback=None):
    "Make depth-first transformer by dispatching based on node type"
    if fallback is None:
        fallback = lambda *args: args

    def dispatch(hdr, *args):
        return type_map.get(hdr, fallback)(*args)

    return ast_transformer(dispatch)

