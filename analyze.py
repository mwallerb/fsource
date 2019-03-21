#!/usr/bin/env python
from __future__ import print_function
from collections import OrderedDict

import lexer

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

    # Fallback is to just reproduce the node as-is with header etc.
    if fallback is None:
        fallback = lambda *args: args

    def dispatch(hdr, *args):
        try:
            return type_map[hdr](*args)
        except KeyError:
            return fallback(hdr, *args)

    return ast_transformer(dispatch)


class Shape:
    def __init__(self, *dims):
        self.rank = len(dims)
        self.has_implied = dims[-1].stop == '*'
        self.deferred = [e for (e, d) in enumerate(dims) if d.stop is None]
        self.dims = dims

    def imbue_entity(self, entity):
        if entity.shape is not None:
            raise ValueError("meh")
        entity.shape = self

    def __repr__(self): return repr(self.__dict__)

class IntentAttr:
    def __init__(self, in_, out):
        self.in_ = in_
        self.out = out

    def imbue_entity(self, entity):
        entity.set_class('argument')
        entity.intent = self

    def __repr__(self): return repr(self.__dict__)

class ParameterAttr:
    def __init__(self): pass
    def imbue_entity(self, entity): entity.set_class('parameter')

class ValueAttr:
    def __init__(self): pass
    def imbue_entity(self, entity):
        entity.set_class('argument')
        entity.passby = 'value'

class OptionalAttr:
    def __init__(self): pass
    def imbue_entity(self, entity):
        entity.set_class('argument')
        entity.required = False

class IgnoredAttr:
    def __init__(self, *args): pass
    def imbue_entity(self, entity): pass
    def imbue_subprogram(self, subp): pass
    def imbue_module(self, mod): pass

class Entity:
    def set_class(self, new_class):
        if self.class_ is None:
            self.class_ = new_class
        elif self.class_ != new_class:
            raise ValueError("class conflict for entity %s: %s -> %s" %
                             (self.name, self.class_, new_class))

    def __init__(self, type_, attrs, name, shape_p, kind_p, init):
        self.type_ = type_
        self.name = name
        self.init = init

        self.class_ = None
        self.shape = None
        self.intent = None
        self.passby = 'reference'
        self.required = True

        for attr in attrs:
            attr.imbue_entity(self)
        if shape_p is not None:
            shape_p.imbue_entity(self)
        if kind_p is not None:
            # TODO: set kind of type ...
            raise ValueError("not implemented")

    def imbue_subprogram(self, subp):
        if self.name in subp.args:
            subp.args[self.name] = self
            self.set_class('argument')

    def imbue_module(self, mod):
        mod.entities[self.name] = self

class Module:
    def __init__(self, name, decls, contained):
        self.name = name

        self.dependencies = OrderedDict()
        self.entities = OrderedDict()
        self.types = OrderedDict()
        self.subprograms = OrderedDict()

        for decl in decls:
            print (decl)
            decl.imbue_module(self)
        if contained is not None:
            for decl in contained:
                print (decl)
                decl.imbue_module(self)

class Program(Module): pass

class Subroutine:
    def __init__(self, name, prefixes, args, bind_c, decls):
        self.name = name

        # set up dictionary for arguments
        self.args = OrderedDict()
        nstarargs = 0
        for arg in args:
            if arg == '*':
                arg = '*', nstarargs
                nstarargs += 1
            self.args[arg] = None

        for prefix in prefixes:
            prefix.imbue_subprogram(self)
        print (decls)
        for decl in decls:
            decl.imbue_subprogram(self)

        self.bind_c = bind_c

    def imbue_module(self, mod):
        mod.subprograms[self.name] = self
        self.container = mod

def ast_version(*args):
    return tuple(map(int, args))

def idem(x): return x

def idems(*x): return x

tf = {
    "ast_version": ast_version,
    "filename": idem,
    "id": idem,
    "string": lexer.parse_string,
    "int": int,
    "bool": lexer.parse_bool,
    "float": lexer.parse_float,
    "explicit_dim": lambda l, u: slice((1 if l is None else l), u),
    "deferred_dim": lambda l, u: slice((1 if l is None else l), u),
    "implicit_dim": lambda l, u: slice((1 if l is None else l), u),
    "entity_attrs": idems,
    "declaration_block": idems,
    "contained_block": idems,
    "entity_decl": Entity,
    "module_decl": Module,
    "interface_decl": IgnoredAttr,

    "shape": Shape,
    "parameter": ParameterAttr,
    "intent": IntentAttr,
    "optional": OptionalAttr,
    "value": ValueAttr,
    "type_decl": IgnoredAttr,
    "program_decl": Program,

    "subroutine_decl": Subroutine,
    "arg_list": idems,
    "sub_prefix_list": idems,
    "use_stmt": IgnoredAttr,
}

my_dispatch = ast_dispatcher(tf)
