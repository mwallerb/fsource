#!/usr/bin/env python
from __future__ import print_function
from collections import OrderedDict
import copy

import lexer
import parser

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

class Overridable:
    def __init__(self, name, default=None):
        self.name = name
        self.value_attr = "__overridable" + name
        self.default_attr = "__default" + name
        self.default = default

    def __get__(self, obj, owner):
        try:
            return getattr(obj, self.value_attr)
        except AttributeError:
            try:
                return getattr(obj. self.default_attr)
            except AttributeError:
                return self.default

    def __set__(self, obj, new_value):
        try:
            current = getattr(obj, self.value_attr)
        except AttributeError:
            setattr(obj, self.value_attr, new_value)
        else:
            if current != new_value:
                raise RuntimeError("conflicting values for %s: %s -> %s" %
                                   (self.name, current, new_value))

    def set_default(self, obj, new_default):
        setattr(obj, self.default_attr, new_default)

    def __delete__(self, obj):
        delattr(obj, self.value_attr)
        delattr(obj, self.default_attr)

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
        entity.class_ = 'argument'
        entity.intent = self

    def __repr__(self): return repr(self.__dict__)

class ParameterAttr:
    def imbue_entity(self, entity):
        entity.class_ = 'parameter'

class ValueAttr:
    def imbue_entity(self, entity):
        entity.class_ = 'argument'
        entity.passby = 'value'

class OptionalAttr:
    def imbue_entity(self, entity):
        entity.class_ = 'argument'
        entity.required = False

class IgnoredAttr:
    def __init__(self, *args): pass
    def imbue_entity(self, entity): pass
    def imbue_subprogram(self, subp): pass
    def imbue_module(self, mod): pass

class Entity:
    def __init__(self, type_, attrs, name, shape_p, kind_p, init):
        self.type_ = type_
        self.name = name
        self.init = init

        for attr in attrs:
            attr.imbue_entity(self)
        if shape_p is not None:
            shape_p.imbue_entity(self)
        if kind_p is not None:
            # TODO: set kind of type ...
            raise ValueError("not implemented")

    # Attributes that can be controlled from different sources.
    class_ = Overridable("class_", "local")
    intent = Overridable("intent", IntentAttr(True, True))
    passby = Overridable("passby", "reference")
    required = Overridable("required", True)
    shape = Overridable("shape", None)

    def imbue_subprogram(self, subp):
        if self.name in subp.args:
            subp.args[self.name] = self
            self.class_ = 'argument'

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
            decl.imbue_module(self)
        if contained is not None:
            for decl in contained:
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

# ------------------

def parse_type(typestr):
    return parser.type_spec(parser.TokenStream(typestr))

class CType:
    def __init__(self, base, const=False, volatile=False, ptr=None):
        self.base = base
        self.const = const
        self.volatile = volatile
        self.ptr = ptr

    def __str__(self):
        return ("", "const ")[self.const] \
             + ("", "volatile ")[self.volatile] \
             + self.base \
             + (" ", " *")[self.ptr is not None]

    def __repr__(self):
        return "CType('%s')" % str(self)

# These typemaps are exact as guaranteed by the Fortran standard
EXACT_TYPEMAP = {
    parse_type("integer(c_int)"): CType("int"),
    parse_type("integer(c_short)"): CType("short"),
    parse_type("integer(c_long)"): CType("long"),
    parse_type("integer(c_long_long)"): CType("long long"),
    parse_type("integer(c_signed_char)"): CType("signed char"),
    parse_type("integer(c_size_t)"): CType("size_t"),
    parse_type("integer(c_int8_t)"): CType("int8_t"),
    parse_type("integer(c_int16_t)"): CType("int16_t"),
    parse_type("integer(c_int32_t)"): CType("int32_t"),
    parse_type("integer(c_int64_t)"): CType("int64_t"),
    parse_type("integer(c_intptr_t)"): CType("intptr_t"),
    parse_type("real(c_float)"): CType("float"),
    parse_type("real(c_double)"): CType("double"),
    parse_type("real(c_long_double)"): CType("long double"),
    parse_type("complex(c_float_complex)"): CType("float _Complex"),
    parse_type("complex(c_double_complex)"): CType("double _Complex"),
    parse_type("complex(c_long_double_complex)"): CType("long double _Complex"),
    parse_type("logical(c_bool)"): CType("_Bool"),
    parse_type("character(kind=c_char)"): CType("char"),
    }

# These equivalences are valid on any reasonable architecture.
# If you find a counterexample, please file a bug.
ASSUMED_EQUIVALENCE = {
    parse_type("logical"): parse_type("logical(c_bool)"),
    parse_type("character"): parse_type("character(kind=c_char)"),

    # TODO:
    parse_type("integer(pint)"): parse_type("integer(c_int64_t)")
    }

def dress_ctype(ctype, entity):
    "Add pointers and const qualifiers based on the entity"
    dressed = copy.copy(ctype)
    if entity.shape is not None and entity.shape.rank > 0:
        dressed.ptr = True
    if not entity.required:
        dressed.ptr = True
    if not entity.intent.out:
        if dressed.ptr: dressed.const = True
    else:
        dressed.ptr = True
    return dressed



