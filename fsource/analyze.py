#!/usr/bin/env python
from __future__ import print_function
from collections import OrderedDict
import copy

from . import lexer
from . import parser

class Expr:
    def __init__(self):
        pass





def ast_transformer(fn):
    "Make depth-first transformer of prefix-form abstract syntax tree"
    def ast_transform(ast):
        if isinstance(ast, tuple):
            return fn(ast[0], *map(ast_transform, ast[1:]))
        else:
            return ast
    return ast_transform

def ast_dispatcher(type_map, fallback):
    "Make depth-first transformer by dispatching based on node type"
    def dispatch(hdr, *args):
        try:
            entry = type_map[hdr]
        except KeyError:
            return fallback(hdr, *args)
        else:
            return entry(*args)

    return ast_transformer(dispatch)


def get_code(node):
    try:
        magic = node.__code__
    except AttributeError:
        if node is None:
            return " null "
        if isinstance(node, str):
            return str(node)
        return "???"
    else:
        return magic()

class Ignored:
    "Ignored item of an AST"
    def __init__(self, *args):
        self.args = args

    def imbue_entity(self, parent): pass

    def imbue_subprogram(self, parent): pass

    def imbue_module(self, parent): pass

    def __code__(self):
        return "?%s(%s)" % (self.args[0],
                            ", ".join(map(get_code, self.args[1:])))

class DimSpec:
    "Slice object in AST"
    def __init__(self, start, stop):
        if start is None:
            start = 1

        self.start = start
        self.stop = stop
        self.is_implied = stop == '*'

    def __code__(self):
        return get_code(self.start) + ":" + get_code(self.stop)

class Shape:
    def __init__(self, *dims):
        self.rank = len(dims)
        self.has_implied = dims and dims[-1].stop == '*'
        self.deferred = [e for (e, d) in enumerate(dims) if d.stop is None]
        self.dims = dims

    def imbue_entity(self, entity):
        entity.shape = self

    def __code__(self):
        return "(" + ",".join(map(get_code, self.dims)) + ")"

class IntentAttr:
    def __init__(self, in_, out):
        self.in_ = in_
        self.out = out

    def imbue_entity(self, entity):
        entity.class_ = 'argument'
        entity.intent = self

    def __code__(self):
        translate = {
            (True, False): "IN",
            (True, True):  "IN OUT",
            (False, True): "OUT"
            }
        return "INTENT(" + translate[self.in_, self.out]  + ")"

class ParameterAttr:
    def imbue_entity(self, entity):
        entity.class_ = 'parameter'

    def __code__(self): return "PARAMETER"

class ValueAttr:
    def imbue_entity(self, entity):
        entity.class_ = 'argument'
        entity.passby = 'value'

    def __code__(self): return "VALUE"

class OptionalAttr:
    def imbue_entity(self, entity):
        entity.class_ = 'argument'
        entity.required = False

    def __code__(self): return "OPTIONAL"

class PublicAttr:
    def imbue_entity(self, entity):
        entity.class_ = 'module'
        entity.visible = self

    def __code__(self): return "PUBLIC"

class Entity:
    def __init__(self, type_, attrs, name, shape_p, kind_p, init):
        self.type_ = type_
        self.name = name
        self.attrs = attrs
        self.shape_p = shape_p
        self.kind_p = kind_p
        self.init = init

    def imbue_subprogram(self, subp):
        if self.name in subp.args:
            subp.args[self.name] = self
            self.class_ = 'argument'
        else:
            self.class_ = 'local'
        self.handle_children()

    def imbue_module(self, mod):
        mod.entities[self.name] = self
        self.class_ = 'module'
        self.handle_children()

    def handle_children(self):
        self.intent = None
        self.passby = None
        self.required = None
        self.visible = None
        self.shape = None

        for attr in self.attrs:
            attr.imbue_entity(self)
        if self.shape_p is not None:
            self.shape_p.imbue_entity(self)
        if self.kind_p is not None:
            # TODO: set kind of type ...
            raise ValueError("not implemented")

        if self.shape is None:
            self.shape = Shape()
        if  self.class_ == 'argument':
            if self.intent is None: self.intent = IntentAttr(True, True)
            if self.passby is None: self.passby = 'pointer'
            if self.required is None: self.required = True
        elif self.class_ == 'module':
            if self.visible is None: self.visible = True

    def __code__(self):
        return "{type}{attrs} :: {name}{shape}{kind} {init}\n".format(
                type=get_code(self.type_),
                attrs="".join(", " + get_code(a) for a in self.attrs),
                name=get_code(self.name),
                shape=get_code(self.shape_p),
                kind="*" + get_code(self.kind_p) if self.kind_p is not None else "",
                init=get_code(self.init)
                )

class DerivedType:
    def __init__(self, name, attrs, tags, decls, procs):
        self.name = name
        self.attrs = attrs
        self.tags = tags
        self.decls = decls
        self.procs = procs

    def imbue_compilation_unit(self, unit):
        unit.modules[self.name] = self
        self.handle_children()

    def handle_children(self):
        self.dependencies = OrderedDict()
        self.entities = OrderedDict()
        self.types = OrderedDict()
        self.subprograms = OrderedDict()

        for decl in self.decls:
            decl.imbue_module(self)
        for decl in self.contained:
            decl.imbue_module(self)

    def __code__(self):
        return "TYPE{attrs} :: {name}\n{tags}{decls}END TYPE {name}\n" \
               .format(name=get_code(self.name),
                       attrs="".join(", " + get_code(a) for a in self.attrs),
                       tags="".join(get_code(t) + "\n" for t in self.tags),
                       decls="".join(map(get_code, self.decls))
                       )


class Module:
    def __init__(self, name, decls, contained):
        self.name = name

        self.decls = decls
        self.contained = contained

    def imbue_compilation_unit(self, unit):
        unit.modules[self.name] = self
        self.handle_children()

    def handle_children(self):
        self.dependencies = OrderedDict()
        self.entities = OrderedDict()
        self.types = OrderedDict()
        self.subprograms = OrderedDict()

        for decl in self.decls:
            decl.imbue_module(self)
        for decl in self.contained:
            decl.imbue_module(self)

    def __code__(self):
        return "MODULE {name}\n{decls}CONTAINS\n{cont}END MODULE {name}\n" \
               .format(name=get_code(self.name),
                       decls="".join(map(get_code, self.decls)),
                       cont="".join(map(get_code, self.contained))
                       )

class Program(Module):
    def imbue_compilation_unit(self, unit):
        unit.programs[self.name] = self
        self.handle_children()

    def __code__(self):
        return "PROGRAM {name}\n{decls}CONTAINS\n{cont}END PROGRAM {name}\n" \
               .format(name=get_code(self.name),
                       decls="".join(map(get_code, self.decls)),
                       cont="".join(map(get_code, self.contained))
                       )

class Subroutine:
    def __init__(self, name, prefixes, argsnames, bind_c, decls):
        self.name = name

        self.prefixes = prefixes
        self.argsnames = argsnames
        self.bind_c = bind_c
        self.decls = decls

    def imbue_module(self, mod):
        mod.subprograms[self.name] = self
        self.handle_children()

    def handle_children(self):
        # set up dictionary for arguments
        self.returns = None
        self.args = OrderedDict()
        nstarargs = 0
        for arg in self.argsnames:
            if arg == '*':
                arg = '*', nstarargs
                nstarargs += 1
                self.args[arg] = '*'
            else:
                self.args[arg] = None

        for prefix in self.prefixes:
            prefix.imbue_subprogram(self)
        for decl in self.decls:
            decl.imbue_subprogram(self)

    def __code__(self):
        return "{prefixes}SUBROUTINE {name}({args}) {bind_c}\n" \
               "{decls}\nEND SUBROUTINE {name}\n" \
               .format(name=get_code(self.name),
                       prefixes="".join(get_code(p) + " " for p in self.prefixes),
                       args=", ".join(map(get_code, self.argsnames)),
                       bind_c=get_code(self.bind_c),
                       decls="".join(map(get_code, self.decls)),
                       )

class CompilationUnit:
    def __init__(self, ast_version, fname, *objs):
        self.ast_version = ast_version
        self.filename = fname
        self.objs = objs

    def handle_children(self):
        self.modules = OrderedDict()
        self.programs = OrderedDict()
        for obj in self.objs:
            obj.imbue_compilation_unit(self)

    def __code__(self):
        return "\n\n".join(map(get_code, self.objs))


def ast_version(*args):
    return tuple(map(int, args))

def identifier(x): return x.lower()

def idem(x): return x

def sequence(*x): return x

HANDLERS = {
    #'arg': Ignored,
    #'array': Ignored,
    #'ast_version': ast_version,
    #'bind_c': Ignored,
    #'bool': Ignored,
    #'call': Ignored,
    #'char_sel': Ignored,
    #'character_type': Ignored,
    #'explicit_dim': Ignored,
    #'filename': Ignored,
    #'float': Ignored,
    'id': identifier,
    #'ref': Ignored,
    #'init_assign': Ignored,
    #'int': Ignored,
    #'integer_type': Ignored,
    #'interface_decl': Ignored,
    #'kind': Ignored,
    #'kind_sel': Ignored,
    #'logical_type': Ignored,
    #'neg': Ignored,
    #'only_list': Ignored,
    #'preproc_stmt': Ignored,
    #'real_type': Ignored,
    #'rename_list': Ignored,
    #'string': Ignored,
    #'type': Ignored,
    #'use_stmt': Ignored,

    'compilation_unit': CompilationUnit,
    'entity_decl': Entity,
    'module_decl': Module,
    'program_decl': Program,
    'subroutine_decl': Subroutine,
    'type_decl': DerivedType,

    'intent': IntentAttr,
    'optional': OptionalAttr,
    'parameter': ParameterAttr,
    'shape': Shape,
    'explicit_dim': DimSpec,
    'deferred_dim': DimSpec,
    'implicit_dim': DimSpec,
    'value': ValueAttr,

    'arg_list': sequence,
    'component_block': sequence,
    'contained_block': sequence,
    'declaration_block': sequence,
    'entity_attrs': sequence,
    'sub_prefix_list': sequence,
    'type_attrs': sequence,
    'type_tags': sequence,
    'interface_body': sequence,
}

my_dispatch = ast_dispatcher(HANDLERS, Ignored)

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
    ("integer(c_int)", "int"),
    ("integer(c_short)", "short"),
    ("integer(c_long)", "long"),
    ("integer(c_long_long)", "long long"),
    ("integer(c_signed_char)", "signed char"),
    ("integer(c_size_t)", "size_t"),
    ("integer(c_int8_t)", "int8_t"),
    ("integer(c_int16_t)", "int16_t"),
    ("integer(c_int32_t)", "int32_t"),
    ("integer(c_int64_t)", "int64_t"),
    ("integer(c_intptr_t)", "intptr_t"),
    ("real(c_float)", "float"),
    ("real(c_double)", "double"),
    ("real(c_long_double)", "long double"),
    ("complex(c_float_complex)", "float _Complex"),
    ("complex(c_double_complex)", "double _Complex"),
    ("complex(c_long_double_complex)", "long double _Complex"),
    ("logical(c_bool)", "_Bool"),
    ("character(kind=c_char)", "char"),
    }


# These equivalences are valid on any reasonable architecture.
# If you find a counterexample, please file a bug.
ASSUMED_EQUIVALENCE = {
    ("logical", "logical(c_bool)"),
    ("character", "character(kind=c_char)"),

    # FIXME
    ("integer(pint)", "integer(c_int64_t)"),
    }

def dress_ctype(ctype, entity):
    "Add pointers and const qualifiers based on the entity"
    dressed = copy.copy(ctype)
    if entity.shape.rank > 0:
        dressed.ptr = True
    if not entity.required:
        dressed.ptr = True
    if not entity.intent.out:
        if dressed.ptr: dressed.const = True
    else:
        dressed.ptr = True
    return dressed



