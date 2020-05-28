"""
Copyright 2019 Markus Wallerberger
Released under the GNU Lesser General Public License, Version 3 only.
See LICENSE.txt for permissions on usage, modification and distribution
"""
from __future__ import print_function

import io
import sys
import contextlib
import textwrap
import warnings

from . import parser
from . import lexer
from . import common


_HANDLER_REGISTRY = {}


def ast_handler(ast_tag):
    """Decorate class to handle AST node of certain type"""
    def ast_handler_property(class_):
        global _HANDLER_REGISTRY
        _HANDLER_REGISTRY[ast_tag] = class_
        return class_

    return ast_handler_property


class Node(object):
    """Base class of objects in the abstract syntax representation (ASR)"""
    def imbue(self, parent):
        """Imbue `parent` with information from `self`.

        When objects are created, they only have information about their
        children.  `imbue` is called *after* the complete hierarchy is
        established.  The node `self` is expected to:

         1. publish itself in the parent's namespace, if applicable, and set
            up its own namespace, either by linking to the parents or creating
            a new one.

         2. augment or change the nature of the parent based on the information
            or presence of the current node and its children.
        """
        raise NotImplementedError("imbue is not implemented")

    def resolve(self):
        """Complete namespace and resolve all names in `self`.

        Name resolution in Fortran is slightly tricky: declarations can appear
        before or after its first use.  Also, modules from multiple files can
        reference each other.  Therefore, we need a separate `resolve()` stage.

        This method must be called *after* `imbue()`.  `self` is expected to:

         1. complete its namespace, if present, with all imported names from
            modules and such and with the parent's namespace

         2. resolve name references of itself within the complete namespace
            and recursively descend to its children, if any.
        """
        raise NotImplementedError("resolve is not implemented")

    def fcode(self):
        """Get code of self as string"""
        raise NotImplementedError("want code")


def fcode(obj):
    """Method form of `Node.fcode()` for use in `map`."""
    return obj.fcode()


class Namespace:
    """A set of defined (or known) names in a certain node.

    A namespace is a map of intrinsic (system-defined) names plus a map of
    user-defined names to their respective objects.  A `Node` declares its
    namespace in its `imbue()` method and populates it with the objects
    defined in its scope.  In the `resolve()` method, it will augment its
    namespace with all names known at that point (e.g., through import).
    """
    @classmethod
    def union(cls, *contexts):
        """Union of namespaces, which must be pairwise disjoint"""
        def disjoint_union(dicts):
            result = {}
            for curr in dicts:
                oldlen = len(result)
                result.update(curr)
                if len(result) != oldlen + len(curr):
                    raise ValueError("duplicate items")
            return result

        return cls(disjoint_union(c.names for c in contexts))

    def __init__(self, names=None):
        """Initialize new namespace, which by default is empty."""
        if names is None:
            names = {}
        self.names = names

    def get(self, name):
        """Get node identified by `name` in this namespace"""
        try:
            return self.names[name]
        except KeyError:
            return self.get_intrinsic(name)

    def add(self, node, name=None):
        """Publish `node` in this namespace"""
        if name is None:
            name = node.name
        if name in self.names:
            # TODO: error or at least disable wrapping...?
            warnings.warn("duplicate in namespace: {}".format(name))
        self.names[name] = node

    def inherit(self, context):
        """Import names names from `context` (own names take precedence)"""
        newnames = dict(context.names)
        newnames.update(self.names)
        self.names = newnames

    @classmethod
    def get_intrinsic(cls, name):
        """Get name of intrinsic (system-defined) object"""
        try:
            intrinsics = cls._intrinsics
        except AttributeError:
            cls._intrinsics = {
                'iso_c_binding': IsoCBindingModule(),
                'iflport':       OpaqueModule('iflport'),
                }
            intrinsics = cls._intrinsics
        return intrinsics[name]


class Config:
    def __init__(self, prefix="", implicit_none=True, opaque_scalars=True,
                 opaque_structs=True):
        self.prefix = prefix
        self.implicit_none = implicit_none
        self.opaque_scalars = opaque_scalars
        self.opaque_structs = opaque_structs

    def cname(self, fqname):
        return self.prefix + "_".join(fqname.split("%%"))


class CWrapper:
    @classmethod
    def union(cls, elems, config, sep=""):
        wraps = tuple(elem.cdecl(config) for elem in elems)
        return cls(sep.join(w.decl for w in wraps),
                   sep.join(w.fdecl for w in wraps),
                   inherit=wraps)

    def __init__(self, decl="", fdecl="", fails=(), headers=(),
                 opaque_structs=(), opaque_scalars=(), inherit=()):
        self.decl = decl
        self.fdecl = fdecl
        self.fails = sum((w.fails for w in inherit), fails)
        self.headers = set(headers).union(*(w.headers for w in inherit))
        self.opaque_structs = set(opaque_structs).union(
                                    *(w.opaque_structs for w in inherit))
        self.opaque_scalars = set(opaque_scalars).union(
                                    *(w.opaque_scalars for w in inherit))

    @classmethod
    def fail(cls, name, msg, subfails=()):
        return cls(fails=((name, msg, subfails),))

    @classmethod
    def _format_fail(cls, name, msg, children, prefix):
        cprefix = prefix + "  "
        failstr = "{}- {}: {}\n".format(prefix, name, msg)
        failstr += "".join(cls._format_fail(*child, prefix=cprefix)
                           for child in children)
        return failstr

    def get(self, add_fails=True):
        output = ""
        if self.fails and add_fails:
            output += "/*\n * Wrapping failures:\n"
            output += "".join(self._format_fail(*fail, prefix=" *   ")
                               for fail in self.fails)
            output += " */\n\n"

        if self.headers:
            output += "/* Includes */\n"
            output += "".join("#include {}\n".format(h) for h in self.headers)
            output += "\n"

        if self.opaque_structs:
            output += "/* Opaque structure types */\n"
            output += "".join("struct {};\n".format(s)
                              for s in self.opaque_structs)
            output += "\n"

        if self.opaque_scalars:
            output += "/* Opaque scalar types */\n"
            output += "".join("struct _{s};\n".format(s=s)
                              for s in self.opaque_scalars)
            output += "\n"
            output += "".join("typedef struct _{s} {s};\n".format(s=s)
                              for s in self.opaque_scalars)
            output += "\n"

        if self.decl:
            output += "/* Declarations */\n{}\n".format(self.decl)

        return output


class Ignored(Node):
    """Ignored node of the AST"""
    def __init__(self, node_type, *children):
        self.node_type = node_type
        self.children = children
        warnings.warn("IGNORED NODE %s" % self.node_type)

    def imbue(self, parent):
        pass

    def resolve(self):
        pass

    def fcode(self):
        return "$%s$" % self.node_type

    def cdecl(self, config):
        return CWrapper.fail(self.node_type, "unable to wrap (ignored)")


class NamespaceNode:
    """Node which has its own namespace (modules, subroutines, etc.)"""
    @property
    def children(self):
        raise NotImplementedError("children must be implemented")

    def __init__(self, name):
        self.name = name

    def imbue(self, parent=None):
        self.parent = parent
        if parent is not None:
            self.parent.namespace.add(self)
        self.namespace = Namespace()

        for child in self.children:
            child.imbue(self)

    def resolve(self):
        if self.parent is not None:
            self.namespace.inherit(self.parent.namespace)
        for child in self.children:
            child.resolve()

    @property
    def fqname(self):
        if self.parent is None:
            return self.name
        return "{}%%{}".format(self.parent.fqname, self.name)

    def cdecl(self, config, ref=None):
        if ref is not None:
            raise RuntimeError("do not know what to do with ref()")
        return CWrapper.union(self.children, config)


class TransparentNode:
    def __init__(self, *children):
        self.children = children

    def imbue(self, parent):
        self.namespace = parent.namespace
        for child in self.children:
            child.imbue(parent)

    def resolve(self):
        for child in self.children:
            child.resolve()

    def fcode(self):
        return "".join(map(fcode, self.children))

    def cdecl(self, config, ref=None):
        if ref is not None:
            raise RuntimeError("do not know what to do with ref()")
        return CWrapper.union(self.children, config)


class ReferenceNode(Node):
    def __init__(self, name):
        self.name = name
        self.ref = None

    def imbue(self, parent):
        self.parent = parent
        self.namespace = parent.namespace

    def resolve(self):
        try:
            self.ref = self.namespace.get(self.name)
            return True
        except KeyError:
            if hasattr(self.parent, "fqname"):
                parentname = self.parent.fqname
            else:
                parentname = str(self.parent)
            warnings.warn("Failure to resolve {name} within {parent}".format(
                                name=self.name, parent=parentname))
            return False

    def fcode(self):
        # We return the *name*, not the object.
        if self.ref is not None:
            return self.ref.fqname
        return self.name

    def cdecl(self, config):
        if self.ref is None:
            return CWrapper.fail(self.name, "unknown reference")
        return self.ref.cdecl(config, ref=self)


@ast_handler("compilation_unit")
class CompilationUnit(NamespaceNode):
    """Top node representing one file"""
    def __init__(self, ast_version, fname, *decls):
        NamespaceNode.__init__(self, "")
        self.ast_version = ast_version
        self.filename = fname
        self.decls = decls

    @property
    def children(self): return self.decls

    def fcode(self):
        return textwrap.dedent("""\
            ! file {file}
            ! ast version {astv}
            {decls}! end file {file}
            """).format(file=self.filename,
                        astv=str(self.ast_version),
                        decls="\n".join(map(fcode, self.children))
                        )


@ast_handler("module_decl")
class Module(NamespaceNode):
    def __init__(self, name, decls, contained):
        NamespaceNode.__init__(self, name)
        self.decls = decls
        self.contained = contained

    @property
    def children(self): return self.decls + self.contained

    def imbue(self, parent):
        NamespaceNode.imbue(self, parent)
        self.resolve_guard = False

    def resolve(self):
        # Resolve can be "forced" from a USE statement, so we do not know
        # when it is going to be called.
        if self.resolve_guard:
            raise RuntimeError("cyclic module dependency detected")
        self.resolve_guard = True
        NamespaceNode.resolve(self)
        self.resolve_guard = False

    def fcode(self):
        return textwrap.dedent("""\
            module {name}
            {decls}
            contains
            {contained}
            end module {name}
            """).format(name=self.name,
                        decls="".join(map(fcode, self.decls)),
                        contained="\n".join(map(fcode, self.contained))
                        )


@ast_handler("use_stmt")
class Use(ReferenceNode):
    def __init__(self, modulename, attrs, only, *symbollist):
        ReferenceNode.__init__(self, modulename)
        self.attrs = attrs
        self.only = only is not None
        self.symbollist = symbollist

    def resolve(self):
        if not ReferenceNode.resolve(self):
            return

        # Force recursive import
        # TODO: here we technically need to disambiguate
        #  - module A using module B, C, both which using D
        #  - module A using module B, C, both of which define same symbol
        # TODO: filter names with only and renames
        self.ref.resolve()
        self.namespace.inherit(self.ref.namespace)

    def fcode(self):
        return "use {name}{sep}{only}{symlist}\n".format(
            name=self.name,
            sep=", " if self.only or self.symbollist else "",
            only="only: " if self.only else "",
            symlist=", ".join(map(fcode, self.symbollist)))

    def cdecl(self, config):
        return CWrapper()


@ast_handler("type_decl")
class DerivedType(NamespaceNode):
    def __init__(self, name, attrs, tags, decls, procs):
        NamespaceNode.__init__(self, name)
        self.attrs = attrs
        self.tags = tags
        self.decls = decls
        # TODO: use some better type here
        if procs is None:
            procs = TypeBoundProcedureList(False)
        self.procs = procs

    @property
    def children(self): return self.attrs + self.tags + self.decls + (self.procs,)

    def imbue(self, parent):
        self.cname = None
        self.is_private = False
        self.is_sequence = False
        self.extends = None
        # TODO: remove this?
        self.fields = []
        NamespaceNode.imbue(self, parent)

    def fcode(self):
        return "type{attrs} :: {name}\n{tags}{decls}end type {name}\n".format(
                    name=self.name,
                    attrs="".join(", " + a.fcode() for a in self.attrs),
                    tags="".join(map(fcode, self.tags)),
                    decls="".join(map(fcode, self.decls))
                    )

    def cdecl(self, config, ref=None):
        if self.cname is None:
            fail = CWrapper.fail(self.name, "bind(C) prefix missing")
        else:
            fields = CWrapper.union(self.fields, config)
            if fields.fails:
                fail = CWrapper.fail(self.name, "failed to wrap fields",
                                     fields.fails)
            elif ref is None:
                return CWrapper("struct {name} {{\n{fields}}};\n".format(
                                    name=self.cname, fields=fields.decl),
                                inherit=(fields,))
            else:
                return CWrapper("struct "+self.cname, inherit=(fields,))

        # Using opaque types
        if not config.opaque_structs:
            return CWrapper.fail(self.name, "no opaque structs", (fail,))

        opaquename = "{}{}".format(config.prefix, self.name)
        if ref is None:
            return CWrapper(opaque_structs=[opaquename])
        else:
            return CWrapper("struct " + opaquename, opaque_structs=[opaquename])


@ast_handler("extends")
class Extends(ReferenceNode):
    def imbue(self, parent):
        ReferenceNode.imbue(self, parent)
        parent.extends = self

    def cdecl(self, config):
        return CWrapper.fail(self.parent.name, "no support for extends")


@ast_handler("type_bound_procedures")
class TypeBoundProcedureList(TransparentNode):
    def __init__(self, are_private, *procs):
        TransparentNode.__init__(self, *procs)
        self.are_private = are_private

    def cdecl(self, config):
        return CWrapper.fail(None, "type-bound procedures unsupported")


class Unspecified(Node):
    def __init__(self, name):
        self.name = name

    def cdecl(self, config):
        return CWrapper.fail(None, "type unknown")


class Subprogram(NamespaceNode):
    FTYPE = '@@@'

    def __init__(self, name, prefixes, argnames, suffixes, decls):
        NamespaceNode.__init__(self, name)
        self.prefixes = prefixes
        self.suffixes = suffixes
        self.argnames = argnames
        self.decls = decls

    @property
    def children(self):
        return self.prefixes + self.suffixes + self.decls

    def imbue(self, parent):
        self.cname = None
        self.args = [Unspecified(name) for name in self.argnames]
        self.retval = Unspecified(self.name) if self.FTYPE == 'function' else None
        NamespaceNode.imbue(self, parent)

    def fcode_header(self):
        return "{prefixes}{sep}{tag} {name}({args}) {suffixes}".format(
            prefixes=" ".join(map(fcode, self.prefixes)),
            sep=" " if self.prefixes else "",
            tag=self.FTYPE,
            name=self.name,
            args=", ".join(self.argnames),
            suffixes=" ".join(map(fcode, self.suffixes))
            )

    def fcode(self):
        return "{header}\n{decls}end {tag} {name}\n".format(
            header=self.fcode_header(),
            decls="".join(map(fcode, self.decls)),
            tag=self.FTYPE,
            name=self.name
            )

    def cdecl(self, config):
        bindres = CWrapper()
        if self.cname is None:
            bindres = CWrapper.fail(self.name, "bind(C) suffix missing")

        # arg decls
        args = CWrapper.union(self.args, config, sep=", ")
        if self.retval is not None:
            # pylint: disable=no-member
            ret = self.retval.type_.cdecl(config)
        else:
            ret = CWrapper("void")

        if bindres.fails or args.fails or ret.fails:
            return CWrapper.fail(self.name, "failed to wrap",
                                 bindres.fails + args.fails + ret.fails)

        return CWrapper(
            "{ret} {name}({args});\n".format(ret=ret.decl, name=self.cname,
                                             args=args.decl),
            inherit=(ret, args)
            )


@ast_handler("subroutine_decl")
class Subroutine(Subprogram):
    FTYPE = 'subroutine'

    def __init__(self, name, prefixes, argnames, bindc, decls):
        Subprogram.__init__(self, name, prefixes, argnames,
                            (bindc,) if bindc is not None else (), decls)


@ast_handler("function_decl")
class Function(Subprogram):
    FTYPE = 'function'

    def __init__(self, name, prefixes, argnames, suffixes, decls):
        Subprogram.__init__(self, name, prefixes, argnames, suffixes, decls)


@ast_handler("entity_decl")
class EntityDecl(TransparentNode):
    def __init__(self, type_, attrs, entity_list):
        entities = tuple(Entity(type_, attrs, *e) for e in entity_list)
        TransparentNode.__init__(self, *entities)


class Entity(Node):
    def __init__(self, type_, attrs, name, shape, len_, init):
        self.type_ = type_
        self.attrs = attrs
        self.name = name
        self.shape = shape
        self.len_ = len_
        self.init = init

    def imbue(self, parent):
        # Add self to arguments
        self.parent = parent
        self.parent.namespace.add(self)
        self.namespace = parent.namespace

        self.entity_type = 'variable'
        self.intent = None
        self.storage = None
        self.target = None
        self.required = None
        self.passby = None
        self.volatile = None
        self.cname = None
        self.scope = None

        self.fqname = "{}%%{}".format(parent.fqname, self.name)
        if isinstance(parent, DerivedType):
            parent.fields.append(self)
            self.scope = 'derived_type'
            self.storage = 'fixed'
        elif isinstance(parent, Module):
            self.scope = 'module'
            self.storage = 'fixed'
        elif isinstance(parent, Subprogram):
            # Add self to arguments if applicable
            self.scope = 'subprogram'
            if self.name in parent.argnames:
                parent.args[parent.argnames.index(self.name)] = self
                self.entity_type = 'argument'
                self.intent = True, True
                self.passby = 'reference'
                self.required = True
            elif parent.retval is not None and self.name == parent.retval.name:
                parent.retval = self
                self.entity_type = 'return'
                self.passby = 'value'
            else:
                self.storage = 'fixed'

        for attr in self.attrs:
            attr.imbue(self)
        self.type_.imbue(self)

    def resolve(self):
        self.type_.resolve()

    def fcode(self):
        return "{type}{attrs} :: {name}{shape}{len} {init}\n".format(
            type=fcode(self.type_),
            attrs="".join(", " + fcode(a) for a in self.attrs),
            name=self.name,
            shape=fcode(self.shape) if self.shape is not None else "",
            len=fcode(self.len_) if self.len_ is not None else "",
            init=fcode(self.init) if self.len_ is not None else ""
            )

    def cdecl(self, config):
        if self.scope == 'module':
            if self.cname is None:
                return CWrapper.fail(self.name,
                                     "bind(C) missing on module variable")
            decl = "extern {const}{type_}{ptr}{name}{shape};\n"
        elif self.scope == 'derived_type':
            decl = "  {const}{type_}{ptr}{name}{shape};\n"
        else:
            decl = "{const}{type_}{ptr}{name}{shape}"

        type_ = self.type_.cdecl(config)
        if type_.fails:
            return CWrapper.fail(self.name, "cannot wrap entity type",
                                 type_.fails)

        type_.decl += "" if type_.decl.endswith("*") else " "
        pointer = (self.len_ is not None or
                   self.passby == 'reference' and self.shape is None)
        const = (self.entity_type == 'parameter' or
                 self.entity_type == 'argument' and not self.intent[1])
        shape = self.shape.cdecl(config) if self.shape is not None else CWrapper("")

        return CWrapper(decl.format(
                            const="const " * const, type_=type_.decl,
                            ptr="*" * pointer, name=self.name, shape=shape.decl
                            ),
                        inherit=(type_, shape))


class Attribute(Node):
    def resolve(self): pass


@ast_handler("intent")
class Intent(Attribute):
    def __init__(self, in_, out):
        self.in_ = bool(in_)
        self.out = bool(out)

    def imbue(self, parent):
        parent.intent = self.in_, self.out

    def fcode(self):
        return "intent({}{})".format("in" * self.in_, "out" * self.out)


@ast_handler("bind_c")
class BindC(Attribute):
    def __init__(self, cname):
        self.cname = None if cname is None else cname.value

    def imbue(self, parent):
        if self.cname is None:
            self.cname = parent.name
        parent.cname = self.cname

    def fcode(self):
        if self.cname is None:
            return "bind(C)"
        else:
            return "bind(C, name='{}')".format(self.cname)


@ast_handler("dimension")
class DimensionAttr(Attribute):
    def __init__(self, shape):
        self.shape = shape

    def imbue(self, parent):
        if parent.shape is not None:
            raise RuntimeError("duplicate shape")
        parent.shape = self.shape
        parent.attrs = tuple(a for a in parent.attrs if a is not self)


@ast_handler("result")
class ResultName(Attribute):
    def __init__(self, name):
        self.name = name

    def imbue(self, parent):
        parent.retval.name = self.name

    def fcode(self):
        return "result({})".format(self.name)


@ast_handler("value")
class PassByValue(Attribute):
    def imbue(self, parent): parent.passby = 'value'
    def fcode(self): return "value"


@ast_handler("parameter")
class ParameterAttr(Attribute):
    def imbue(self, parent): parent.entity_type = 'parameter'
    def fcode(self): return "parameter"


@ast_handler("optional")
class OptionalAttr(Attribute):
    def imbue(self, parent): parent.required = False
    def fcode(self): return "optional"


@ast_handler("pointer")
class PointerAttr(Attribute):
    def imbue(self, parent): parent.storage = 'pointer'
    def fcode(self): return "pointer"


@ast_handler("target")
class TargetAttr(Attribute):
    def imbue(self, parent): parent.target = True
    def fcode(self): return "target"


@ast_handler("allocatable")
class AllocatableAttr(Attribute):
    def imbue(self, parent): parent.storage = 'allocatable'
    def fcode(self): return "allocatable"


@ast_handler("volatile")
class VolatileAttr(Attribute):
    def imbue(self, parent): parent.volatile = True
    def fcode(self): return "volatile"


class Opaque:
    def __init__(self, module, name):
        self.module = module
        self.name = name
        self.fqname = "%%" + self.name

    def fcode(self): return self.fqname


class CPtrType:
    def __init__(self, module, name):
        self.module = module
        self.name = name

    def fcode(self):
        return "type(%%{})".format(self.name)

    def cdecl(self, config, decl=False):
        if decl:
            raise RuntimeError("No C declaration for this type")
        return CWrapper("void *")


class IsoCBindingModule:
    NAME = "iso_c_binding"
    TAGS = (
        'c_int', 'c_short', 'c_long', 'c_long_long', 'c_signed_char',
        'c_size_t', 'c_int8_t', 'c_int16_t', 'c_int32_t', 'c_int64_t',
        'c_int_least8_t', 'c_int_least16_t', 'c_int_least32_t',
        'c_int_least64_t', 'c_int_fast8_t', 'c_int_fast16_t', 'c_int_fast32_t',
        'c_int_fast64_t', 'c_intmax_t', 'c_intptr_t', 'c_ptrdiff_t', 'c_float',
        'c_double', 'c_long_double', 'c_float128', 'c_float_complex',
        'c_double_complex', 'c_long_double_complex', 'c_bool', 'c_char'
        )
    TYPES = 'c_ptr', 'c_funptr'

    def __init__(self):
        values = {tag: Opaque(self, tag) for tag in self.TAGS}
        values.update({name: CPtrType(self, name) for name in self.TYPES})
        self.namespace = Namespace(values)

    def resolve(self): pass

    @property
    def name(self): return self.NAME


class OpaqueModule:
    def __init__(self, name):
        self.name = name
        self.namespace = Namespace()

    def resolve(self): pass


@ast_handler("ref")
class Ref(ReferenceNode):
    def __init__(self, name):
        ReferenceNode.__init__(self, name.lower())


class PrimitiveType(Node):
    FTYPE = '$$$null_type$$$'
    KIND_MAPS = {}

    def __init__(self, kind):
        self.kind = kind

    def imbue(self, parent):
        self.parent = parent
        self.namespace = parent.namespace
        # A type may imbue a function as its return type.
        if isinstance(parent, Function):
            retval = Entity(self, (), parent.name, None, None, None)
            retval.imbue(parent)
        if self.kind is not None:
            self.kind.imbue(self)

    def resolve(self):
        if self.kind is not None:
            self.kind.resolve()

    def fcode(self):
        if self.kind is None:
            return self.FTYPE
        return "{ftype}(kind={kind})".format(ftype=self.FTYPE,
                                             kind=fcode(self.kind))

    def cdecl(self, config):
        if self.kind is None:
            fail = CWrapper.fail(self.fcode(), "no kind associated")
        else:
            try:
                ctype, cheaders = self.KIND_MAPS[self.kind.fcode()][:2]
            except KeyError:
                fail = CWrapper.fail(self.fcode(), "kind has no iso_c_binding")
            else:
                return CWrapper(ctype, headers=cheaders)

        # We have failed to wrap directly
        if config.opaque_scalars:
            if self.kind is None:
                kindstr = ""
            elif isinstance(self.kind, Literal):
                kindstr = str(self.kind.value)
            else:
                return CWrapper.fail(self.fcode(), "kind cannot be transformed to string")

            # FIXME - opaques do not work on return values/entities
            opaquename = "f{}{}".format(self.FTYPE, kindstr)
            return CWrapper(opaquename, opaque_scalars=set([opaquename]))
        else:
            return fail



@ast_handler("integer_type")
class IntegerType(PrimitiveType):
    FTYPE = 'integer'

    _STDDEF = {'<stddef.h>'}
    _STDINT = {'<stdint.h>'}
    _NONE = set()
    KIND_MAPS = {
        '%%c_int':         ('int',         _NONE,   'intc',     'c_int'),
        '%%c_short':       ('short',       _NONE,   'short',    'c_short'),
        '%%c_long':        ('long',        _NONE,   'int_',     'c_long'),
        '%%c_long_long':   ('long long',   _NONE,   'longlong', 'c_longlong'),
        '%%c_signed_char': ('signed char', _NONE,   'byte',     'c_byte'),
        '%%c_size_t':      ('size_t',      _STDDEF, 'intp',     'c_ssize_t'),
        '%%c_int8_t':      ('int8_t',      _STDINT, 'int8',     'c_int8'),
        '%%c_int16_t':     ('int16_t',     _STDINT, 'int16',    'c_int16'),
        '%%c_int32_t':     ('int32_t',     _STDINT, 'int32',    'c_int32'),
        '%%c_int64_t':     ('int64_t',     _STDINT, 'int64',    'c_int64'),
        '%%c_intptr_t':    ('intptr_t',    _STDINT, 'intp',     'c_ssize_t'),
        '%%c_ptrdiff_t':   ('ptrdiff_t',   _STDINT, 'intp',     'c_ssize_t'),
        }
    KIND_MAPS.update({
        '1': KIND_MAPS['%%c_int8_t'],
        '2': KIND_MAPS['%%c_int16_t'],
        '4': KIND_MAPS['%%c_int32_t'],
        '8': KIND_MAPS['%%c_int64_t']
        })


@ast_handler("real_type")
class RealType(PrimitiveType):
    FTYPE = 'real'

    _NONE = set()
    KIND_MAPS = {
        '%%c_float':       ('float',       _NONE, 'float32', 'c_float'),
        '%%c_double':      ('double',      _NONE, 'float64', 'c_double'),
        '%%c_long_double': ('long double', _NONE, 'longdouble', 'c_longdouble')
        }
    KIND_MAPS.update({
        '4': KIND_MAPS['%%c_float'],
        '8': KIND_MAPS['%%c_double']
        })


@ast_handler("complex_type")
class ComplexType(PrimitiveType):
    FTYPE = 'complex'

    _NONE = set()
    KIND_MAPS = {
        '%%c_float_complex': ('float _Complex', _NONE, 'complex64', 'c_float'),
        '%%c_double_complex': ('double _Complex', _NONE, 'complex128', 'c_double'),
        '%%c_long_double_complex':
                ('long double _Complex', _NONE, 'clongdouble', 'c_longdouble'),
        }
    KIND_MAPS.update({
        '8': KIND_MAPS['%%c_float_complex'],
        '16': KIND_MAPS['%%c_double_complex']
        })


@ast_handler("logical_type")
class LogicalType(PrimitiveType):
    FTYPE = 'logical'
    KIND_MAPS = {
        '%%c_bool': ('_Bool', set(), 'bool_', 'c_bool'),
        }
    KIND_MAPS['1'] = KIND_MAPS['%%c_bool']

    def cdecl(self, config):
        if self.kind is None:
            return CWrapper("_Bool")
        return PrimitiveType.cdecl(self, config)


@ast_handler("character_type")
class CharacterType:
    KIND_MAPS = {
        '%%c_char': ('char', set(), 'char', 'c_char'),
        }
    KIND_MAPS['1'] = KIND_MAPS['%%c_char']

    def __init__(self, char_sel):
        if char_sel is None:
            char_sel = None, None
        len_, kind = char_sel
        self.len_ = len_
        self.kind = kind

    def imbue(self, parent):
        self.parent = parent
        self.namespace = parent.namespace
        # TODO: imbue length onto entities
        if isinstance(parent, Function):
            retval = Entity(self, (), parent.name, None, None, None)
            retval.imbue(parent)
        if isinstance(self.len_, Node):
            self.len_.imbue(self)
        if isinstance(self.kind, Node):
            self.kind.imbue(self)

    def resolve(self):
        if isinstance(self.len_, Node):
            self.len_.resolve()
        if isinstance(self.kind, Node):
            self.kind.resolve()

    def fcode(self):
        if self.kind is None and self.len_ is None:
            return "character"
        return "character({})".format(
            ", ".join("{}={}".format(k, v if isinstance(v,str) else v.fcode())
                      for k, v in {'len':self.len_, 'kind':self.kind}.items()
                      if v is not None))

    def cdecl(self, config):
        if self.kind is None:
            return CWrapper("char")
        ctype, cheaders = self.KIND_MAPS[self.kind.fcode()][:2]
        return CWrapper(ctype, headers=cheaders)


class Literal(Node):
    @classmethod
    def parse(cls, token): raise NotImplementedError()

    def __init__(self, token):
        self.token = token
        self.value = self.parse(token)
        self.kind = None

    def imbue(self, parent): pass

    def resolve(self): pass

    def fcode(self): return self.token


@ast_handler("int")
class IntLiteral(Literal):
    @classmethod
    def parse(cls, token):
        return int(token)


@ast_handler("string")
class StringLiteral(Literal):
    @classmethod
    def parse(cls, token):
        return lexer.parse_string(token)


class Dim(Node):
    def __init__(self, lower, upper):
        if lower is None:
            lower = IntLiteral("1")
        self.lower = lower
        self.upper = upper

    def imbue(self, parent):
        self.parent = parent
        self.namespace = parent.namespace
        if isinstance(self.lower, Node):
            self.lower.imbue(self)
        if isinstance(self.upper, Node):
            self.lower.imbue(self)

    def resolve(self):
        if isinstance(self.lower, Node):
            self.lower.resolve()
        if isinstance(self.upper, Node):
            self.lower.resolve()


@ast_handler("explicit_dim")
class ExplicitDim(Dim):
    def imbue(self, parent):
        Dim.imbue(self, parent)
        if parent.shape_type == 'implied':
            raise RuntimeError("Cannot follow implied with explicit dim")

    def fcode(self):
        return "{}:{}".format(self.lower.fcode(), self.upper.fcode())

    def cdecl(self, config):
        try:
            size = self.upper.value - self.lower.value
        except AttributeError:
            return CWrapper.fail(self.fcode(), "Shape is not constant")
        return CWrapper("[{}]".format(size))


@ast_handler("implied_dim")
class ImpliedDim(Dim):
    def imbue(self, parent):
        Dim.imbue(self, parent)
        if parent.shape_type != 'explicit':
            raise RuntimeError("Cannot follow non-explicit with implied dim")
        parent.shape_type = 'implied'

    def fcode(self):
        return "{}:*".format(self.lower.fcode())

    def cdecl(self, config):
        return CWrapper("[]")


@ast_handler("deferred_dim")
class DeferredDim(Dim):
    def imbue(self, parent):
        Dim.imbue(self, parent)
        if parent.shape_type == 'implied':
            raise RuntimeError("Cannot follow implied with explicit dim")
        parent.shape_type = 'deferred'

    def fcode(self):
        return "{}:".format(self.lower.fcode())

    def cdecl(self, config):
        # Deferred dims cause variables to carry its dimension parameters with
        # them in a non-bind(C) way
        return CWrapper.fail(self.fcode(), "unable to wrap deferred dimension")


@ast_handler("shape")
class Shape(Node):
    def __init__(self, *dims):
        self.dims = dims

    def imbue(self, parent):
        self.parent = parent
        self.namespace = parent.namespace
        self.shape_type = 'explicit'
        for dim in self.dims:
            dim.imbue(self)

    def resolve(self):
        for dim in self.dims:
            dim.resolve()

    def fcode(self):
        return "({})".format(",".join(map(fcode, self.dims)))

    def cdecl(self, config):
        return CWrapper.union(self.dims, config)


@ast_handler("derived_type")
class DerivedTypeRef(Node):
    def __init__(self, name):
        self.name = name

    def imbue(self, parent):
        self.parent = parent
        self.namespace = parent.namespace

        # A type may imbue a function as its return type.
        # TODO: duplication with PrimitiveType, merge? Also, somewhat hacky
        if isinstance(parent, Function):
            retval = Entity(self, (), parent.name, None, None, None)
            retval.imbue(parent)

    def resolve(self):
        try:
            self.ref = self.namespace.get(self.name)
        except KeyError:
            warnings.warn("DID NOT FIND TYPE %s" % self.name)
            self.ref = None

    def fcode(self):
        if self.ref is not None:
            return self.ref.fcode()
        return "type({})".format(self.name)

    def cdecl(self, config):
        if self.ref is None:
            return CWrapper.fail(self.name, "type declaration not found")
        return self.ref.cdecl(config, False)


@ast_handler("preproc_stmt")
class PreprocStmt(Node):
    def __init__(self, stmt):
        self.stmt = stmt

    def imbue(self, parent): pass

    def resolve(self): pass

    def fcode(self): return self.stmt

    def cdecl(self, config): return CWrapper(self.stmt)


def unpack(arg):
    """Unpack a single argument as-is"""
    return arg


def unpack_sequence(*items):
    """Return sequence of itmes as a tuple"""
    return items


_HANDLER_REGISTRY.update({
    'filename':          unpack,
    'ast_version':       unpack_sequence,
    'declaration_block': unpack_sequence,
    'contained_block':   unpack_sequence,
    'type_attrs':        unpack_sequence,
    'type_tags':         unpack_sequence,
    'component_block':   unpack_sequence,
    'arg_list':          unpack_sequence,
    'sub_prefix_list':   unpack_sequence,
    'func_prefix_list':  unpack_sequence,
    'func_suffix_list':  unpack_sequence,
    'entity_attrs':      unpack_sequence,
    'entity_list':       unpack_sequence,
    'entity':            unpack_sequence,
    'kind_sel':          unpack,
    'char_sel':          unpack_sequence,
    'id':                lambda name: name.lower(),
    })


def sexpr_transformer(branch_map, fallback=None):
    """Return depth-first transformation of an AST as S-expression.

    An S-expression is either a branch or a leaf.  A leaf is an arbitrary
    string or `None`.  A branch is a tuple `(tag, *tail)`, where the `tag` is
    a string and each item of `tail` is again an S-expression.

    Construct and return a transformer which does the following: for a
    branch, we look up the tag in `branch_map`.  If it is found, the
    corresponding entry is called with tag and tail as its arguments, but we
    first run the tail through the transformer.  Any leaf is simply returned
    as-is.

    If the tag is not found in branch_map, we instead call `fallback`. In this
    case, the arguments are not processed, which allows pruning subtrees not
    interesting to the consumer.
    """
    def default_fallback(tag, *tail):
        raise ValueError("unexpected tag: {}".format(tag))
    if fallback is None:
        fallback = default_fallback

    def transformer(ast):
        if isinstance(ast, tuple):
            # Branch node
            node_type = ast[0]
            try:
                handler = branch_map[node_type]
            except KeyError:
                return fallback(*ast)
            else:
                return handler(*map(transformer, ast[1:]))
        else:
            # Leaf node
            return ast

    return transformer


TRANSFORMER = sexpr_transformer(_HANDLER_REGISTRY, Ignored)
