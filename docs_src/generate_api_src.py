import inspect
import os
import shutil
from importlib import import_module
from os import path
from typing import Optional

from jinja2 import Environment, FileSystemLoader


def dot_path(module: list) -> str:
    return '.'.join(module)


def rel_path(parents: list, name: str):
    return f"{dot_path(parents + [name])}"


def title(name: str):
    return f"{name}\n{'=' * len(name)}"


class Generator:
    def __init__(self, templates, target):
        self.environment = Environment(loader=FileSystemLoader(templates))
        self.target = target

    def generate(self, parents, name, template, bindings):
        t = self.environment.get_template(template)
        c = t.render(**bindings)
        with open(f"{self.target}/{rel_path(parents, name)}.rst", 'w') as out:
            out.write(c)


class DocNode:
    def __init__(self, name: str, obj: object):
        self.name = name
        self.obj = obj

    def docstring(self):
        return self.obj.__doc__

    def uid(self) -> str:
        pass

    def get_short_description(self) -> Optional[str]:
        short = self.docstring()
        if short:
            short = short.split(".")[0].strip()
            short = short.split("\n")[0].strip()
        return short

    def __repr__(self):
        return f"{self.name} {self.get_short_description()}"

    def generate(self, g: Generator, parents: list):
        pass


class FunctionDocNode(DocNode):
    def __init__(self, name: str, obj):
        super().__init__(name, obj)

    def generate(self, g: Generator, parents: list):
        bindings = {
            "module": dot_path(parents),
            "name": self.name,
            "title": title(self.name)
        }
        g.generate(parents, self.name, "function.rst", bindings)

    def uid(self) -> str:
        return self.obj.__module__ + "." + self.obj.__qualname__


class MethodDocNode(DocNode):
    def __init__(self, name: str, obj, uid : str):
        super().__init__(name, obj)
        self._uid = uid

    def docstring(self):
        return inspect.getdoc(self.obj)

    def generate(self, g: Generator, parents: list):
        bindings = {
            "module": dot_path(parents[:-1]),
            "class": parents[-1],
            "name": self.name,
            "title": title(self.name)
        }
        g.generate(parents, self.name, "method.rst", bindings)

    def uid(self) -> str:
        return self._uid


class PropertyDocNode(DocNode):
    def __init__(self, name: str, obj, uid : str):
        super().__init__(name, obj)
        self._uid = uid

    def docstring(self):
        return inspect.getdoc(self.obj)

    def generate(self, g: Generator, parents: list):
        bindings = {
            "module": dot_path(parents[:-1]),
            "class": parents[-1],
            "name": self.name,
            "title": title(self.name)
        }
        g.generate(parents, self.name, "property.rst", bindings)

    def uid(self) -> str:
        return self._uid


class ClassDocNode(DocNode):
    def __init__(self, name: str, obj: object, methods: list[DocNode], properties: list[DocNode]):
        super().__init__(name, obj)
        self.methods = methods
        self.properties = properties

    def generate(self, g: Generator, parents: list):
        for property in self.properties:
            property.generate(g, parents + [self.name])

        for method in self.methods:
            method.generate(g, parents + [self.name])

        bindings = {
            "module": dot_path(parents),
            "name": self.name,
            "properties": [
                (rel_path(parents + [self.name], prop.name), prop.name, prop.get_short_description()) for prop in self.properties
            ],
            "methods": [
                (rel_path(parents + [self.name], method.name), method.name, method.get_short_description()) for method in self.methods
            ],
            "title": title(self.name)
        }
        g.generate(parents, self.name, "class.rst", bindings)

    def uid(self) -> str:
        return self.obj.__module__ + "." + self.name


class ModuleDocNode(DocNode):
    def __init__(self, name: str, obj: object, sub_mods: list[DocNode], classes: list[DocNode], functions: list[DocNode]):
        super().__init__(name, obj)
        self.sub_mods = sub_mods
        self.classes = classes
        self.functions = functions

    def generate(self, g: Generator, parents: list):
        module = parents + [self.name]
        for sub_mod in self.sub_mods:
            sub_mod.generate(g, module)
        for cls in self.classes:
            cls.generate(g, module)
        for func in self.functions:
            func.generate(g, module)

        bindings = {
            "module": dot_path(module),
            "sub_mods": [
                (rel_path(module, sub_mod.name), sub_mod.name, sub_mod.get_short_description()) for sub_mod in self.sub_mods
            ],
            "classes": [
                (rel_path(module, cls.name), cls.name, cls.get_short_description()) for cls in self.classes
            ],
            "functions": [
                (rel_path(module, func.name), func.name, func.get_short_description()) for func in self.functions
            ],
            "title": title(dot_path(module))
        }
        g.generate(parents, self.name, "module.rst", bindings)

    def uid(self) -> str:
        return self.obj.__name__


class Pruner:
    def __init__(self):
        self.registry = set()

    def _not_pruned(self, node: DocNode | None):
        if node is None:
            return False
        _id = node.uid()
        if _id not in self.registry:
            self.registry.add(_id)
            return True
        return False

    def prune(self, nodes):
        return [n for n in nodes if self._not_pruned(n)]


def process_function(name, func):
    if name.startswith("_"):
        return None
    if inspect.getdoc(func) is None:
        return None
    return FunctionDocNode(name, func)


def process_method(name, method, uid):
    if name.startswith("_") and name not in ["__getitem__"]:
        return None
    if inspect.getdoc(method) is None:
        return None
    return MethodDocNode(name, method, uid)


def process_property(name, prop, uid):
    if name.startswith("_") and name not in ["__getitem__"]:
        return None
    if inspect.getdoc(prop) is None:
        return None
    return PropertyDocNode(name, prop, uid)


def process_class(pruner, name, cls):
    if name.startswith("_"):
        return None
    methods = pruner.prune([process_method(name, func, ".".join([cls.__module__, cls.__name__, name])) for name, func in sorted(inspect.getmembers(cls, lambda x: inspect.ismethod(x) or inspect.isfunction(x)))])
    properties = pruner.prune([process_property(name, prop, ".".join([cls.__module__, cls.__name__, name])) for name, prop in sorted(inspect.getmembers(cls, lambda x: isinstance(x, property)))])
    return ClassDocNode(name, cls, methods, properties)


def process_module(pruner, name, mod):
    if name.startswith("_"):
        return None

    classes = pruner.prune([process_class(pruner, name, cls) for name, cls in sorted(inspect.getmembers(mod, predicate=inspect.isclass)) if
                            cls.__module__.startswith(mod.__name__)])
    functions = pruner.prune([process_function(name, func) for name, func in sorted(inspect.getmembers(mod, predicate=inspect.isfunction)) if
                              func.__module__.startswith(mod.__name__)])
    sub_mods = pruner.prune([process_module(pruner, name, sub_mod) for name, sub_mod in sorted(inspect.getmembers(mod, predicate=inspect.ismodule)) if
                             sub_mod.__name__.startswith(mod.__name__)])

    if sub_mods or classes or functions:
        return ModuleDocNode(name, mod, sub_mods, classes, functions)


def list_modules(base, current):
    current_abs = path.join(base, current)
    if path.isfile(current_abs) and current_abs.endswith(".py"):
        return [[current[:-3]]]
    return [[current], *[
        [current, *m]
        for f in sorted(os.listdir(current_abs))
        if not (f.startswith("__") or f.startswith("tests") or f.startswith(".")) # don't parse hidden files..
        for m in list_modules(path.join(base, current), f)
    ]]


def generate_tangle_reference():
    tangles_root = path.abspath(path.join(__file__, "..", ".."))
    target = path.join(tangles_root, "docs_src", "source", "reference", "api")
    if path.isdir(target):
        print(f"Cleaning {target}")
        shutil.rmtree(target)
    os.mkdir(target)

    modules = {m: import_module(m) for m in [dot_path(mm) for mm in list_modules(tangles_root, "tangles")]}

    p = Pruner()
    doc = process_module(p, "tangles", modules["tangles"])
    g = Generator("reference_templates", target)
    doc.generate(g, [])


if __name__ == "__main__":
    generate_tangle_reference()
