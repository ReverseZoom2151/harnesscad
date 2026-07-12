"""Editability operations on OpenECAD scripts (Yuan, Shi & Huang, 2024).

The paper's headline claim is that OpenECAD code is *editable*: a human (or tool)
can locate a named variable and change it, then re-emit valid code. Sec. 6.5
demonstrates exactly this -- "We changed the ``extent_one`` of the table legs
extrusion from 1000 to 500" -- and Table 6 notes size and position errors are
"relatively easily resolved by post-editing the extent size ... or the size of
curves ... in the CAD code".

This module implements those deterministic edits over
:mod:`programs.openecad_script` programs, always returning a *new* program so the
original is untouched:

* :func:`rename_variable` -- rename a named variable consistently across its
  definition and every reference;
* :func:`set_keyword` -- reparametrize a command by setting a keyword argument
  (e.g. change ``extent_one``);
* :func:`reparametrize` -- apply several keyword edits at once.

No learning is involved; this is the mechanical editing the paper relies on.
"""

from __future__ import annotations

from programs import openecad_script as oe


def variable_names(program: oe.Program) -> set[str]:
    """All variable names assigned anywhere in *program*."""
    names: set[str] = set()
    for st in program.statements:
        names.update(st.targets)
    return names


def _rename_in_value(value: object, old: str, new: str) -> object:
    if isinstance(value, oe.Ref):
        return oe.Ref(new) if value.name == old else value
    if isinstance(value, oe.Call):
        return oe.Call(value.func, tuple(
            oe.Arg(_rename_in_value(a.value, old, new), a.key) for a in value.args))
    if isinstance(value, list):
        return [_rename_in_value(v, old, new) for v in value]
    if isinstance(value, tuple):
        return tuple(_rename_in_value(v, old, new) for v in value)
    return value


def count_references(program: oe.Program, name: str) -> int:
    """How many times *name* is used as a variable reference (not as a target)."""
    count = 0

    def walk(value: object) -> None:
        nonlocal count
        if isinstance(value, oe.Ref):
            if value.name == name:
                count += 1
        elif isinstance(value, oe.Call):
            for a in value.args:
                walk(a.value)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)

    for st in program.statements:
        for v in st.values:
            walk(v)
    return count


def rename_variable(program: oe.Program, old: str, new: str) -> oe.Program:
    """Return a copy of *program* with variable *old* renamed to *new*.

    Renames both the definition (target) and every reference. Raises
    :class:`ValueError` if *old* is undefined or *new* already names a variable.
    """
    if not new.isidentifier():
        raise ValueError(f"invalid new variable name: {new!r}")
    names = variable_names(program)
    if old not in names:
        raise ValueError(f"variable {old!r} is not defined")
    if new in names:
        raise ValueError(f"variable {new!r} already exists")

    new_statements = []
    for st in program.statements:
        targets = tuple(new if t == old else t for t in st.targets)
        values = tuple(_rename_in_value(v, old, new) for v in st.values)
        new_statements.append(oe.Assign(targets, values))
    return oe.Program(new_statements)


def _find_command_index(program: oe.Program, target: str) -> int:
    for i, st in enumerate(program.statements):
        if st.targets == (target,) and st.call is not None:
            return i
    raise ValueError(f"no command assigned to variable {target!r}")


def set_keyword(program: oe.Program, target: str, key: str,
                value: object) -> oe.Program:
    """Return a copy of *program* with ``target``'s call keyword *key* set.

    Replaces the argument in place if *key* is already present, otherwise appends
    it. Raises if *target* is not a command assignment.
    """
    idx = _find_command_index(program, target)
    call = program.statements[idx].call
    new_args = list(call.args)
    for j, a in enumerate(new_args):
        if a.key == key:
            new_args[j] = oe.Arg(value, key)
            break
    else:
        new_args.append(oe.Arg(value, key))
    new_call = oe.Call(call.func, tuple(new_args))
    new_statements = list(program.statements)
    new_statements[idx] = oe.Assign((target,), (new_call,))
    return oe.Program(new_statements)


def reparametrize(program: oe.Program, target: str,
                  **edits: object) -> oe.Program:
    """Apply several keyword edits to ``target``'s call at once."""
    out = program
    for key, value in edits.items():
        out = set_keyword(out, target, key, value)
    return out
