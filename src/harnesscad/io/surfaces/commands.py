"""Keyboard-first, shell-free command grammar for HarnessCAD."""

from __future__ import annotations

import difflib
import shlex
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from harnesscad.core.cisp.ops import (
    AddCircle,
    AddInstance,
    AddLine,
    AddRectangle,
    Extrude,
    Fillet,
    Hole,
    Mate,
    NewSketch,
    Op,
    SetParam,
)


class Mode(str, Enum):
    SKETCH = "sketch"
    FEATURE = "feature"
    ASSEMBLY = "assembly"
    QUERY = "query"


@dataclass(frozen=True)
class CommandState:
    mode: Mode = Mode.SKETCH


@dataclass(frozen=True)
class OpIntent:
    op: Op


@dataclass(frozen=True)
class QueryIntent:
    query: str


@dataclass(frozen=True)
class UndoIntent:
    count: int = 1


@dataclass(frozen=True)
class ModeIntent:
    mode: Mode


@dataclass(frozen=True)
class HelpIntent:
    text: str


Intent = Union[OpIntent, QueryIntent, UndoIntent, ModeIntent, HelpIntent]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    modes: tuple[Mode, ...]
    usage: str
    description: str


class CommandParseError(ValueError):
    def __init__(
        self, message: str, *, token: Optional[str] = None,
        suggestions: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.message = message
        self.token = token
        self.suggestions = suggestions

    def accessible_message(self) -> str:
        suffix = (
            f" Did you mean: {', '.join(self.suggestions)}?"
            if self.suggestions else ""
        )
        return self.message + suffix


SPECS = (
    CommandSpec("new-sketch", (Mode.SKETCH,), "new-sketch [XY|XZ|YZ]",
                "Create a sketch on a principal plane."),
    CommandSpec("circle", (Mode.SKETCH,), "circle SKETCH CX CY R",
                "Add a circle to a sketch."),
    CommandSpec("line", (Mode.SKETCH,), "line SKETCH X1 Y1 X2 Y2",
                "Add a line segment to a sketch."),
    CommandSpec("rectangle", (Mode.SKETCH,), "rectangle SKETCH X Y W H",
                "Add a rectangle to a sketch."),
    CommandSpec("extrude", (Mode.FEATURE,), "extrude SKETCH DISTANCE",
                "Extrude a sketch profile."),
    CommandSpec("fillet", (Mode.FEATURE,), "fillet RADIUS [EDGE ...]",
                "Round selected solid edges."),
    CommandSpec("hole", (Mode.FEATURE,), "hole FACE X Y DIAMETER [DEPTH]",
                "Create a through or blind semantic hole."),
    CommandSpec("set-param", (Mode.FEATURE,), "set-param INDEX PARAM VALUE",
                "Edit a prior operation parameter."),
    CommandSpec("instance", (Mode.ASSEMBLY,), "instance PART [X Y Z]",
                "Place a part instance."),
    CommandSpec("mate", (Mode.ASSEMBLY,), "mate KIND INSTANCE_A INSTANCE_B",
                "Constrain two assembly instances."),
    CommandSpec("query", (Mode.QUERY,), "query NAME",
                "Request a read-only backend query."),
)

_GLOBAL = ("mode", "help", "commands", "undo")
_BY_NAME = {spec.name: spec for spec in SPECS}


class CommandSurface:
    """Stateful parser. It returns intents and never executes them."""

    def __init__(self, state: CommandState = CommandState()) -> None:
        self.state = state

    def parse(self, text: str) -> Intent:
        try:
            tokens = shlex.split(text)
        except ValueError as exc:
            raise CommandParseError(f"Could not parse command: {exc}") from exc
        if not tokens:
            raise CommandParseError("Enter a command. Type 'help' for available commands.")

        name, args = tokens[0].lower().replace("_", "-"), tokens[1:]
        if name in ("help", "commands"):
            if len(args) > 1:
                raise CommandParseError("Usage: help [COMMAND]")
            return HelpIntent(self.help(args[0] if args else None))
        if name == "mode":
            _arity(name, args, 1)
            try:
                mode = Mode(args[0].lower())
            except ValueError as exc:
                choices = tuple(mode.value for mode in Mode)
                raise CommandParseError(
                    f"Unknown mode {args[0]!r}. Available modes: {', '.join(choices)}",
                    token=args[0], suggestions=_suggest(args[0], choices),
                ) from exc
            self.state = CommandState(mode)
            return ModeIntent(mode)
        if name == "undo":
            if len(args) > 1:
                raise CommandParseError("Usage: undo [COUNT]")
            count = _integer(args[0], "COUNT") if args else 1
            if count < 1:
                raise CommandParseError("COUNT must be at least 1")
            return UndoIntent(count)

        available = self.available_commands()
        if name not in available:
            all_names = tuple(sorted(set(available) | set(_GLOBAL)))
            if name in _BY_NAME:
                required = ", ".join(mode.value for mode in _BY_NAME[name].modes)
                raise CommandParseError(
                    f"Command {name!r} is unavailable in {self.state.mode.value} mode; "
                    f"switch to {required} mode.",
                    token=name,
                )
            raise CommandParseError(
                f"Unknown command {name!r} in {self.state.mode.value} mode.",
                token=name, suggestions=_suggest(name, all_names),
            )
        return self._parse_op_or_query(name, args)

    def available_commands(self) -> tuple[str, ...]:
        return tuple(
            spec.name for spec in SPECS if self.state.mode in spec.modes
        )

    def help(self, command: Optional[str] = None) -> str:
        if command:
            name = command.lower().replace("_", "-")
            if name not in _BY_NAME:
                raise CommandParseError(
                    f"No help for unknown command {command!r}.",
                    token=command,
                    suggestions=_suggest(name, tuple(_BY_NAME)),
                )
            spec = _BY_NAME[name]
            return f"{spec.usage}\n{spec.description}"
        lines = [
            f"Current mode: {self.state.mode.value}.",
            "Global: mode MODE; undo [COUNT]; help [COMMAND].",
            "Available commands:",
        ]
        lines.extend(
            f"- {spec.usage}: {spec.description}"
            for spec in SPECS if self.state.mode in spec.modes
        )
        return "\n".join(lines)

    def _parse_op_or_query(self, name: str, args: list[str]) -> Intent:
        if name == "new-sketch":
            if len(args) > 1:
                raise CommandParseError("Usage: new-sketch [XY|XZ|YZ]")
            plane = args[0].upper() if args else "XY"
            if plane not in ("XY", "XZ", "YZ"):
                raise CommandParseError("Plane must be XY, XZ, or YZ")
            return OpIntent(NewSketch(plane))
        if name == "circle":
            _arity(name, args, 4)
            return OpIntent(AddCircle(args[0], *(_number(v) for v in args[1:])))
        if name == "line":
            _arity(name, args, 5)
            return OpIntent(AddLine(args[0], *(_number(v) for v in args[1:])))
        if name == "rectangle":
            _arity(name, args, 5)
            return OpIntent(AddRectangle(args[0], *(_number(v) for v in args[1:])))
        if name == "extrude":
            _arity(name, args, 2)
            return OpIntent(Extrude(args[0], _number(args[1])))
        if name == "fillet":
            if not args:
                raise CommandParseError("Usage: fillet RADIUS [EDGE ...]")
            return OpIntent(Fillet(tuple(args[1:]), _number(args[0])))
        if name == "hole":
            if len(args) not in (4, 5):
                raise CommandParseError("Usage: hole FACE X Y DIAMETER [DEPTH]")
            depth = _number(args[4]) if len(args) == 5 else None
            return OpIntent(Hole(
                args[0], _number(args[1]), _number(args[2]), _number(args[3]),
                depth, depth is None,
            ))
        if name == "set-param":
            _arity(name, args, 3)
            return OpIntent(SetParam(_integer(args[0], "INDEX"), args[1], _value(args[2])))
        if name == "instance":
            if len(args) not in (1, 4):
                raise CommandParseError("Usage: instance PART [X Y Z]")
            xyz = tuple(_number(v) for v in args[1:]) if len(args) == 4 else (0., 0., 0.)
            return OpIntent(AddInstance(args[0], *xyz))
        if name == "mate":
            _arity(name, args, 3)
            return OpIntent(Mate(args[0], args[1], args[2]))
        if name == "query":
            _arity(name, args, 1)
            return QueryIntent(args[0])
        raise AssertionError(name)


def _arity(name: str, args: list[str], count: int) -> None:
    if len(args) != count:
        usage = _BY_NAME[name].usage if name in _BY_NAME else f"{name} ..."
        raise CommandParseError(f"Usage: {usage}")


def _number(value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise CommandParseError(f"Expected a number, got {value!r}", token=value) from exc


def _integer(value: str, label: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise CommandParseError(f"{label} must be an integer, got {value!r}") from exc


def _value(value: str):
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _suggest(token: str, choices: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(difflib.get_close_matches(token.lower(), choices, n=3, cutoff=0.45))
