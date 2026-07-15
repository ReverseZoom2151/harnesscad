"""The COMPUTER-USE surface -- drive a real CAD GUI, and know which one can run here.

``io/cua`` carried a whole computer-use-for-CAD subsystem: five live-GUI
Environments (FreeCAD, Onshape, SolidWorks, Fusion, Inventor) and the deterministic
action primitives they compose from -- template matching, computed viewport picks,
coordinate decoding, window focus discipline, the reset checklist, the UIA hazards
audit, the viewport region grouper, the Computer/Agent wire protocol and the
grounding action format. None of it was reachable: nothing imports these modules and
no dispatcher reached them. This module is the dispatcher, mirroring every other
subsystem's ``registry.py``.

    environments()      -> every CUA environment by name + its available() status
    discover()          -> every routed primitive module (present/absent)
    environment_class(n)-> the Environment class for one app (lazy, never launched)

WHY AN ENUMERATOR AND NOT A DRIVER. The commercial environments (SolidWorks, Fusion,
Inventor, Onshape) are credential/install-gated and almost never runnable on a build
box; FreeCAD's GUI needs the windowed binary. Each environment's ``available()``
reports -- never raising, never hanging -- exactly what is missing, so this surface
is the honest way to ask "which CAD apps could this machine drive?" without touching
one. Nothing here launches an app, opens a document, or drives a GUI.

Every route imports lazily, so this module loads and enumerates with no CAD
application, no browser, and no accessibility stack present. Deterministic,
stdlib-only, no network.
"""

from __future__ import annotations

import argparse
import importlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "CuaError",
    "environments",
    "environment_names",
    "environment_class",
    "discover",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

_CUA = "harnesscad.io.cua."


class CuaError(ValueError):
    """Base class for every computer-use-surface failure."""


# --------------------------------------------------------------------------- #
# The environments: one live-GUI Environment per CAD app. Each module exposes a
# class + an available() -> (bool, reason) that never raises and never launches.
# --------------------------------------------------------------------------- #
_ENVIRONMENTS: Tuple[Tuple[str, str, str, str], ...] = (
    ("freecad", "environment_freecad", "FreeCADGuiEnvironment",
     "FreeCAD's windowed GUI, driven by its accessibility tree; the proven path "
     "(scripted-backend differential to 4.5e-16)"),
    ("onshape", "environment_onshape", "OnshapeGuiEnvironment",
     "an Onshape browser session as the actuator with the REST API as a separate, "
     "agent-untouchable oracle; credential- and browser-gated"),
    ("solidworks", "environment_solidworks", "SolidWorksGuiEnvironment",
     "the SolidWorks ribbon as the actuator with the ISldWorks COM object as the "
     "oracle; install-gated (win32com + a registered COM class)"),
    ("fusion", "environment_fusion", "FusionGuiEnvironment",
     "the Fusion 360 ribbon as the actuator with the adsk Python API as the "
     "oracle; runs only inside a live Fusion process"),
    ("inventor", "environment_inventor", "InventorGuiEnvironment",
     "the Inventor ribbon as the actuator with the Inventor.Application COM object "
     "as the oracle; install-gated (win32com + a registered COM class)"),
)


# --------------------------------------------------------------------------- #
# The action primitives every environment composes from. DATA + deterministic
# logic; each is import-safe without the optional GUI/vision dependencies.
# --------------------------------------------------------------------------- #
_ROUTES: Tuple[Tuple[str, str, str, str], ...] = (
    ("primitive", "primitives", _CUA + "primitives",
     "the OS-control action-primitive spec + deterministic icon template matching "
     "(NCC), the free grounding path pixel-stable CAD toolbars allow"),
    ("primitive", "picks", _CUA + "picks",
     "turn a CISP selector into concrete entities, then COMPUTED viewport clicks "
     "(never a vision guess; the pick is projected from the owned B-rep)"),
    ("primitive", "coords", _CUA + "coords",
     "decode a model's coordinate output and REFUSE to guess its space (the "
     "declared-space discipline, not magnitude inference)"),
    ("primitive", "windowing", _CUA + "windowing",
     "focus/raise discipline, self-occlusion detection, and monitor targeting"),
    ("primitive", "reset", _CUA + "reset",
     "the environment-state reset checklist and the state-leak detector (a CUA "
     "whose state leaks between trials is worthless)"),
    ("primitive", "hazards", _CUA + "hazards",
     "the Windows UIA known-hazards checklist, as checkable data, plus plan/driver "
     "audits"),
    ("primitive", "region_group", _CUA + "region_group",
     "a deterministic screen-region grouper for the opaque 3D viewport"),
    ("primitive", "wire", _CUA + "wire",
     "the Computer/Agent JSON wire-envelope + reflective handler dispatch"),
    ("primitive", "action_stream", _CUA + "action_stream",
     "ShowUI's interleaved action format and the grounding-pair corpus format"),
)


# --------------------------------------------------------------------------- #
# Index membership (static; no import)
# --------------------------------------------------------------------------- #
def _index() -> Dict[str, Any]:
    return {e.dotted: e
            for e in capability_registry.find(layer="io", package="cua")}


def _available(dotted: str) -> bool:
    return dotted in _index()


# --------------------------------------------------------------------------- #
# Environments
# --------------------------------------------------------------------------- #
def _env_status(module_dotted: str) -> Tuple[bool, str]:
    """Import one environment module and ask its ``available()`` -- defensively.

    The environment modules are written to degrade to ``available() -> False`` with
    no CAD app, browser, or accessibility stack present, so this must not raise even
    when everything the environment needs is missing.
    """
    try:
        mod = importlib.import_module(module_dotted)
    except Exception as exc:  # noqa: BLE001 - a broken optional dep is "unavailable"
        return False, "the environment module did not import: %s" % exc
    fn = getattr(mod, "available", None)
    if not callable(fn):
        return False, "the environment exposes no available() surface"
    try:
        ok, why = fn()
    except Exception as exc:  # noqa: BLE001 - available() promises not to, but guard
        return False, "available() raised: %s" % exc
    return bool(ok), str(why)


def environment_names() -> Tuple[str, ...]:
    """The names of every CUA environment, whether or not it can run here."""
    return tuple(name for name, _s, _c, _d in _ENVIRONMENTS)


def environments() -> List[dict]:
    """Every CUA environment by name, with its live ``available()`` verdict.

    ``indexed`` is static index membership; ``available`` is the environment's own
    runtime reachability check (credentials, install, browser, a11y stack), with
    ``reason`` naming exactly what is missing when it is False.
    """
    out: List[dict] = []
    for name, suffix, cls, doc in _ENVIRONMENTS:
        module = _CUA + suffix
        avail, why = _env_status(module)
        out.append({"environment": name, "module": module, "class": cls,
                    "doc": doc, "indexed": _available(module),
                    "available": avail, "reason": why})
    return out


def environment_class(name: str) -> type:
    """The Environment CLASS for one app, imported lazily. Never instantiated here.

    This is the real entry surface the commercial environments lacked: a caller
    that HAS the app can obtain the class and construct it itself (which is what
    launches/attaches). Enumerating and handing back the class touches no GUI.
    """
    for env_name, suffix, cls, _doc in _ENVIRONMENTS:
        if env_name == name:
            module = importlib.import_module(_CUA + suffix)
            try:
                return getattr(module, cls)
            except AttributeError as exc:
                raise CuaError("environment %r module %s has no class %r"
                               % (name, module.__name__, cls)) from exc
    raise CuaError("no such CUA environment %r (known: %s)"
                   % (name, ", ".join(environment_names())))


# --------------------------------------------------------------------------- #
# Primitive routes + discovery
# --------------------------------------------------------------------------- #
def routed_modules() -> Tuple[str, ...]:
    """Every present module this registry routes -- primitives AND environments."""
    modules = {m for _g, _n, m, _d in _ROUTES}
    modules.update(_CUA + suffix for _n, suffix, _c, _d in _ENVIRONMENTS)
    return tuple(sorted(m for m in modules if _available(m)))


def discover() -> List[dict]:
    """Every routed primitive module: its group, route, dotted path, and presence."""
    return [{"group": g, "route": n, "module": m, "doc": d,
             "present": _available(m)}
            for (g, n, m, d) in _ROUTES]


def unadapted() -> List[Tuple[str, str]]:
    """Indexed io/cua modules this registry does not yet route (support surfaces).

    The support modules -- frames, quantity, uia, guardrails, viewport, coordinate,
    and the per-app bindings tables -- are consumed by the environments and
    primitives, not dispatched directly, so they are listed as adapted-by-use rather
    than routed.
    """
    routed = set(routed_modules())
    return [(d, "support surface (consumed by an environment/primitive, not routed)")
            for d in sorted(_index())
            if d not in routed and not d.endswith(".registry")]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every routed primitive module")
    parser.add_argument("--environments", action="store_true",
                        help="list every CUA environment and whether it can run here")
    parser.add_argument("--unadapted", action="store_true",
                        help="list io/cua support modules with no direct route")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "unadapted", False):
        rows = unadapted()
        if getattr(args, "json", False):
            print(json.dumps([{"module": d, "reason": r} for d, r in rows],
                             indent=2, sort_keys=True))
            return 0
        for dotted, reason in rows:
            print("%s\n    %s" % (dotted, reason))
        return 0

    if getattr(args, "environments", False):
        rows = environments()
        if getattr(args, "json", False):
            print(json.dumps(rows, indent=2, sort_keys=True))
            return 0
        width = max(len(r["environment"]) for r in rows)
        for r in rows:
            mark = "+" if r["available"] else "-"
            print("%s %-*s  %s" % (mark, width, r["environment"], r["doc"]))
            if not r["available"]:
                print("  %*s  unavailable: %s" % (width, "", r["reason"]))
        return 0

    rows = discover()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(r["route"]) for r in rows)
    for r in rows:
        mark = " " if r["present"] else "-"
        print("%s %-9s %-*s  %s" % (mark, r["group"], width, r["route"], r["doc"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad cua",
        description="computer-use surface: the five live-GUI CAD environments and "
                    "the deterministic action primitives they compose from")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
