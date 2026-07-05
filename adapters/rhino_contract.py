"""Safe DTO/protocol boundary for an optional Rhino/Grasshopper host."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class HostCapabilities:
    commands: frozenset[str]
    analyses: frozenset[str] = frozenset()

@dataclass(frozen=True)
class HostScript:
    id: str
    target: str
    source: str
    commands: tuple[str, ...]
    bake: bool = False

@dataclass(frozen=True)
class HostResult:
    ok: bool
    message: str
    rollback_token: str | None = None

class RhinoHost(Protocol):
    def capabilities(self) -> HostCapabilities: ...
    def preview(self, script: HostScript) -> dict: ...
    def execute(self, script: HostScript) -> HostResult: ...
    def rollback(self, token: str) -> HostResult: ...

def validate_script(script, capabilities):
    issues=[]
    if script.target not in {"rhinoscript","grasshopper"}: issues.append("unsupported-target")
    denied=sorted(set(script.commands)-set(capabilities.commands))
    if denied: issues.append("unsupported-command:"+",".join(denied))
    if not script.source.strip(): issues.append("empty-source")
    return tuple(issues)
