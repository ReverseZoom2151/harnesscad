"""Kernel-neutral, directed B-rep hierarchy with strict integrity checks."""

from __future__ import annotations

from dataclasses import dataclass

Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class Vertex:
    id: str
    point: Point3


@dataclass(frozen=True)
class Edge:
    id: str
    start: str
    end: str
    geometry: tuple[float, ...] = ()


@dataclass(frozen=True)
class Coedge:
    id: str
    edge: str
    forward: bool = True


@dataclass(frozen=True)
class Loop:
    id: str
    coedges: tuple[str, ...]
    outer: bool = False


@dataclass(frozen=True)
class Face:
    id: str
    loops: tuple[str, ...]
    geometry: tuple[float, ...] = ()


@dataclass(frozen=True)
class Shell:
    id: str
    faces: tuple[str, ...]
    closed: bool = True


@dataclass(frozen=True)
class BRepHierarchy:
    vertices: tuple[Vertex, ...]
    edges: tuple[Edge, ...]
    coedges: tuple[Coedge, ...]
    loops: tuple[Loop, ...]
    faces: tuple[Face, ...]
    shells: tuple[Shell, ...]

    def validate(self, *, manifold: bool = False) -> tuple[str, ...]:
        issues: list[str] = []
        tables = {
            "vertex": self.vertices, "edge": self.edges, "coedge": self.coedges,
            "loop": self.loops, "face": self.faces, "shell": self.shells,
        }
        ids = {}
        for kind, records in tables.items():
            ids[kind] = {record.id for record in records}
            if len(ids[kind]) != len(records):
                issues.append(f"duplicate-{kind}-id")
            if any(not record.id for record in records):
                issues.append(f"empty-{kind}-id")
        for edge in self.edges:
            if edge.start not in ids["vertex"] or edge.end not in ids["vertex"]:
                issues.append(f"edge-{edge.id}-unknown-vertex")
        edge_by_id = {edge.id: edge for edge in self.edges}
        coedge_by_id = {coedge.id: coedge for coedge in self.coedges}
        for coedge in self.coedges:
            if coedge.edge not in ids["edge"]:
                issues.append(f"coedge-{coedge.id}-unknown-edge")
        for loop in self.loops:
            if not loop.coedges:
                issues.append(f"loop-{loop.id}-empty")
                continue
            if any(item not in ids["coedge"] for item in loop.coedges):
                issues.append(f"loop-{loop.id}-unknown-coedge")
                continue
            oriented = []
            for item in loop.coedges:
                coedge = coedge_by_id[item]
                edge = edge_by_id.get(coedge.edge)
                if edge:
                    oriented.append((edge.start, edge.end) if coedge.forward
                                    else (edge.end, edge.start))
            if oriented and any(a[1] != b[0]
                                for a, b in zip(oriented, oriented[1:] + oriented[:1])):
                issues.append(f"loop-{loop.id}-not-closed")
        loop_by_id = {loop.id: loop for loop in self.loops}
        for face in self.faces:
            known = [loop_by_id[item] for item in face.loops if item in loop_by_id]
            if len(known) != len(face.loops):
                issues.append(f"face-{face.id}-unknown-loop")
            if sum(loop.outer for loop in known) != 1:
                issues.append(f"face-{face.id}-outer-loop-count")
        for shell in self.shells:
            if any(item not in ids["face"] for item in shell.faces):
                issues.append(f"shell-{shell.id}-unknown-face")
        if manifold:
            incidence = {edge.id: 0 for edge in self.edges}
            for coedge in self.coedges:
                if coedge.edge in incidence:
                    incidence[coedge.edge] += 1
            for edge_id, count in sorted(incidence.items()):
                if count != 2:
                    issues.append(f"edge-{edge_id}-incidence-{count}")
        return tuple(dict.fromkeys(issues))

    def assert_valid(self, *, manifold: bool = False) -> None:
        issues = self.validate(manifold=manifold)
        if issues:
            raise ValueError("; ".join(issues))

    def face_neighbors(self) -> dict[str, tuple[str, ...]]:
        loop_to_face = {loop: face.id for face in self.faces for loop in face.loops}
        coedge_to_loop = {coedge: loop.id for loop in self.loops for coedge in loop.coedges}
        edge_faces: dict[str, set[str]] = {}
        coedges = {item.id: item for item in self.coedges}
        for coedge_id, loop_id in coedge_to_loop.items():
            coedge = coedges.get(coedge_id)
            face = loop_to_face.get(loop_id)
            if coedge and face:
                edge_faces.setdefault(coedge.edge, set()).add(face)
        neighbors = {face.id: set() for face in self.faces}
        for linked in edge_faces.values():
            for face in linked:
                neighbors[face].update(linked - {face})
        return {key: tuple(sorted(value)) for key, value in sorted(neighbors.items())}
