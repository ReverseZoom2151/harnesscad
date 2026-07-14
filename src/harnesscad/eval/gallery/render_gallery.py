"""The gallery driver: build every catalogued part, render it, and QC the PNG.

Three stages, all deterministic:

1. **build**   -- an op stream goes through :class:`~harnesscad.io.surfaces.
   server.CISPServer` on the part's preferred backend; a mesh part comes
   straight out of the geometry services. Either way the product is a
   ``(vertices, faces)`` triangle mesh.
2. **render**  -- :func:`harnesscad.io.render.render` rasterises it: z-buffered
   shaded solid, crease-aware normals, feature edges drawn over the top. Parts
   flagged ``drawing`` also get a multi-view orthographic SVG from
   :func:`harnesscad.io.drawing.orthographic_drawing`.
3. **verify**  -- the PNG we just wrote is DECODED BACK (stdlib ``zlib``,
   scanline un-filtering) and checked: magic bytes, dimensions, luminance
   variance, silhouette coverage, distinct-colour count. A render that is blank,
   flat, empty or overflowing the frame FAILS here and is reported as failed --
   it is never quietly shipped.

Content-addressed: every result carries the SHA-256 of the PNG bytes and of the
mesh, so a rebuild that changes an image is visible in the manifest diff.
Stdlib only; no wall clock enters any written artefact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.eval.gallery import parts as catalogue
from harnesscad.eval.gallery.parts import COMPARE_BACKENDS, COMPARE_PART, Part
from harnesscad.io import drawing as drawing_route
from harnesscad.io import gate
from harnesscad.io import render as render_route
from harnesscad.io.surfaces.server import CISPServer

__all__ = [
    "GalleryError",
    "DEFAULT_OUT",
    "QC",
    "build_part",
    "render_part",
    "verify_png",
    "decode_png",
    "build_gallery",
    "compare_backends",
    "add_arguments",
    "run_cli",
]

#: Where the gallery is written, relative to the repo root.
DEFAULT_OUT = os.path.join("assets", "gallery")

#: Hero render size. Fixed, so every image in the gallery composes on a page.
WIDTH, HEIGHT, SSAA = 1200, 900, 2

#: The gallery's own light rig. The renderer's DEFAULT_LIGHTS sum to
#: ambient 0.30 + key 0.82 = 1.12, so any face square-on to the key light
#: CLIPS TO WHITE -- on a flat end-cap (the swept duct) that is a white patch
#: indistinguishable from the background. Here the key is dialled back so the
#: brightest possible surface lands just under saturation and every face keeps
#: its shading.
GALLERY_MATERIAL = render_route.Material(base_color=(172, 184, 199), ambient=0.28,
                                         specular=0.20, shininess=42.0)
GALLERY_LIGHTS = (
    render_route.Light(direction=(-0.35, 0.55, 0.76), color=(255, 253, 246),
                       intensity=0.62),
    render_route.Light(direction=(0.62, -0.18, 0.42), color=(206, 220, 240),
                       intensity=0.26),
)


class GalleryError(Exception):
    """A part could not be built, rendered, or passed image QC."""


# ---------------------------------------------------------------------------
# image QC -- decode our own PNG and prove it is a real render
# ---------------------------------------------------------------------------
class QC:
    """The thresholds a gallery PNG must clear to be shipped.

    ``MIN_SHADES`` is the one that carries the "not a single flat colour" claim,
    and it is deliberately NOT a count of distinct RGB triples: a legitimately
    PLANAR solid (a shelled box, a comb of fins) has only a handful of distinct
    triples -- one Lambert value per plane -- and would fail a naive colour
    count while being a perfectly correct render. What actually distinguishes a
    render from a flat fill is the number of distinct SHADING TONES inside the
    silhouette: a flat fill has 1 (2 with an outline), a box has ~13, a curved
    or lattice part has 100+. ``MAX_FLAT_SHARE`` then rejects the near-flat blob
    that clears the tone count on its anti-aliased fringe alone.
    """

    #: luminance variance over the whole frame; a blank / flat image is ~0
    MIN_VARIANCE = 40.0
    #: fraction of pixels that are not the background colour
    MIN_COVERAGE = 0.04
    MAX_COVERAGE = 0.96
    #: distinct luminance levels INSIDE the silhouette (see the note above)
    MIN_SHADES = 5
    #: the most common tone may not own this much of the silhouette
    MAX_FLAT_SHARE = 0.90
    #: the render background (render_route.render's default)
    BACKGROUND = (255, 255, 255)


def decode_png(path: str) -> Tuple[int, int, List[int]]:
    """Decode an 8-bit RGB PNG to (width, height, [r, g, b, ...]).

    Stdlib only: parse the chunks, inflate IDAT, undo the per-scanline filters
    (PNG filter types 0..4). We decode the file we just WROTE -- that is the
    point: the QC must not trust the writer.
    """
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise GalleryError("%s is not a PNG (bad magic)" % path)
    pos = 8
    width = height = depth = colour = -1
    idat = bytearray()
    while pos + 8 <= len(raw):
        (length,) = struct.unpack(">I", raw[pos:pos + 4])
        tag = raw[pos + 4:pos + 8]
        body = raw[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", body[:10])
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
    if width <= 0 or height <= 0:
        raise GalleryError("%s has no IHDR" % path)
    if depth != 8 or colour not in (2, 6):
        raise GalleryError("%s is not 8-bit RGB/RGBA (depth=%d colour=%d)"
                           % (path, depth, colour))
    channels = 3 if colour == 2 else 4
    data = zlib.decompress(bytes(idat))
    stride = width * channels
    if len(data) != (stride + 1) * height:
        raise GalleryError("%s: IDAT is %d bytes, expected %d"
                           % (path, len(data), (stride + 1) * height))

    out = bytearray(stride * height)
    prev = bytearray(stride)
    off = 0
    for y in range(height):
        ftype = data[off]
        off += 1
        line = bytearray(data[off:off + stride])
        off += stride
        if ftype == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif ftype == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        elif ftype != 0:
            raise GalleryError("%s: unknown PNG filter %d on row %d"
                               % (path, ftype, y))
        out[y * stride:(y + 1) * stride] = line
        prev = line

    if channels == 3:
        return width, height, list(out)
    rgb: List[int] = []
    for i in range(0, len(out), 4):
        rgb.extend(out[i:i + 3])
    return width, height, rgb


def verify_png(path: str, width: int, height: int) -> dict:
    """Decode the PNG and measure it. Raises :class:`GalleryError` if it is not a render.

    Checks, in order:
      * the file decodes at all, and at the expected dimensions;
      * luminance variance -- a blank, black or flat-filled frame has ~0;
      * silhouette coverage -- the fraction of non-background pixels must be a
        sensible share of the frame (not empty, not overflowing it);
      * shading tones inside the silhouette -- a lit solid has many, a flat
        fill has one (and no single tone may own almost all of the part).
    """
    w, h, px = decode_png(path)
    if (w, h) != (width, height):
        raise GalleryError("%s is %dx%d, expected %dx%d" % (path, w, h, width, height))

    n = w * h
    total = 0.0
    total_sq = 0.0
    bg = QC.BACKGROUND
    covered = 0
    colours = set()
    shades: Dict[int, int] = {}
    for i in range(0, len(px), 3):
        r, g, b = px[i], px[i + 1], px[i + 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        total += lum
        total_sq += lum * lum
        colours.add((r, g, b))
        if (r, g, b) != bg:
            covered += 1
            key = int(lum)
            shades[key] = shades.get(key, 0) + 1
    mean = total / n
    variance = max(0.0, total_sq / n - mean * mean)
    coverage = covered / n
    flat_share = (max(shades.values()) / covered) if covered else 1.0
    metrics = {
        "width": w, "height": h,
        "variance": round(variance, 2),
        "coverage": round(coverage, 4),
        "colours": len(colours),
        "shades": len(shades),
        "flat_share": round(flat_share, 4),
        "mean_luminance": round(mean, 2),
        "bytes": os.path.getsize(path),
    }
    problems = []
    if variance < QC.MIN_VARIANCE:
        problems.append("variance %.2f < %.1f (blank or flat image)"
                        % (variance, QC.MIN_VARIANCE))
    if coverage < QC.MIN_COVERAGE:
        problems.append("silhouette covers %.2f%% of the frame < %.1f%% (nothing drawn)"
                        % (coverage * 100.0, QC.MIN_COVERAGE * 100.0))
    if coverage > QC.MAX_COVERAGE:
        problems.append("silhouette covers %.2f%% of the frame > %.1f%% (overflows it)"
                        % (coverage * 100.0, QC.MAX_COVERAGE * 100.0))
    if len(shades) < QC.MIN_SHADES:
        problems.append("only %d shading tones in the silhouette < %d (flat fill)"
                        % (len(shades), QC.MIN_SHADES))
    if flat_share > QC.MAX_FLAT_SHARE:
        problems.append("one tone owns %.1f%% of the silhouette > %.1f%% (near-flat)"
                        % (flat_share * 100.0, QC.MAX_FLAT_SHARE * 100.0))
    if problems:
        raise GalleryError("%s failed image QC: %s" % (path, "; ".join(problems)))
    return metrics


def consistently_wound(mesh) -> bool:
    """Is every interior edge traversed once in each direction?

    Backface culling is only sound on a CONSISTENTLY ORIENTED mesh. The meshes
    out of ``features.sweep.extrude_along_path`` are closed but NOT consistently
    oriented -- its end caps are wound against its tube walls (the halfedge
    check reports nonmanifold-edge + boundary on the swept duct, the spring and
    the bolt's prisms alike). Culling such a mesh deletes the faces nearest the
    camera and the solid renders inside-out.

    So we measure instead of assuming: if any directed edge occurs twice, the
    winding is inconsistent and the part is rendered with culling OFF, where the
    z-buffer and the renderer's two-sided lighting produce the correct image.
    The SDF-tessellated parts (marching cubes) ARE consistently wound and keep
    the cull.
    """
    seen = set()
    for (a, b, c) in mesh[1]:
        for e in ((a, b), (b, c), (c, a)):
            if e in seen:
                return False
            seen.add(e)
    return True


def _sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _mesh_digest(mesh) -> str:
    verts, faces = mesh
    h = hashlib.sha256()
    for v in verts:
        h.update(b"%.6f,%.6f,%.6f;" % (v[0], v[1], v[2]))
    for f in faces:
        h.update(b"%d,%d,%d;" % (f[0], f[1], f[2]))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build_part(part: Part, backend: Optional[str] = None) -> dict:
    """Build ``part`` and return ``{mesh, backend, ok, diagnostics, seconds}``.

    ``kind == "ops"`` runs the op stream through a real CISPServer on the named
    backend (default: the part's preferred one) and takes the session's mesh.
    ``kind == "mesh"`` calls the geometry service directly. A part whose ops are
    rejected comes back ``ok=False`` with the backend's typed diagnostics -- it
    is never silently substituted with something else.
    """
    started = time.perf_counter()
    if part.kind == "mesh":
        mesh = part.builder()
        verts, faces = mesh
        ok = bool(faces)
        return {
            "mesh": mesh, "backend": catalogue.SERVICES_BACKEND, "ok": ok,
            "diagnostics": [] if ok else [{"severity": "error", "code": "empty-mesh",
                                           "message": "the builder produced no triangles"}],
            "seconds": time.perf_counter() - started,
            "vertices": len(verts), "faces": len(faces),
        }

    name = backend or part.backend
    server = CISPServer(backend=name)
    if server.backend_name != name:
        raise GalleryError("backend %r is unavailable (%s)" % (name, server.backend_note))
    result = server.applyOps([dict(op) for op in part.builder()])
    diags = [d for d in result.get("diagnostics", []) if d["severity"] == "error"]
    if not result["ok"]:
        return {"mesh": None, "backend": name, "ok": False, "diagnostics": diags,
                "seconds": time.perf_counter() - started, "vertices": 0, "faces": 0}

    from harnesscad.io.formats import registry as formats
    verts, faces = formats.to_mesh(server.backend).indexed()
    mesh = ([tuple(float(c) for c in v) for v in verts],
            [tuple(int(i) for i in f) for f in faces])
    return {
        "mesh": mesh, "backend": name, "ok": bool(mesh[1]), "diagnostics": diags,
        "seconds": time.perf_counter() - started,
        "vertices": len(mesh[0]), "faces": len(mesh[1]),
        "digest": result.get("digest"),
        # Kept so the output gate can check the DECLARED intent of the op stream
        # (a shell that grew the part, a cut that added volume) and not merely
        # the measured mesh. Without it the gallery is blind to exactly the class
        # of bug that put an oversize enclosure in the README.
        "session": server.session,
    }


def render_part(part: Part, out_dir: str, backend: Optional[str] = None,
                stem: Optional[str] = None) -> dict:
    """Build, render, QC. Returns the manifest row; raises GalleryError on failure."""
    built = build_part(part, backend=backend)
    if not built["ok"]:
        codes = ", ".join("%s: %s" % (d["code"], d["message"])
                          for d in built["diagnostics"]) or "no geometry"
        raise GalleryError("%s did not build on %s (%s)"
                           % (part.name, built["backend"], codes))

    os.makedirs(out_dir, exist_ok=True)
    base = stem or part.name
    png = os.path.join(out_dir, base + ".png")
    wound = consistently_wound(built["mesh"])
    t0 = time.perf_counter()
    render_route.render(built["mesh"], png, view=part.view,
                        width=WIDTH, height=HEIGHT, ssaa=SSAA,
                        shading="smooth", edges=True, projection="orthographic",
                        material=GALLERY_MATERIAL, lights=GALLERY_LIGHTS,
                        cull=wound, source=built.get("session"))
    render_seconds = time.perf_counter() - t0
    metrics = verify_png(png, WIDTH, HEIGHT)

    row = {
        "name": part.name,
        "png": os.path.basename(png),
        "backend": built["backend"],
        "kind": part.kind,
        "capability": part.capability,
        "operation": part.operation,
        "cisp_ops": list(part.cisp_ops),
        "summary": part.summary,
        "demonstrates": part.demonstrates,
        "unsupported": list(part.unsupported),
        "resolution": part.resolution,
        "vertices": built["vertices"],
        "faces": built["faces"],
        "consistently_wound": wound,
        "backface_cull": wound,
        "build_seconds": round(built["seconds"], 2),
        "render_seconds": round(render_seconds, 2),
        "sha256": _sha256(png),
        "mesh_sha256": _mesh_digest(built["mesh"]),
        "qc": metrics,
    }

    if part.drawing:
        svg_path = os.path.join(out_dir, base + "-drawing.svg")
        svg = drawing_route.orthographic_drawing(
            built["mesh"], views=("front", "top", "side", "iso"),
            width=1100.0, height=800.0, show_hidden=True, show_dimensions=True,
            title="harnesscad / " + part.name)
        # THE GATE: a dimensioned engineering drawing of a wrong part is the
        # single most dangerous artifact the harness can emit.
        gate.guard(built["mesh"], svg_path, source=built.get("session"))
        with open(svg_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg)
        row["drawing"] = os.path.basename(svg_path)
        row["drawing_metrics"] = drawing_route.drawing_metrics(svg)
    return row


# ---------------------------------------------------------------------------
# the whole gallery
# ---------------------------------------------------------------------------
def build_gallery(out_dir: str = DEFAULT_OUT, only: Optional[str] = None,
                  compare: bool = True, log=None) -> dict:
    """Render every catalogued part (or just ``only``) and write the manifest."""
    say = log or (lambda _m: None)
    selected = [catalogue.get(only)] if only else list(catalogue.CATALOGUE)
    rows: List[dict] = []
    failures: List[dict] = []
    for part in selected:
        try:
            row = render_part(part, out_dir)
        except (GalleryError, Exception) as exc:  # noqa: BLE001 - reported, not raised
            failures.append({"name": part.name, "error": "%s: %s"
                             % (type(exc).__name__, exc)})
            say("FAIL %-22s %s" % (part.name, exc))
            continue
        rows.append(row)
        say("ok   %-22s %-9s %6d faces  build %5.1fs  render %5.1fs  "
            "var %7.1f  cov %.3f"
            % (row["name"], row["backend"], row["faces"], row["build_seconds"],
               row["render_seconds"], row["qc"]["variance"], row["qc"]["coverage"]))

    comparison: List[dict] = []
    if compare and (only is None or only == COMPARE_PART):
        comparison = compare_backends(out_dir, log=say)

    manifest = {
        "parts": rows,
        "failed": failures,
        "comparison": comparison,
        "compare_part": COMPARE_PART,
        "counts": {"catalogued": len(catalogue.CATALOGUE),
                   "rendered": len(rows), "failed": len(failures)},
    }
    if not only:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "manifest.json")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
            fh.write("\n")
        manifest["manifest"] = path
    return manifest


def compare_backends(out_dir: str = DEFAULT_OUT,
                     backends: Sequence[str] = COMPARE_BACKENDS,
                     log=None) -> List[dict]:
    """Render the SAME part on every kernel: ``compare-<backend>.png``.

    The point is visible, not rhetorical: the frep backend meshes a signed
    distance field on a grid, so its counterbores are faceted polygons; the
    OCCT / CGAL / Blender kernels carry the cylinder exactly and tessellate it
    at export, so their bores are smooth. Same op stream, same camera, four
    different geometry engines.
    """
    say = log or (lambda _m: None)
    part = catalogue.get(COMPARE_PART)
    rows: List[dict] = []
    for name in backends:
        try:
            row = render_part(part, out_dir, backend=name, stem="compare-" + name)
        except (GalleryError, Exception) as exc:  # noqa: BLE001
            say("FAIL compare-%-13s %s" % (name, exc))
            rows.append({"backend": name, "error": str(exc)})
            continue
        row["backend"] = name
        rows.append(row)
        say("ok   compare-%-13s %6d faces  build %5.1fs  render %5.1fs"
            % (name, row["faces"], row["build_seconds"], row["render_seconds"]))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--list", action="store_true", dest="list_parts",
                        help="print the catalogue (name, capability, backend)")
    parser.add_argument("--build", action="store_true",
                        help="build + render + QC every catalogued part")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="output directory (default: %s)" % DEFAULT_OUT)
    parser.add_argument("--only", default=None, metavar="NAME",
                        help="render just this part")
    parser.add_argument("--no-compare", action="store_true", dest="no_compare",
                        help="skip the cross-backend comparison strip")
    parser.add_argument("--json", action="store_true",
                        help="emit the catalogue / manifest as JSON")
    return parser


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "list_parts", False) or not getattr(args, "build", False):
        rows = [p.to_dict() for p in catalogue.CATALOGUE]
        if getattr(args, "json", False):
            print(json.dumps({"parts": rows, "compare_part": COMPARE_PART,
                              "compare_backends": list(COMPARE_BACKENDS)},
                             indent=2, sort_keys=True))
            return 0
        print("%d parts in the gallery catalogue\n" % len(rows))
        for r in rows:
            print("%-22s %-10s %s" % (r["name"], r["backend"], r["summary"]))
            print("%-22s %s %s" % ("", "demonstrates:", r["demonstrates"]))
            print("%-22s %s %s%s" % ("", "capability:  ", r["capability"],
                                     (" -> " + r["operation"]) if r["operation"] else ""))
            if r["cisp_ops"]:
                print("%-22s %s %s" % ("", "cisp ops:    ", " ".join(r["cisp_ops"])))
            if r["unsupported"]:
                print("%-22s %s %s -- %s" % ("", "cannot build:",
                                             ", ".join(r["unsupported"]), r["why_not"]))
            print()
        print("cross-backend comparison: %s on %s"
              % (COMPARE_PART, ", ".join(COMPARE_BACKENDS)))
        return 0

    manifest = build_gallery(out_dir=args.out, only=args.only,
                             compare=not getattr(args, "no_compare", False),
                             log=lambda m: print(m, flush=True))
    if getattr(args, "json", False):
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if not manifest["failed"] else 1
    c = manifest["counts"]
    print()
    print("rendered %d/%d parts into %s" % (c["rendered"], c["catalogued"], args.out))
    if manifest["comparison"]:
        good = [r for r in manifest["comparison"] if "error" not in r]
        print("cross-backend: %s on %d kernels"
              % (COMPARE_PART, len(good)))
    if manifest["failed"]:
        print("FAILED (%d):" % len(manifest["failed"]))
        for f in manifest["failed"]:
            print("  %-22s %s" % (f["name"], f["error"]))
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="gallery", description=__doc__)
    add_arguments(parser)
    return run_cli(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
