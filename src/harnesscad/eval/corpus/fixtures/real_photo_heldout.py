"""GenCAD-Code's real-photo held-out set: image->CAD robustness, REAL not synthetic.

Source: GenCAD-Code (CADCODER, https://github.com/CADCODER/GenCAD-Code),
``real_photo_test_set/`` -- 400 real PHOTOGRAPHS of 50 DeepCAD test-split
objects that were physically 3D-PRINTED and photographed under varied capture
conditions. Every corpus in the harness before this one was synthetic (rendered
images, procedural meshes); this is the harness's first REAL image->CAD
robustness eval set, and its ground truth is resolvable because each photo maps
back -- through the committed ``id_deepcad_pairs`` table -- to a DeepCAD model
the harness already holds.

WHAT MAKES IT AN EVAL SET (not just pictures): the ground-truth CAD is committed,
just indirectly. ``id_map()`` bridges every photo's ``Object_ID`` to a
``DeepCAD_ID``; the DeepCAD solid IS the ground truth. So an image->CAD model's
output can be scored against a real solid, over 400 real photographs.

THE VALUE IS THE METADATA AXES. Each photo carries five closed-vocabulary
capture axes -- colour, orientation, proximity (zoom), background, lighting --
so an image->CAD robustness eval can SLICE accuracy by domain-shift factor
(does accuracy fall on the wood background? at zoom? in one orientation?).
Synthetic renders cannot supply real photographic shift; this set does.

LICENSE: MANIFEST-ONLY. GenCAD-Code commits NO LICENSE and is explicitly
"derived from the DeepCAD dataset" (README line 5), so the derived corpus does
not inherit a redistribution grant on this record. NOTHING is vendored. The
committed bytes here are ONLY ``real_photo_heldout/MANIFEST.json``, which records
every photo's resources-relative path + SHA-256 PLUS the small factual payload
(the 400 metadata rows, the five closed axis vocabularies, and the
Object_ID -> DeepCAD_ID map). Photo files resolve from ``resources/`` at run
time and DEGRADE TO EMPTY (``path=None``) when the checkout is absent; the
FACTS (rows, axes, map) are always present because they are committed metadata,
not licensed pixels. This mirrors the Graph-CAD / CADPrompt manifest pattern.

Photo<->row pairing reproduces the committed
``scripts/upload_realphoto_to_hf.py``: the 400 PNG filenames sorted
lexicographically are zipped, in order, onto the 400 ``RealPhotoTestSet.xlsx``
rows. The MANIFEST bakes that pairing so the loader needs no spreadsheet reader.

Stdlib only. Deterministic. ASCII. No geometry kernel. Degrades to empty when
resources/ is absent.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from harnesscad.eval.corpus.fixtures import (
    Manifest,
    fixtures_dir,
    load_manifest,
    sha256_of,
)

__all__ = [
    "PhotoRecord",
    "AXES",
    "EXPECTED_OBJECTS",
    "EXPECTED_PHOTOS",
    "EXPECTED_PHOTOS_PER_OBJECT",
    "manifest",
    "id_map",
    "axes",
    "photo_records",
    "available_photos",
    "resolve_deepcad_id",
    "records_by_axis",
    "main",
]

_SOURCE = "real_photo_heldout"

#: The corpus is exactly 50 distinct 3D-printed objects x 8 photos = 400.
EXPECTED_OBJECTS = 50
EXPECTED_PHOTOS_PER_OBJECT = 8
EXPECTED_PHOTOS = 400

#: The five robustness axes and their CLOSED vocabularies, as observed in the
#: committed ``RealPhotoTestSet.xlsx`` (asserted against the manifest in
#: --selfcheck). ``notes`` is a free-text capture log, deliberately NOT an axis.
AXES: Dict[str, Tuple[str, ...]] = {
    "color": ("blue",),
    "orientation": ("Position1", "Position2"),
    "proximity": ("normal", "zoom"),
    "background": ("granite", "wood"),
    "lighting": ("natural",),
}


@dataclass(frozen=True)
class PhotoRecord:
    """One real photograph and its resolvable ground-truth CAD identity.

    ``path`` is ``None`` when the resources checkout is absent -- the metadata
    (axes + ids) is committed fact and always present; only the pixels degrade.
    """

    file: str                 # png filename, e.g. IMG_0834.png
    object_id: int            # 1..50, the printed-object index
    deepcad_id: str           # the DeepCAD model id = the ground-truth CAD
    color: str
    orientation: str
    proximity: str
    background: str
    lighting: str
    path: Optional[Path]
    sha256: str

    @property
    def available(self) -> bool:
        return self.path is not None

    @property
    def entry_name(self) -> str:
        """The MANIFEST entry name for this photo's file."""
        stem = self.file[:-4] if self.file.endswith(".png") else self.file
        return "pngs/%s" % stem


def manifest() -> Manifest:
    return load_manifest(_SOURCE)


def _raw() -> dict:
    """The MANIFEST.json as raw JSON -- carries the factual payload the standard
    :class:`Manifest` schema does not model (id_map, axes, photo rows)."""
    data_dir = fixtures_dir() / _SOURCE
    return json.loads((data_dir / "MANIFEST.json").read_text(encoding="utf-8"))


def id_map() -> Dict[str, str]:
    """Object_ID (as string) -> DeepCAD_ID. The bridge to ground-truth CAD."""
    return dict(_raw()["id_map"])


def axes() -> Dict[str, Tuple[str, ...]]:
    """The five closed axis vocabularies as recorded in the manifest."""
    return {k: tuple(v) for k, v in _raw()["axes"].items()}


def photo_records() -> List[PhotoRecord]:
    """All 400 photo records (facts always present; ``path`` may be ``None``)."""
    m = manifest()
    raw = _raw()
    imap = raw["id_map"]
    out: List[PhotoRecord] = []
    for row in raw["photos"]:
        oid = int(row["object_id"])
        stem = row["file"][:-4] if row["file"].endswith(".png") else row["file"]
        entry = m.by_name("pngs/%s" % stem)
        path = m.resolve(entry) if entry is not None else None
        sha = entry.sha256 if entry is not None else ""
        out.append(PhotoRecord(
            file=row["file"],
            object_id=oid,
            deepcad_id=imap[str(oid)],
            color=row["color"],
            orientation=row["orientation"],
            proximity=row["proximity"],
            background=row["background"],
            lighting=row["lighting"],
            path=path,
            sha256=sha,
        ))
    return out


def available_photos() -> List[PhotoRecord]:
    """Only records whose PNG resolves from resources/. Empty when absent."""
    return [r for r in photo_records() if r.available]


def resolve_deepcad_id(record: PhotoRecord) -> str:
    """The ground-truth DeepCAD model id for a photo -- its resolvable GT CAD."""
    return record.deepcad_id


def records_by_axis(axis: str, value: str) -> List[PhotoRecord]:
    """Every record whose ``axis`` equals ``value`` -- the robustness slice.

    ``axis`` must be one of :data:`AXES`; ``value`` must be in that axis's
    closed vocabulary.
    """
    if axis not in AXES:
        raise KeyError("no such axis: %r (known: %s)"
                       % (axis, ", ".join(sorted(AXES))))
    if value not in AXES[axis]:
        raise ValueError("value %r not in closed vocabulary for %s: %s"
                         % (value, axis, ", ".join(AXES[axis])))
    return [r for r in photo_records() if getattr(r, axis) == value]


def _selfcheck() -> int:
    m = manifest()
    raw = _raw()
    assert m.license == "NOLICENSE-DEEPCAD-DERIVED", m.license

    # Manifest-only by design: nothing vendored, every entry carries a resource
    # path + a full SHA-256.
    assert not m.verify_vendored(), "no vendored files were expected"
    for e in m.entries:
        assert e.vendored is None, "unexpected vendored file: %s" % e.name
        assert e.resource, "entry %s has no resource path" % e.name
        assert len(e.sha256) == 64, "entry %s has no sha256" % e.name

    # Role census: 400 photos + the 2 provenance spreadsheets.
    photos_entries = m.by_role("photo")
    assert len(photos_entries) == EXPECTED_PHOTOS, len(photos_entries)
    assert len(m.by_role("metadata_source")) == 1
    assert len(m.by_role("idmap_source")) == 1

    # The id map covers all 50 objects, every value a DeepCAD id string.
    imap = raw["id_map"]
    assert len(imap) == EXPECTED_OBJECTS, len(imap)
    assert set(imap.keys()) == {str(i) for i in range(1, EXPECTED_OBJECTS + 1)}
    for oid, did in imap.items():
        assert isinstance(did, str) and did, (oid, did)

    # The five axes: the embedded closed vocabularies MATCH the manifest's, and
    # the manifest declares exactly these five axes (no more, no fewer).
    assert set(raw["axes"].keys()) == set(AXES.keys()), raw["axes"].keys()
    for name, vocab in AXES.items():
        assert tuple(raw["axes"][name]) == tuple(vocab), (name, raw["axes"][name])

    recs = photo_records()
    assert len(recs) == EXPECTED_PHOTOS, len(recs)

    # Every photo row resolves to a DeepCAD id via the map, and every axis value
    # is drawn from its CLOSED vocabulary (the robustness slices are total).
    for r in recs:
        assert r.deepcad_id == imap[str(r.object_id)], r.file
        assert r.deepcad_id, r.file
        assert r.color in AXES["color"], (r.file, r.color)
        assert r.orientation in AXES["orientation"], (r.file, r.orientation)
        assert r.proximity in AXES["proximity"], (r.file, r.proximity)
        assert r.background in AXES["background"], (r.file, r.background)
        assert r.lighting in AXES["lighting"], (r.file, r.lighting)

    # 50 distinct objects, each with exactly 8 photos.
    counts: Dict[int, int] = {}
    for r in recs:
        counts[r.object_id] = counts.get(r.object_id, 0) + 1
    assert len(counts) == EXPECTED_OBJECTS, len(counts)
    assert set(counts.values()) == {EXPECTED_PHOTOS_PER_OBJECT}, set(counts.values())

    # Slicing works and partitions the corpus along a multi-value axis.
    p1 = records_by_axis("orientation", "Position1")
    p2 = records_by_axis("orientation", "Position2")
    assert len(p1) + len(p2) == EXPECTED_PHOTOS, (len(p1), len(p2))

    avail = m.availability()
    if avail["present"] == 0:
        print("SELFCHECK OK: manifest-only, resources/ absent -> photo pixels "
              "degrade to empty as designed; the facts are intact "
              "(%d photos, %d objects x %d, 5 closed axes, id_map covers all "
              "%d objects, every row resolves to a DeepCAD id)"
              % (EXPECTED_PHOTOS, EXPECTED_OBJECTS, EXPECTED_PHOTOS_PER_OBJECT,
                 EXPECTED_OBJECTS))
        return 0

    # When present, spot-verify resolved photo bytes against the manifest SHA.
    present = available_photos()
    for r in present[:8]:
        entry = m.by_name(r.entry_name)
        assert entry is not None
        assert sha256_of(r.path) == entry.sha256, "drift: %s" % r.file
    print("SELFCHECK OK: %d/%d photos resolvable from resources/ (SHA spot-"
          "checked); %d objects x %d, 5 closed axes, all rows -> DeepCAD id"
          % (len(present), EXPECTED_PHOTOS, EXPECTED_OBJECTS,
             EXPECTED_PHOTOS_PER_OBJECT))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="GenCAD-Code real-photo held-out set (400 photos of 50 "
                    "3D-printed DeepCAD objects; manifest-only, facts embedded, "
                    "nothing vendored).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate the manifest, the closed axis "
                             "vocabularies, the id map and the 50x8 census; "
                             "degrades cleanly when resources/ is absent.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0
    try:
        return _selfcheck()
    except AssertionError as exc:
        print("SELFCHECK FAILED: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
