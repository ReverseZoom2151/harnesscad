"""Code-CAD ecosystem knowledge base (deterministic, stdlib-only).

Mined from the ``curated-code-cad`` awesome-list (Irev-Dev / CadHub), which is a
curated markdown index of every notable code-CAD language, library, kernel and
node editor, with a per-entry license line, a prose description of the paradigm,
and an editorial section ("Which one should you use?") that states the
representation trade-offs explicitly.

Scope, and how this differs from ``adapters/cadhub_language_registry``:

* ``cadhub_language_registry`` is about EXECUTING a language: extension, entry
  file, artifact kinds, diagnostics dialect, parameter plumbing. It covers only
  the four languages CadHub actually runs (openscad, cadquery, jscad, curv).
* THIS module is about SELECTING a system: paradigm (CSG / B-rep / mesh / SDF /
  F-rep / parametric-sketch / functional), the underlying geometry kernel (OCCT /
  Carve / manifold / libfive / custom / none), the representation, the file
  formats in and out, host language, license, maturity, whether it is a language
  or a library or an application, and its niche. It covers the whole ecosystem
  (30 systems), including ones the harness cannot execute.

A text-to-CAD harness needs the second table to answer "which code-CAD system
should I target for this request?" before the first table tells it "and here is
how to run it".

Grounding rule: every attribute is either (a) stated in the curated list, or
(b) an uncontroversial, verifiable fact about a well-known tool. Anything the
list does not state and that is not certain is left as ``UNKNOWN`` rather than
guessed -- callers can distinguish "no" from "not known".

Deterministic: a frozen table, pure lookups, sorted outputs. No I/O, no clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

UNKNOWN = "unknown"

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# What kind of artefact the project is.
CAT_LANGUAGE = "language"  # its own DSL / language you write models in
CAT_LIBRARY = "library"  # a library hosted in a general-purpose language
CAT_KERNEL = "kernel"  # a geometry kernel, not a modelling front end
CAT_APPLICATION = "application"  # a GUI CAD app that happens to be scriptable
CAT_PLATFORM = "platform"  # a hosting/sharing platform for code-CAD
CAT_NODE_EDITOR = "node-editor"  # visual / block programming, "not quite code-CAD"

CATEGORIES = (
    CAT_APPLICATION,
    CAT_KERNEL,
    CAT_LANGUAGE,
    CAT_LIBRARY,
    CAT_NODE_EDITOR,
    CAT_PLATFORM,
)

# Modelling paradigms.
PARA_CSG = "csg"  # boolean combinations of primitives
PARA_BREP = "brep"  # boundary representation (faces/edges/vertices, exact)
PARA_MESH = "mesh"  # polygon-mesh representation
PARA_SDF = "sdf"  # signed distance functions
PARA_FREP = "frep"  # functional / implicit representation (superset of sdf usage)
PARA_SKETCH = "parametric-sketch"  # sketch + feature tree (traditional MCAD)
PARA_FUNCTIONAL = "functional"  # functional-programming host / pure-function models
PARA_TRANSPILE = "transpile"  # emits another system's source rather than geometry
PARA_VISUAL = "visual"  # node/block graph authoring

PARADIGMS = (
    PARA_BREP,
    PARA_CSG,
    PARA_FREP,
    PARA_FUNCTIONAL,
    PARA_MESH,
    PARA_SDF,
    PARA_SKETCH,
    PARA_TRANSPILE,
    PARA_VISUAL,
)

# Geometry kernels.
K_OCCT = "occt"  # OpenCascade (B-rep, C++). The list's headline recommendation.
K_CARVE = "carve"  # Carve mesh boolean engine (used by AngelCAD)
K_CGAL = "cgal"  # CGAL Nef polyhedra (OpenSCAD's classic backend)
K_MANIFOLD = "manifold"  # elalish/manifold mesh library
K_LIBFIVE = "libfive"  # libfive F-rep kernel
K_CUSTOM = "custom"  # the project ships its own kernel
K_NONE = "none"  # no geometry kernel at all (transpilers, hubs, editors)

KERNELS = (K_CARVE, K_CGAL, K_CUSTOM, K_LIBFIVE, K_MANIFOLD, K_NONE, K_OCCT, UNKNOWN)

# Maturity, as characterised by the list.
MAT_MATURE = "mature"
MAT_ACTIVE = "active"
MAT_EARLY = "early"  # the list says "early-stage project" (Fornjot)
MAT_UNMAINTAINED = "unmaintained"  # the list says "Looks to be unmaintained" (BlocksCAD)


class UnknownSystem(KeyError):
    """Raised when a system name/alias is not in the catalogue."""


@dataclass(frozen=True)
class SystemSpec:
    """One code-CAD system in the ecosystem catalogue."""

    name: str
    display_name: str
    category: str
    host_language: str  # language you write models in ("custom-dsl" for own DSL)
    paradigms: Tuple[str, ...]
    kernel: str
    representation: str  # what the kernel stores: "brep" | "mesh" | "implicit" | ...
    license: str  # verbatim from the curated list where stated
    formats_in: Tuple[str, ...] = ()  # empty tuple == not stated / unknown
    formats_out: Tuple[str, ...] = ()
    scripting: bool = True
    gui: bool = False
    online_editor: bool = False  # the list records this per entry
    maturity: str = MAT_ACTIVE
    niche: str = ""
    notes: str = ""
    aliases: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_language(self) -> bool:
        return self.category == CAT_LANGUAGE

    @property
    def is_library(self) -> bool:
        return self.category == CAT_LIBRARY

    @property
    def kernel_free(self) -> bool:
        return self.kernel == K_NONE

    def has_paradigm(self, paradigm: str) -> bool:
        return paradigm in self.paradigms

    def exports(self, fmt: str) -> bool:
        return fmt.lower().lstrip(".") in self.formats_out

    def imports(self, fmt: str) -> bool:
        return fmt.lower().lstrip(".") in self.formats_in

    def as_row(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "category": self.category,
            "host_language": self.host_language,
            "paradigms": list(self.paradigms),
            "kernel": self.kernel,
            "representation": self.representation,
            "license": self.license,
            "formats_in": list(self.formats_in),
            "formats_out": list(self.formats_out),
            "gui": self.gui,
            "online_editor": self.online_editor,
            "maturity": self.maturity,
            "niche": self.niche,
        }


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------
# Licenses are quoted as the curated list writes them. Empty format tuples mean
# the list does not state them and they are not certain -- NOT "none".

_SPECS: Tuple[SystemSpec, ...] = (
    # --- special mentions -------------------------------------------------
    SystemSpec(
        name="openscad",
        display_name="OpenSCAD",
        category=CAT_LANGUAGE,
        host_language="custom-dsl",
        paradigms=(PARA_CSG, PARA_MESH, PARA_FUNCTIONAL),
        kernel=K_CGAL,
        representation="mesh",
        license="GPL-2",
        formats_in=("stl", "dxf", "svg", "off", "amf", "3mf"),
        formats_out=("stl", "off", "amf", "3mf", "dxf", "svg"),
        online_editor=True,
        maturity=MAT_MATURE,
        niche="the OG code-CAD; safest first choice, huge corpus of examples",
        notes=(
            "List: own language means no package manager and awkward function "
            "definitions; performance degrades on complex parts; mesh kernel "
            "limits use beyond 3d printing. The experimental 'manifold' backend "
            "gives a large speedup."
        ),
        aliases=("scad",),
    ),
    SystemSpec(
        name="opencascade",
        display_name="OpenCascade (OCCT)",
        category=CAT_KERNEL,
        host_language="cpp",
        paradigms=(PARA_BREP, PARA_CSG),
        kernel=K_OCCT,
        representation="brep",
        license="LGPL-2.1",
        formats_in=("step", "iges", "brep", "stl"),
        formats_out=("step", "iges", "brep", "stl"),
        maturity=MAT_MATURE,
        niche="the mature open-source B-rep kernel that most serious code-CAD wraps",
        notes="List: 'a c++ library that a number of the projects below wrap'.",
        aliases=("occt", "oce"),
    ),
    # --- alphabetical body ------------------------------------------------
    SystemSpec(
        name="angelcad",
        display_name="AngelCAD",
        category=CAT_LANGUAGE,
        host_language="angelscript",
        paradigms=(PARA_CSG, PARA_MESH),
        kernel=K_CARVE,
        representation="mesh",
        license="GPL-2 or GPL-3",
        formats_out=("amf",),
        niche="fast mesh booleans; C-like scripting with variables, functions, classes",
        notes=(
            "List: embedded general-purpose CSG scripting via AngelScript; boolean "
            "engine is Carve, 'many times faster than other mesh-based systems'; "
            "can run OpenSCAD script for interoperability; DXF import announced."
        ),
    ),
    SystemSpec(
        name="bitbybit",
        display_name="bitbybit",
        category=CAT_LIBRARY,
        host_language="typescript",
        paradigms=(PARA_VISUAL, PARA_CSG),
        kernel=UNKNOWN,
        representation=UNKNOWN,
        license="MIT",
        gui=True,
        online_editor=True,
        niche="node editor plus a TypeScript code interface in the same browser app",
    ),
    SystemSpec(
        name="build123d",
        display_name="build123d",
        category=CAT_LIBRARY,
        host_language="python",
        paradigms=(PARA_BREP, PARA_CSG, PARA_SKETCH),
        kernel=K_OCCT,
        representation="brep",
        license="Apache-2.0",
        formats_out=("step", "stl"),
        niche="Python B-rep with builder/algebra APIs; CadQuery-adjacent",
        notes=(
            "Not written up in the curated list's prose, but shipped as one of the "
            "birdhouse reference implementations (birdhouse/build123d.py)."
        ),
    ),
    SystemSpec(
        name="cadhub",
        display_name="CadHub",
        category=CAT_PLATFORM,
        host_language="multi",
        paradigms=(),
        kernel=K_NONE,
        representation="none",
        license="GPL-3",
        online_editor=True,
        gui=True,
        niche="community hub / online IDE; integrates OpenSCAD, CadQuery, JSCAD, curv",
        notes="Maintainer of this curated list. 'Codepen crossed with a thing repository'.",
    ),
    SystemSpec(
        name="cadquery",
        display_name="CadQuery",
        category=CAT_LIBRARY,
        host_language="python",
        paradigms=(PARA_BREP, PARA_CSG, PARA_SKETCH),
        kernel=K_OCCT,
        representation="brep",
        license="Apache-2.0",
        formats_in=("step", "brep"),
        formats_out=("step", "stl", "dxf", "svg", "amf", "3mf", "brep"),
        online_editor=True,
        maturity=MAT_MATURE,
        niche="design-intent Python B-rep; selector language for edges/faces (fillets)",
        notes=(
            "List: wraps and extends OpenCascade; API mirrors how you would describe "
            "the object to a human; selectors, e.g. .edges('|Z').fillet(0.125). One of "
            "the list's four recommended B-rep packages."
        ),
        aliases=("cq",),
    ),
    SystemSpec(
        name="cascadestudio",
        display_name="CascadeStudio",
        category=CAT_LIBRARY,
        host_language="javascript",
        paradigms=(PARA_BREP, PARA_CSG),
        kernel=K_OCCT,
        representation="brep",
        license="MIT",
        gui=True,
        online_editor=True,
        niche="OpenCascade in the browser via WebAssembly",
        notes="One of the list's four recommended B-rep packages.",
    ),
    SystemSpec(
        name="curv",
        display_name="Curv",
        category=CAT_LANGUAGE,
        host_language="custom-dsl",
        paradigms=(PARA_SDF, PARA_FREP, PARA_FUNCTIONAL),
        kernel=K_CUSTOM,
        representation="implicit",
        license="Apache-2.0",
        formats_out=("stl", "obj"),
        maturity=MAT_ACTIVE,
        niche="mathematical 3D art: full colour, animation, 2D+3D; inspired by OpenSCAD and shadertoy",
        notes="List: 'If you want to make 3D art, Curv is specifically trying to hit that niche.'",
    ),
    SystemSpec(
        name="declaracad",
        display_name="DeclaraCAD",
        category=CAT_LIBRARY,
        host_language="enaml",
        paradigms=(PARA_BREP, PARA_CSG),
        kernel=K_OCCT,
        representation="brep",
        license=UNKNOWN,
        gui=True,
        niche="declarative (enaml) B-rep modelling on OpenCascade",
        notes=(
            "Named in the list's recommendation paragraph as one of the four B-rep "
            "packages, and shipped as a birdhouse reference implementation "
            "(birdhouse/DeclaraCAD.enaml); it has no entry section of its own."
        ),
    ),
    SystemSpec(
        name="fornjot",
        display_name="Fornjot",
        category=CAT_LIBRARY,
        host_language="rust",
        paradigms=(PARA_BREP,),
        kernel=K_CUSTOM,
        representation="brep",
        license="Zero-Clause BSD",
        maturity=MAT_EARLY,
        niche="next-generation code-first CAD in Rust; own B-rep kernel",
        notes="List: explicitly 'an early-stage project'.",
    ),
    SystemSpec(
        name="freecad",
        display_name="FreeCAD",
        category=CAT_APPLICATION,
        host_language="python",
        paradigms=(PARA_SKETCH, PARA_BREP, PARA_CSG),
        kernel=K_OCCT,
        representation="brep",
        license="LGPLv2",
        formats_in=("step", "iges", "brep", "stl", "dxf", "svg"),
        formats_out=("step", "iges", "brep", "stl", "dxf", "svg"),
        gui=True,
        maturity=MAT_MATURE,
        niche="traditional GUI MCAD with Python scripting; best interoperability",
        notes=(
            "List: scripts both the model and the GUI; ships an OpenSCAD workbench and "
            "an external CadQuery workbench, 'making it the best in this list at "
            "interoperability'. Uses OpenCascade under the hood."
        ),
    ),
    SystemSpec(
        name="implicitcad",
        display_name="ImplicitCAD",
        category=CAT_LANGUAGE,
        host_language="haskell",
        paradigms=(PARA_FREP, PARA_SDF, PARA_CSG, PARA_FUNCTIONAL),
        kernel=K_CUSTOM,
        representation="implicit",
        license="AGPL-3",
        formats_out=("stl", "svg"),
        online_editor=True,
        niche="OpenSCAD-like language on an implicit kernel; models can also be written in Haskell",
        notes=(
            "List: part of an 'almost stack' with ExplicitCAD (GUI) and HSlice (STL slicer)."
        ),
    ),
    SystemSpec(
        name="jscad",
        display_name="JSCAD",
        category=CAT_LIBRARY,
        host_language="javascript",
        paradigms=(PARA_CSG, PARA_MESH),
        kernel=K_CUSTOM,
        representation="mesh",
        license="MIT",
        formats_out=("stl", "dxf", "svg"),
        online_editor=True,
        maturity=MAT_MATURE,
        niche="precise models for 3D printing in plain JavaScript; 2D and 3D",
        notes=(
            "List: available as website, CLI for backend processing, user application "
            "and a set of libraries. Formerly OpenJSCAD. Exports 'STL, DXF, SVG, etc'."
        ),
        aliases=("openjscad",),
    ),
    SystemSpec(
        name="libfive",
        display_name="libfive",
        category=CAT_LIBRARY,
        host_language="scheme",
        paradigms=(PARA_FREP, PARA_SDF, PARA_CSG),
        kernel=K_LIBFIVE,
        representation="implicit",
        license="Mozilla Public License 2.0 and GPL-2 or later",
        gui=True,
        niche="parametric and procedural solid modelling on an F-rep kernel; Lisp/Scheme front end",
    ),
    SystemSpec(
        name="manifold",
        display_name="manifold",
        category=CAT_KERNEL,
        host_language="cpp",
        paradigms=(PARA_CSG, PARA_MESH, PARA_SDF),
        kernel=K_MANIFOLD,
        representation="mesh",
        license="Apache-2.0",
        formats_out=("gltf", "3mf"),
        online_editor=True,
        maturity=MAT_ACTIVE,
        niche="fast, robust manifold mesh library; the modern mesh boolean engine",
        notes=(
            "List: C++ API plus C, Python and JavaScript/TypeScript bindings; used as "
            "OpenSCAD's experimental backend with commonly >100x speedup; the online "
            "editor exports glTF and 3MF ('3mf preserves manifoldness')."
        ),
    ),
    SystemSpec(
        name="pythonocc",
        display_name="pythonOCC",
        category=CAT_LIBRARY,
        host_language="python",
        paradigms=(PARA_BREP, PARA_CSG),
        kernel=K_OCCT,
        representation="brep",
        license="LGPL-3",
        formats_in=("step", "iges", "brep", "stl"),
        formats_out=("step", "iges", "brep", "stl"),
        online_editor=True,
        maturity=MAT_MATURE,
        niche="thin Python bindings over the whole OpenCascade API",
        notes="One of the list's four recommended B-rep packages.",
        aliases=("pythonocc-core", "occ"),
    ),
    SystemSpec(
        name="rapcad",
        display_name="RapCAD",
        category=CAT_LANGUAGE,
        host_language="custom-dsl",
        paradigms=(PARA_CSG, PARA_MESH),
        kernel=UNKNOWN,
        representation="mesh",
        license="GPL-3",
        gui=True,
        niche="OpenSCAD-like but procedural (mutable variables) and exactly-rounded",
        notes=(
            "List: key differences from OpenSCAD are procedural rather than functional "
            "style and arbitrary-precision arithmetic throughout, so no unexpected "
            "double/float rounding errors. Ships a RapCAD/OpenSCAD/ImplicitCAD feature "
            "matrix (doc/feature_matrix.asciidoc)."
        ),
    ),
    SystemSpec(
        name="replicad",
        display_name="replicad",
        category=CAT_LIBRARY,
        host_language="javascript",
        paradigms=(PARA_BREP, PARA_CSG, PARA_SKETCH),
        kernel=K_OCCT,
        representation="brep",
        license="AGPL",
        formats_out=("step", "stl"),
        online_editor=True,
        niche="browser-based B-rep with a CadQuery-inspired API, on opencascade.js",
    ),
    SystemSpec(
        name="scad-clj",
        display_name="scad-clj",
        category=CAT_LIBRARY,
        host_language="clojure",
        paradigms=(PARA_TRANSPILE, PARA_CSG, PARA_FUNCTIONAL),
        kernel=K_NONE,
        representation="none",
        license="EPL-1.0",
        formats_out=("scad",),
        niche="OpenSCAD DSL embedded in Clojure; emits .scad, geometry is OpenSCAD's job",
        notes="List: 'Functions generally mirror OpenSCAD, with a couple of notable exceptions.'",
    ),
    SystemSpec(
        name="scad-hs",
        display_name="scad-hs",
        category=CAT_LIBRARY,
        host_language="haskell",
        paradigms=(PARA_TRANSPILE, PARA_CSG, PARA_FUNCTIONAL),
        kernel=K_NONE,
        representation="none",
        license="BSD-3-Clause License",
        formats_out=("scad",),
        niche="OpenSCAD DSL embedded in Haskell; same author as scad-clj",
    ),
    SystemSpec(
        name="sdf-csg",
        display_name="sdf-csg",
        category=CAT_LIBRARY,
        host_language="javascript",
        paradigms=(PARA_SDF, PARA_FREP, PARA_CSG),
        kernel=K_CUSTOM,
        representation="implicit",
        license="The Unlicense",
        niche="mesh generation from SDFs plus CSG ops, after Inigo Quilez's 3D SDF article",
    ),
    SystemSpec(
        name="sdfx",
        display_name="sdfx",
        category=CAT_LIBRARY,
        host_language="go",
        paradigms=(PARA_SDF, PARA_FREP, PARA_CSG),
        kernel=K_CUSTOM,
        representation="implicit",
        license="MIT",
        formats_out=("stl",),
        niche="Go code-CAD on an SDF kernel; fillets and chamfers; ships a standard library (obj/)",
        notes="List: notes @soypat's rewrite (github.com/soypat/sdf) as worth checking out.",
    ),
    SystemSpec(
        name="solidpython",
        display_name="SolidPython",
        category=CAT_LIBRARY,
        host_language="python",
        paradigms=(PARA_TRANSPILE, PARA_CSG),
        kernel=K_NONE,
        representation="none",
        license="GPL-2 or later",
        formats_out=("scad",),
        niche="Python front end that emits OpenSCAD code; geometry delegated to OpenSCAD",
    ),
    SystemSpec(
        name="tovero",
        display_name="Tovero",
        category=CAT_LIBRARY,
        host_language="common-lisp",
        paradigms=(PARA_FREP, PARA_SDF, PARA_CSG),
        kernel=K_LIBFIVE,
        representation="implicit",
        license="LGPL-2.1 or later and GPL-2 or later",
        gui=True,
        niche="Common Lisp binding of libfive with a REPL viewer; composes with Clive scene graph",
    ),
    # --- node editors / other (the list's own trailing section) -----------
    SystemSpec(
        name="blockscad",
        display_name="BlocksCAD",
        category=CAT_NODE_EDITOR,
        host_language="blocks",
        paradigms=(PARA_VISUAL, PARA_CSG),
        kernel=UNKNOWN,
        representation=UNKNOWN,
        license="GPL-3",
        scripting=False,
        gui=True,
        online_editor=True,
        maturity=MAT_UNMAINTAINED,
        niche="block-based CAD for education",
        notes="List: 'Looks to be unmaintained.'",
    ),
    SystemSpec(
        name="dynamo",
        display_name="Dynamo",
        category=CAT_NODE_EDITOR,
        host_language="visual",
        paradigms=(PARA_VISUAL,),
        kernel=UNKNOWN,
        representation=UNKNOWN,
        license="Apache-2.0",
        scripting=False,
        gui=True,
        niche="visual programming for AEC; used with Autodesk software, and with FreeCAD via DynFreeCAD",
    ),
    SystemSpec(
        name="makecode",
        display_name="MakeCode",
        category=CAT_NODE_EDITOR,
        host_language="blocks",
        paradigms=(PARA_VISUAL, PARA_CSG, PARA_MESH),
        kernel=UNKNOWN,
        representation=UNKNOWN,
        license="MIT",
        scripting=False,
        gui=True,
        online_editor=True,
        niche="block editor tuned for functional 3D prints: stacking/layout helpers, styled edges (fillets), fast hull ('wrap shapes')",
    ),
    SystemSpec(
        name="sverchok",
        display_name="Sverchok",
        category=CAT_NODE_EDITOR,
        host_language="visual",
        paradigms=(PARA_VISUAL, PARA_MESH),
        kernel=UNKNOWN,
        representation="mesh",
        license="GPL-3",
        scripting=False,
        gui=True,
        niche="parametric node geometry inside Blender; architecture-oriented",
    ),
)

_BY_NAME: Dict[str, SystemSpec] = {}
for _spec in _SPECS:
    _BY_NAME[_spec.name] = _spec
    for _alias in _spec.aliases:
        _BY_NAME[_alias] = _spec


# ---------------------------------------------------------------------------
# Representation taxonomy (the list's editorial "Which one should you use?")
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepresentationNote:
    """The curated list's stated trade-offs for one representation."""

    representation: str
    strengths: Tuple[str, ...]
    limitations: Tuple[str, ...]


REPRESENTATION_NOTES: Dict[str, RepresentationNote] = {
    "brep": RepresentationNote(
        representation="brep",
        strengths=(
            "exact boundary representation, the list's headline recommendation",
            "future-proof: does not limit the application domain",
            "supports operations CSG makes awkward, e.g. internal fillets",
        ),
        limitations=(
            "the only mature open-source B-rep kernel is OpenCascade (C++, LGPL-2.1)",
        ),
    ),
    "mesh": RepresentationNote(
        representation="mesh",
        strengths=("simple, fast enough for 3d printing, ubiquitous tooling",),
        limitations=(
            "limits use beyond 3d-printed parts",
            "the list names optics and injection moulding as problem domains",
        ),
    ),
    "implicit": RepresentationNote(
        representation="implicit",
        strengths=(
            "natural fillets/chamfers and smooth blends (sdfx does fillets and chamfers)",
            "good fit for procedural and generative art (Curv's niche)",
        ),
        limitations=(
            "must be meshed before most downstream use",
        ),
    ),
    "none": RepresentationNote(
        representation="none",
        strengths=("transpilers inherit the target system's geometry for free",),
        limitations=("cannot evaluate geometry themselves; need the target toolchain",),
    ),
}

# Paradigm-level caveat the list states about the CSG mindset.
CSG_CAVEAT = (
    "Most code-CAD tools are plagued with a CSG mindset (unions, subtractions and "
    "intersections of primitives). It is an inherently limited paradigm: internal "
    "fillets, which matter for reducing stress concentrations, become very difficult."
)

# The list's explicit recommendation: prefer a B-rep (OpenCascade) package.
RECOMMENDED_BREP_SYSTEMS: Tuple[str, ...] = (
    "cadquery",
    "cascadestudio",
    "declaracad",
    "pythonocc",
)

# The reference part implemented in each system, shipped in the list's repo.
BIRDHOUSE_IMPLEMENTATIONS: Tuple[str, ...] = (
    "build123d",
    "cadquery",
    "cascadestudio",
    "declaracad",
    "freecad",
    "jscad",
    "openscad",
    "sdfx",
)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def system_names() -> List[str]:
    """Canonical system names, sorted."""
    return sorted(spec.name for spec in _SPECS)


def get(name: str) -> SystemSpec:
    """Look a system up by name or alias (case-insensitive)."""
    try:
        return _BY_NAME[name.strip().lower()]
    except KeyError as exc:
        raise UnknownSystem(name) from exc


def has(name: str) -> bool:
    return name.strip().lower() in _BY_NAME


def all_systems() -> List[SystemSpec]:
    """Every spec, sorted by name."""
    return [get(n) for n in system_names()]


def by_paradigm(paradigm: str) -> List[str]:
    if paradigm not in PARADIGMS:
        raise ValueError("unknown paradigm: %s" % paradigm)
    return sorted(s.name for s in _SPECS if s.has_paradigm(paradigm))


def by_kernel(kernel: str) -> List[str]:
    if kernel not in KERNELS:
        raise ValueError("unknown kernel: %s" % kernel)
    return sorted(s.name for s in _SPECS if s.kernel == kernel)


def by_host_language(language: str) -> List[str]:
    key = language.strip().lower()
    return sorted(s.name for s in _SPECS if s.host_language == key)


def by_category(category: str) -> List[str]:
    if category not in CATEGORIES:
        raise ValueError("unknown category: %s" % category)
    return sorted(s.name for s in _SPECS if s.category == category)


def exporters_of(fmt: str) -> List[str]:
    """Systems the catalogue records as able to export ``fmt`` (e.g. 'step')."""
    return sorted(s.name for s in _SPECS if s.exports(fmt))


def importers_of(fmt: str) -> List[str]:
    return sorted(s.name for s in _SPECS if s.imports(fmt))


def sdf_systems() -> List[str]:
    """Systems whose modelling paradigm includes signed distance functions."""
    return by_paradigm(PARA_SDF)


def brep_systems() -> List[str]:
    return by_paradigm(PARA_BREP)


def csg_systems() -> List[str]:
    return by_paradigm(PARA_CSG)


def kernel_free_systems() -> List[str]:
    """Systems with no geometry kernel of their own (transpilers, hubs)."""
    return sorted(s.name for s in _SPECS if s.kernel_free)


def occt_based_systems() -> List[str]:
    return by_kernel(K_OCCT)


def host_languages() -> List[str]:
    return sorted({s.host_language for s in _SPECS})


def known_formats_out() -> List[str]:
    out = set()
    for spec in _SPECS:
        out.update(spec.formats_out)
    return sorted(out)


def unknown_attributes(name: str) -> List[str]:
    """Attributes the catalogue deliberately leaves unknown for this system.

    Lets a caller see where the curated list is silent instead of trusting a
    fabricated value.
    """
    spec = get(name)
    gaps: List[str] = []
    if spec.kernel == UNKNOWN:
        gaps.append("kernel")
    if spec.representation == UNKNOWN:
        gaps.append("representation")
    if spec.license == UNKNOWN:
        gaps.append("license")
    if not spec.formats_out:
        gaps.append("formats_out")
    if not spec.formats_in:
        gaps.append("formats_in")
    return sorted(gaps)


def select(
    *,
    paradigm: Optional[str] = None,
    kernel: Optional[str] = None,
    host_language: Optional[str] = None,
    category: Optional[str] = None,
    exports: Optional[str] = None,
    imports: Optional[str] = None,
    scripting: Optional[bool] = None,
    online_editor: Optional[bool] = None,
) -> List[str]:
    """Names of systems matching every supplied constraint, sorted."""
    names: List[str] = []
    for spec in _SPECS:
        if paradigm is not None and not spec.has_paradigm(paradigm):
            continue
        if kernel is not None and spec.kernel != kernel:
            continue
        if host_language is not None and spec.host_language != host_language.strip().lower():
            continue
        if category is not None and spec.category != category:
            continue
        if exports is not None and not spec.exports(exports):
            continue
        if imports is not None and not spec.imports(imports):
            continue
        if scripting is not None and spec.scripting is not scripting:
            continue
        if online_editor is not None and spec.online_editor is not online_editor:
            continue
        names.append(spec.name)
    return sorted(names)


def catalogue_matrix() -> List[Dict[str, object]]:
    """The whole table as flat rows, sorted by name."""
    return [spec.as_row() for spec in all_systems()]
