"""catalogue — the computer-use field map, as a structured reference module.

The ACU repo ("Awesome Agents for Computer Use") is a curated list: benchmarks,
grounding models, datasets, safety papers, frameworks. A prose list is not
queryable, so its content is captured here as data with query helpers, kept for
one practical reason: to place HarnessCAD's grounding work on the map and to make
the field's blind spot legible.

**The blind spot, stated as data.** Every benchmark below scores agents on WEB
pages, MOBILE apps, or generic OS chrome — all of which have an accessibility tree
to harvest ground truth from. NONE targets a 3D CAD viewport, which has no such
tree (see eval/grounding/corpus). So :func:`cad_gap` returns the benchmarks and
grounding models whose harvesting method cannot transfer to a CAD viewport, which
is precisely the gap CADSpot / the self-labelling corpus fills. That is the
finding to carry forward: the reason no CAD-viewport grounding set exists is not
neglect, it is that you cannot scrape one.

Pure data + selectors, stdlib only. The lists are a snapshot for reference, not a
live index, and are intentionally not exhaustive beyond the load-bearing entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Benchmark:
    name: str
    year: str
    domain: str            # web | mobile | os-desktop | tool-api | grounding
    note: str
    has_accessibility_tree: bool  # can ground truth be harvested from an a11y DOM?


@dataclass(frozen=True)
class GroundingModel:
    name: str
    year: str
    note: str
    outputs_bbox: bool     # bbox string vs normalised point
    trained_on_a11y_scrape: bool  # was its data harvested from a11y trees?


@dataclass(frozen=True)
class Dataset:
    name: str
    year: str
    domain: str
    note: str


@dataclass(frozen=True)
class SafetyPaper:
    name: str
    year: str
    note: str


# --- benchmarks -------------------------------------------------------------
BENCHMARKS: Tuple[Benchmark, ...] = (
    Benchmark("OSWorld", "2024", "os-desktop",
              "open-ended tasks in real computer environments; the reference "
              "desktop-agent benchmark", True),
    Benchmark("Windows Agent Arena", "2024", "os-desktop",
              "multi-modal OS agents at scale on Windows", True),
    Benchmark("ScreenSpot", "2024", "grounding",
              "GUI grounding accuracy (introduced with SeeClick); click-location "
              "on screenshots", True),
    Benchmark("tau-bench", "2024", "tool-api",
              "tool-agent-user interaction and policy compliance; SOTA <50% "
              "success, <25% pass^8 consistency", False),
    Benchmark("AndroidWorld", "2024", "mobile",
              "dynamic benchmarking for autonomous Android agents", True),
    Benchmark("Spider2-V", "2024", "os-desktop",
              "automating data-science / engineering workflows", True),
    Benchmark("AppWorld", "2024", "tool-api",
              "750 tasks, 457 APIs across 9 apps; GPT-4o ~49% normal, ~30% "
              "challenge", False),
    Benchmark("VisualWebArena", "2024", "web",
              "multimodal agents on realistic visual web tasks", True),
    Benchmark("Mobile-Env", "2023", "mobile",
              "qualified LLM-GUI interaction benchmarks", True),
    Benchmark("A3", "2025", "mobile",
              "Android Agent Arena; 201 tasks across 21 third-party apps", True),
    Benchmark("MobileAgentBench", "2024", "mobile",
              "efficient benchmark for mobile LLM agents", True),
)

# --- grounding models -------------------------------------------------------
GROUNDING_MODELS: Tuple[GroundingModel, ...] = (
    GroundingModel("OS-Atlas", "2024",
                   "foundation action model for generalist GUI agents; returns "
                   "<|box_start|>..<|box_end|> bboxes (used by os_computer_use)",
                   True, True),
    GroundingModel("ShowUI", "2024",
                   "lightweight vision-language-action GUI model; returns "
                   "normalised [u, v] points", False, True),
    GroundingModel("SeeClick", "2024",
                   "GUI grounding for visual agents; introduced ScreenSpot",
                   True, True),
    GroundingModel("UGround", "2024",
                   "universal visual grounding for GUI agents (OSU-NLP)",
                   True, True),
    GroundingModel("OmniParser", "2024",
                   "pure-vision screen parsing to structured elements (Microsoft)",
                   True, True),
    GroundingModel("Ferret-UI", "2024",
                   "grounded mobile-UI understanding (Apple)", True, True),
    GroundingModel("CogAgent", "2023",
                   "visual language model for GUI agents across PC and Android",
                   True, True),
    GroundingModel("PTA-1", "2024",
                   "270M Florence-2-based element localiser (AskUI)", True, True),
)

# --- datasets ---------------------------------------------------------------
DATASETS: Tuple[Dataset, ...] = (
    Dataset("Mind2Web", "2023", "web", "large-scale web interaction dataset"),
    Dataset("Android in the Wild", "2023", "mobile", "large-scale device control"),
    Dataset("WebShop", "2022", "web", "grounded language agents in web interaction"),
    Dataset("Rico", "2017", "mobile", "mobile-app UI dataset, design-focused"),
    Dataset("UiPad", "2024", "os-desktop",
            "macOS desktop UI with accessibility trees"),
    Dataset("OS-Genesis", "2024", "os-desktop",
            "reverse-task-synthesis GUI trajectories"),
)

# --- safety -----------------------------------------------------------------
SAFETY: Tuple[SafetyPaper, ...] = (
    SafetyPaper("Attacking VLM Computer Agents via Pop-ups", "2024",
                "adversarial pop-ups hijack agent clicks"),
    SafetyPaper("EIA: Environmental Injection Attack", "2024",
                "privacy leakage via injected web content"),
    SafetyPaper("GuardAgent", "2024",
                "a guard agent safeguards an LLM agent via knowledge reasoning"),
)


# --- selectors --------------------------------------------------------------
def benchmarks_for(domain: str) -> List[Benchmark]:
    return [b for b in BENCHMARKS if b.domain == domain]


def find_benchmark(name: str) -> Optional[Benchmark]:
    key = name.lower()
    for b in BENCHMARKS:
        if b.name.lower() == key:
            return b
    return None


def find_grounding_model(name: str) -> Optional[GroundingModel]:
    key = name.lower()
    for m in GROUNDING_MODELS:
        if m.name.lower() == key:
            return m
    return None


def bbox_grounders() -> List[GroundingModel]:
    """Models that return bbox strings (need midpoint extraction, see
    io/cua/coords.parse_point) vs normalised points."""
    return [m for m in GROUNDING_MODELS if m.outputs_bbox]


def cad_gap() -> dict:
    """The field's blind spot, as data: benchmarks and grounding models whose
    ground-truth HARVESTING relies on an accessibility tree, and therefore cannot
    transfer to an a11y-less 3D CAD viewport. This is the gap the self-labelling
    corpus fills -- returned so a report can cite it exactly."""
    return {
        "benchmarks_a11y_dependent": [b.name for b in BENCHMARKS
                                      if b.has_accessibility_tree],
        "grounders_from_a11y_scrape": [m.name for m in GROUNDING_MODELS
                                       if m.trained_on_a11y_scrape],
        "cad_viewport_benchmarks": [],  # none exist
        "finding": ("every listed benchmark and grounding model depends on an "
                    "accessibility tree for ground truth; a 3D CAD viewport is "
                    "one opaque node with no such tree, so neither the models nor "
                    "their data-harvesting method transfers -- which is why no "
                    "CAD-viewport grounding set exists and why one must be "
                    "self-labelled analytically"),
    }


def counts() -> dict:
    return {"benchmarks": len(BENCHMARKS), "grounding_models": len(GROUNDING_MODELS),
            "datasets": len(DATASETS), "safety": len(SAFETY)}
