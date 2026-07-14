"""CAD GUI grounding: the self-labelling corpus, and the benchmark built on it.

* :mod:`corpus`  — generates ``(screenshot, description, (x, y), verified=True)``
  pairs by projecting a B-rep we own into a known camera and letting the
  APPLICATION adjudicate every click. No human, no heuristic, no vision model.
* :mod:`cadspot` — a ScreenSpot-style grounding benchmark for a CAD application,
  split by region (ribbon / dialog / feature tree / **3D viewport**).

Nothing is re-exported: both modules import their dependencies lazily and both
degrade to a clean "unavailable" when FreeCAD or the CUA extras are absent.
"""
