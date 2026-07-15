"""assembly -- deterministic checks over placed multi-part assemblies.

Assembly-level verification treats parts as already authored and *placed* (via
translation/rotation) and asks questions about the arrangement rather than the
geometry of any single part: do two parts interfere, and if so what is the
cheapest translation that clears them?

The initial contents are mined from Zoo-adjacent CADCLAW's interference gate,
reduced to its deterministic, stdlib-only core: axis-aligned bounding-box
overlap detection with a minimum-clearance fix-vector suggestion.
"""
