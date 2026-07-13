"""MUSE assemblability scorer (design-intent alignment, Assemblability pillar).

Deterministic re-encoding of the MUSE benchmark's Assemblability pillar (Dong et
al., "MUSE: Benchmarking Manufacturable, Functional, and Assemblable
Text-to-CAD Generation"). The pillar has two binary sub-criteria (Table 8):

  * Assembly-ready -- does the generated model preserve the intended component
    topology? The inferred Physical Assembly Graph G'=(V',E') must be
    isomorphic to the target graph G=(V,E) (Definition 1, topological validity
    Phi(C) ~= G), including the central-hub structure.
  * Connectable    -- are the physical joints placed, oriented and constrained
    correctly? Each mating interface must match the required joint type and its
    degrees-of-freedom behaviour (engineering knowledge Table 7), have the
    correct assembly direction, and a part-fit clearance inside the process
    tolerance band (no interpenetration, no illegal floating).

This is distinct from ``verifiers/assembly.py`` / ``verifiers/interference.py``
and ``quality/assembly_readiness.py``: here the two MUSE sub-criteria are
computed from an injected assembly graph plus mating-interface descriptions,
grounded in the paper's Connection-Method table.

No wall clock, no randomness.
"""

from __future__ import annotations

from itertools import permutations

from harnesscad.eval.bench.protocols.muse_manufacturability import process_tolerance

# --- Engineering Knowledge Table 7: Connection Methods ------------------------
# constrained_dof: number of the six rigid-body DoF the joint removes when
# properly fitted. "None" is the standalone / one-piece case (no joint).
JOINTS = {
    "Interlocking": {"constrained_dof": 6, "elastic": False, "rotational_axis": False},
    "Snap-fit": {"constrained_dof": 6, "elastic": True, "rotational_axis": False},
    "Nailing": {"constrained_dof": 4, "elastic": False, "rotational_axis": True},
    "Pivot": {"constrained_dof": 5, "elastic": False, "rotational_axis": True},
    "Bonding": {"constrained_dof": 6, "elastic": False, "rotational_axis": False},
    "None": {"constrained_dof": 0, "elastic": False, "rotational_axis": False},
}

# Normalisation rules from the rubric prompt ("use Interlocking instead of
# Mortise & Tenon; use Nailing / Pinning instead of dowel joint").
_JOINT_ALIASES = {
    "mortise & tenon": "Interlocking", "mortise and tenon": "Interlocking",
    "interlocking": "Interlocking",
    "snap-fit": "Snap-fit", "snap fit": "Snap-fit", "snapfit": "Snap-fit",
    "nailing": "Nailing", "nailing / pinning": "Nailing", "pinning": "Nailing",
    "dowel": "Nailing", "dowel joint": "Nailing", "nail": "Nailing",
    "pivot": "Pivot", "hinge": "Pivot", "pivot / hinge": "Pivot",
    "bonding": "Bonding", "glue": "Bonding", "adhesive": "Bonding",
    "none": "None", "n/a": "None", "standalone": "None",
}


def normalize_joint(name):
    """Normalise a joint-type name to the canonical Table-7 vocabulary."""
    key = str(name).strip().lower()
    if key in _JOINT_ALIASES:
        return _JOINT_ALIASES[key]
    if name in JOINTS:
        return name
    raise ValueError("unknown joint type: %r" % (name,))


def joint_constrained_dof(name):
    """Degrees of freedom removed by a (canonicalised) joint (Table 7)."""
    return JOINTS[normalize_joint(name)]["constrained_dof"]


def _canonical_edges(edges):
    return frozenset(frozenset((str(a), str(b))) for a, b in edges)


def graphs_isomorphic(target, inferred):
    """Test isomorphism of two undirected graphs given as (nodes, edges).

    Each graph is a pair (nodes, edges); nodes is an iterable of hashable node
    ids, edges is an iterable of unordered id pairs. Matching is by structure
    (an unlabelled isomorphism over the small assembly graphs MUSE uses).
    Returns True iff a bijection preserving adjacency exists.
    """
    tn = [str(n) for n in target[0]]
    inn = [str(n) for n in inferred[0]]
    if len(tn) != len(inn):
        return False
    te = _canonical_edges(target[1])
    ie = _canonical_edges(inferred[1])
    if len(te) != len(ie):
        return False
    # Degree-sequence prefilter.
    if sorted(_degrees(tn, te).values()) != sorted(_degrees(inn, ie).values()):
        return False
    inferred_edges = ie
    for perm in permutations(inn):
        mapping = dict(zip(tn, perm))
        mapped = frozenset(
            frozenset((mapping[a], mapping[b])) for a, b in
            (tuple(e) if len(e) == 2 else (next(iter(e)), next(iter(e)))
             for e in te))
        if mapped == inferred_edges:
            return True
    return False


def _degrees(nodes, edges):
    deg = {n: 0 for n in nodes}
    for e in edges:
        for n in e:
            if n in deg:
                deg[n] += 1
    return deg


def hub_node(nodes, edges):
    """Return the unique maximum-degree node (central hub), or None if tied."""
    deg = _degrees([str(n) for n in nodes], _canonical_edges(edges))
    if not deg:
        return None
    top = max(deg.values())
    hubs = [n for n, d in deg.items() if d == top]
    return hubs[0] if len(hubs) == 1 else None


def score_assembly_ready(design):
    """Binary Assembly-ready sub-criterion for one injected design.

    design keys:
      target_graph   : (nodes, edges) -- the intended Physical Assembly Graph.
      inferred_graph : (nodes, edges) -- graph inferred from the generated model.
      require_hub_match : bool (default True) -- also require the central hub
                          (max-degree node) count to agree.
    Returns {"assembly_ready": 0/1, "reasons": (...)}.
    """
    target = design["target_graph"]
    inferred = design["inferred_graph"]
    reasons = []

    if len(list(target[0])) != len(list(inferred[0])):
        reasons.append("node_count_mismatch")
    if not graphs_isomorphic(target, inferred):
        reasons.append("graph_not_isomorphic")

    if design.get("require_hub_match", True):
        th = hub_node(*target)
        ih = hub_node(*inferred)
        # Compare presence/absence of a unique hub, not the label.
        if (th is None) != (ih is None):
            reasons.append("hub_structure_mismatch")

    return {"assembly_ready": 0 if reasons else 1, "reasons": tuple(reasons)}


def score_connectable(design):
    """Binary Connectable sub-criterion for one injected design.

    design keys:
      process    : manufacturing process name (for the tolerance band).
      interfaces : iterable of dicts, one per mating edge, with keys:
          name              : interface label.
          required_joint    : required joint type (Table 7 vocabulary/aliases).
          actual_joint      : realised joint type.
          required_direction: assembly direction vector, e.g. (0, 0, 1) (opt).
          actual_direction  : realised assembly direction (opt).
          clearance         : part-fit gap in mm (negative = interpenetration).
    An interface is connectable when the realised joint matches the required
    type, the assembly direction matches (if given), and the clearance lies in
    [tol_min, tol_max] for the process (no interpenetration, no floating).
    Returns {"connectable": 0/1, "reasons": (...)}.
    """
    process = design.get("process")
    tol_min = tol_max = None
    if process is not None:
        tol_min, tol_max = process_tolerance(process)
    reasons = []

    for iface in design.get("interfaces", ()):
        name = iface.get("name", "?")
        req = normalize_joint(iface["required_joint"])
        act = normalize_joint(iface["actual_joint"])
        if req != act:
            reasons.append("joint_type_mismatch:%s" % name)
            # A wrong joint type also means the DoF behaviour is wrong.
        req_dir = iface.get("required_direction")
        act_dir = iface.get("actual_direction")
        if req_dir is not None and act_dir is not None:
            if tuple(req_dir) != tuple(act_dir):
                reasons.append("wrong_assembly_direction:%s" % name)
        clr = iface.get("clearance")
        if clr is not None and req != "None":
            if clr < 0:
                reasons.append("interpenetration:%s" % name)
            elif tol_min is not None and clr < tol_min:
                reasons.append("illegal_fusion:%s" % name)
            elif tol_max is not None and clr > tol_max:
                reasons.append("floating_gap:%s" % name)

    return {"connectable": 0 if reasons else 1, "reasons": tuple(reasons)}


def muse_assemblability(design):
    """Full Assemblability pillar score for one injected design.

    Returns the two binary sub-criteria, their average (MUSE Table 3 pillar
    score), and the merged reason list.
    """
    a = score_assembly_ready(design)
    c = score_connectable(design)
    average = (a["assembly_ready"] + c["connectable"]) / 2.0
    return {"assembly_ready": a["assembly_ready"], "connectable": c["connectable"],
            "average": average,
            "reasons": tuple(a["reasons"]) + tuple(c["reasons"])}
