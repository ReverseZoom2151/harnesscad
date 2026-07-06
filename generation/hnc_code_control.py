"""Neural code-tree serialization and three-level control masking for HNC-CAD.

HNC-CAD (Xu et al., ICML 2023) controls generation by specifying a *tree of neural
codes* at three levels (Solid, Profile, Loop). This module is the deterministic
representation + control layer that sits on top of the codebooks:

* **Code-tree serialization** (paper Sec. 5, "Code Tree Generator"). The code tree is
  flattened by a depth-first traversal with a ``<SEP>`` boundary marking each new
  profile grouping. For a model with one solid, two profiles, and (2, 2) loops the
  paper's order is::

      [S, <SEP>, P, L, L, <SEP>, P, L, L]

  Each element is a one-hot feature whose size is the total number of codes across
  the three codebooks *plus one* for the separator (paper Sec. 5).

* **Three-level control masking** -- HNC-CAD's defining capability, and the thing it
  is criticized for *lacking* by FlexCAD's flat token masking
  (:mod:`reconstruction.flexcad_text`): the user can fix or edit code nodes at *any*
  of the three tree levels, so a single edit propagates through the hierarchy. This
  is distinct from FlexCAD masking, which addresses flat text-token fields of one
  concrete model; here we mask *neural-code nodes* of the abstract code tree. The
  scheme supports:

  - :func:`level_mask` -- fix (freeze) whole levels, leaving the rest to be generated;
  - :func:`edit_code` -- edit a single node's code (Sec. 6.4, "Code Tree Editing":
    loop codes control shape geometry, profile codes control 2D loop dimension /
    positioning, solid codes control extrusion height / 3D combination);
  - :func:`autocomplete_mask` -- from a partial set of known nodes, mark the rest as
    to-be-predicted (Sec. 6.4, "Autocompletion from User Input").

The autoregressive transformers that realize the generation are out of scope; the
tree bookkeeping and masks here are deterministic. Pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# Tree levels (root -> leaf).
SOLID = "solid"
PROFILE = "profile"
LOOP = "loop"
LEVELS = (SOLID, PROFILE, LOOP)

SEP = "<SEP>"


@dataclass(frozen=True)
class CodeNode:
    """A node in the code tree: a level plus its code index within that codebook."""

    level: str
    code: int

    def __post_init__(self):
        if self.level not in LEVELS:
            raise ValueError(f"unknown level {self.level!r}")
        if self.code < 0:
            raise ValueError("code index must be non-negative")


@dataclass(frozen=True)
class CodeTree:
    """A neural code tree: one solid code, and per-profile (profile-code, loop-codes).

    ``profiles[i]`` is a ``(profile_code, (loop_code, ...))`` pair. This mirrors the
    S-P-L hierarchy where each profile groups a set of loops.
    """

    solid: int
    profiles: tuple[tuple[int, tuple[int, ...]], ...]


# --- serialization ---------------------------------------------------------
def serialize(tree: CodeTree) -> tuple[object, ...]:
    """Depth-first flatten with a ``<SEP>`` before each profile grouping (Sec. 5).

    Returns a tuple of :class:`CodeNode` and :data:`SEP` markers in the paper's order
    ``[S, <SEP>, P, L, ..., <SEP>, P, L, ...]``.
    """
    out: list[object] = [CodeNode(SOLID, tree.solid)]
    for prof_code, loop_codes in tree.profiles:
        out.append(SEP)
        out.append(CodeNode(PROFILE, prof_code))
        for lc in loop_codes:
            out.append(CodeNode(LOOP, lc))
    return tuple(out)


@dataclass(frozen=True)
class FeatureLayout:
    """One-hot feature layout: code index -> global slot across the 3 codebooks + SEP.

    Global order is [loop codes | profile codes | solid codes | separator]. The
    feature size is ``loop + profile + solid + 1`` (paper Sec. 5).
    """

    loop_size: int
    profile_size: int
    solid_size: int

    @property
    def sep_slot(self) -> int:
        return self.loop_size + self.profile_size + self.solid_size

    @property
    def feature_size(self) -> int:
        return self.sep_slot + 1

    def slot(self, element: object) -> int:
        """Global one-hot slot for a :class:`CodeNode` or the :data:`SEP` marker."""
        if element == SEP:
            return self.sep_slot
        if not isinstance(element, CodeNode):
            raise ValueError(f"cannot place element {element!r}")
        if element.level == LOOP:
            base, size = 0, self.loop_size
        elif element.level == PROFILE:
            base, size = self.loop_size, self.profile_size
        else:  # SOLID
            base, size = self.loop_size + self.profile_size, self.solid_size
        if element.code >= size:
            raise ValueError(f"{element.level} code {element.code} exceeds size {size}")
        return base + element.code

    def onehot(self, element: object) -> tuple[int, ...]:
        """Full one-hot vector for an element."""
        vec = [0] * self.feature_size
        vec[self.slot(element)] = 1
        return tuple(vec)


# --- three-level control masking -------------------------------------------
@dataclass(frozen=True)
class ControlMask:
    """A boolean mask over the serialized code tree.

    ``fixed[i]`` is True when the i-th serialized element is user-specified (frozen);
    ``<SEP>`` markers are always structural (fixed). Elements that are not fixed are
    to be (re)generated by the model.
    """

    elements: tuple[object, ...]
    fixed: tuple[bool, ...]

    def generated_positions(self) -> tuple[int, ...]:
        return tuple(i for i, f in enumerate(self.fixed) if not f)

    def fixed_positions(self) -> tuple[int, ...]:
        return tuple(i for i, f in enumerate(self.fixed) if f)


def level_mask(tree: CodeTree, fixed_levels: set[str]) -> ControlMask:
    """Freeze whole levels; every other code node is left to be generated.

    ``<SEP>`` markers are always fixed (structural). This expresses controls like
    "keep the solid arrangement, regenerate everything below" (fix {solid}).
    """
    bad = fixed_levels - set(LEVELS)
    if bad:
        raise ValueError(f"unknown levels: {sorted(bad)}")
    elements = serialize(tree)
    fixed = []
    for el in elements:
        if el == SEP:
            fixed.append(True)
        else:
            fixed.append(el.level in fixed_levels)
    return ControlMask(elements, tuple(fixed))


def edit_code(tree: CodeTree, level: str, index: int, new_code: int) -> CodeTree:
    """Return a new tree with one node's code changed (Sec. 6.4 code-tree editing).

    ``index`` addresses the solid (ignored for level ``solid``), the profile, or the
    loop. For ``loop`` the index is a flat position over all loops in tree order.
    """
    if new_code < 0:
        raise ValueError("new_code must be non-negative")
    if level == SOLID:
        return replace(tree, solid=new_code)
    if level == PROFILE:
        profs = list(tree.profiles)
        if not 0 <= index < len(profs):
            raise IndexError(f"profile index {index} out of range")
        _, loops = profs[index]
        profs[index] = (new_code, loops)
        return replace(tree, profiles=tuple(profs))
    if level == LOOP:
        profs = [(pc, list(lcs)) for pc, lcs in tree.profiles]
        flat = 0
        for pi, (_pc, lcs) in enumerate(profs):
            for li in range(len(lcs)):
                if flat == index:
                    lcs[li] = new_code
                    return replace(
                        tree,
                        profiles=tuple((pc, tuple(l)) for pc, l in profs),
                    )
                flat += 1
        raise IndexError(f"loop index {index} out of range ({flat} loops)")
    raise ValueError(f"unknown level {level!r}")


def autocomplete_mask(tree: CodeTree, known: set[tuple[str, int]]) -> ControlMask:
    """Mark the ``known`` (level, tree-index) nodes as fixed; the rest are predicted.

    Models the paper's autocomplete-from-partial-input: the user supplies a few nodes
    (e.g. one extruded profile), and the code-tree generator predicts the rest.
    ``known`` uses the same flat indexing as :func:`edit_code` (solid index 0).
    """
    elements = serialize(tree)
    # Precompute the (level, index) address of each serialized code node.
    addresses: list[tuple[str, int] | None] = []
    loop_counter = 0
    profile_counter = -1
    for el in elements:
        if el == SEP:
            addresses.append(None)
        elif el.level == SOLID:
            addresses.append((SOLID, 0))
        elif el.level == PROFILE:
            profile_counter += 1
            addresses.append((PROFILE, profile_counter))
        else:  # LOOP
            addresses.append((LOOP, loop_counter))
            loop_counter += 1
    fixed = []
    for el, addr in zip(elements, addresses):
        if el == SEP:
            fixed.append(True)
        else:
            fixed.append(addr in known)
    return ControlMask(elements, tuple(fixed))
