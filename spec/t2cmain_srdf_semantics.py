"""SRDF (Semantic Robot Description Format) parser with URDF cross-validation.

Ported from ``packages/cadjs/src/lib/urdf/parseSrdf.js`` of the ``text-to-cad``
(CAD Skills) repository.  An SRDF layers MoveIt planning semantics on top of a
URDF: planning groups, end effectors, named group states, and the disabled
self-collision matrix.  Nothing in the harness modelled any of this.

The transferable content is not the XML reading -- it is the *semantic closure*
and the validation rules, all of which need the linked URDF to evaluate:

* :func:`chain_joint_names` walks the URDF joint tree from a chain's
  ``base_link`` to its ``tip_link`` and returns the ordered joint path;
* :func:`group_joint_names` / :func:`group_link_names` compute a planning
  group's effective joints and links by unioning its explicit joints, its
  chains (expanded via the tree walk), and its subgroups (recursively, with
  cycle protection).  Fixed and mimic joints are excluded from the joint
  closure, because they are not plannable degrees of freedom;
* an end effector's ``parent_link`` must live in its ``parent_group`` and must
  be adjacent (share a joint) to -- but disjoint from -- the links of the
  effector group it hangs off;
* a ``group_state`` may only set plannable joints *of its own group*, and each
  value is checked against the URDF limit, remembering that SRDF values are in
  radians for revolute joints while the parsed URDF stores degrees;
* disabled-collision pairs are deduplicated on the *unordered* link pair, may
  not be self-pairs, must carry a reason, and the free-text reason is classified
  into a stable ``adjacent`` / ``sampled`` / ``setup_assistant`` / ``assumed`` /
  ``manual`` source enum.

Deterministic and stdlib-only.  Document order is preserved throughout.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from spec.t2cmain_urdf_parser import UrdfDocument, _children, _local_name

COLLISION_SOURCES = ("adjacent", "sampled", "setup_assistant", "assumed", "manual")


class SrdfParseError(ValueError):
    """Raised when an SRDF is malformed or inconsistent with its URDF."""


@dataclass(frozen=True)
class Chain:
    base_link: str
    tip_link: str


@dataclass(frozen=True)
class PlanningGroup:
    name: str
    joint_names: Tuple[str, ...] = ()
    link_names: Tuple[str, ...] = ()
    chains: Tuple[Chain, ...] = ()
    subgroups: Tuple[str, ...] = ()


@dataclass(frozen=True)
class EndEffector:
    name: str
    parent_link: str
    group: str
    parent_group: str = ""
    link: str = ""


@dataclass(frozen=True)
class GroupState:
    name: str
    group: str
    joint_values_rad: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DisabledCollisionPair:
    link1: str
    link2: str
    reason: str
    source: str


@dataclass(frozen=True)
class SrdfDocument:
    robot_name: str
    planning_groups: Tuple[PlanningGroup, ...] = ()
    end_effectors: Tuple[EndEffector, ...] = ()
    group_states: Tuple[GroupState, ...] = ()
    disabled_collision_pairs: Tuple[DisabledCollisionPair, ...] = ()

    def group(self, name: str) -> Optional[PlanningGroup]:
        for planning_group in self.planning_groups:
            if planning_group.name == name:
                return planning_group
        return None


def classify_collision_reason(reason: str) -> str:
    """Map a free-text ``disable_collisions`` reason onto a stable source enum."""
    normalized = str(reason or "").strip().lower()
    if "adjacent" in normalized:
        return "adjacent"
    if any(token in normalized for token in ("never", "always", "sample", "default")):
        return "sampled"
    if "setup" in normalized or "assistant" in normalized:
        return "setup_assistant"
    if "assum" in normalized:
        return "assumed"
    return "manual"


def chain_joint_names(urdf: UrdfDocument, base_link: str, tip_link: str) -> Tuple[str, ...]:
    """Ordered joint names on the URDF tree path from ``base_link`` to ``tip_link``."""
    if not base_link or not tip_link or base_link == tip_link:
        return ()
    joints_by_parent: Dict[str, List] = {}
    for joint in urdf.joints:
        joints_by_parent.setdefault(joint.parent_link, []).append(joint)

    stack: List[Tuple[str, Tuple[str, ...]]] = [(base_link, ())]
    visited: Set[str] = set()
    while stack:
        link_name, path = stack.pop()
        if link_name == tip_link:
            return path
        if link_name in visited:
            continue
        visited.add(link_name)
        for joint in reversed(joints_by_parent.get(link_name, [])):
            stack.append((joint.child_link, path + (joint.name,)))
    return ()


def _plannable(urdf: UrdfDocument, joint_name: str) -> bool:
    joint = urdf.model.joints_by_name.get(joint_name)
    return bool(joint and joint.type != "fixed" and joint.mimic is None)


def group_joint_names(
    group: PlanningGroup,
    urdf: UrdfDocument,
    groups_by_name: Dict[str, PlanningGroup],
    _visiting: Optional[Set[str]] = None,
) -> Tuple[str, ...]:
    """The plannable joints of a group: explicit, else chains + subgroups."""
    if group.joint_names:
        return tuple(name for name in group.joint_names if _plannable(urdf, name))

    visiting = set() if _visiting is None else _visiting
    names: List[str] = []
    seen: Set[str] = set()

    def append(candidates) -> None:
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                names.append(candidate)

    for chain in group.chains:
        append(
            name
            for name in chain_joint_names(urdf, chain.base_link, chain.tip_link)
            if _plannable(urdf, name)
        )
    visiting.add(group.name)
    for subgroup_name in group.subgroups:
        if subgroup_name in visiting:
            continue
        subgroup = groups_by_name.get(subgroup_name)
        if subgroup is not None:
            append(group_joint_names(subgroup, urdf, groups_by_name, visiting))
    visiting.discard(group.name)
    return tuple(names)


def group_link_names(
    group: PlanningGroup,
    urdf: UrdfDocument,
    groups_by_name: Dict[str, PlanningGroup],
    _visiting: Optional[Set[str]] = None,
) -> Set[str]:
    """The links owned by a group: explicit links, child links of its joints and
    chains, plus everything its subgroups own."""
    visiting = set() if _visiting is None else _visiting
    links: Set[str] = set(group.link_names)

    def add_child_link(joint_name: str) -> None:
        joint = urdf.model.joints_by_name.get(joint_name)
        if joint is not None and joint.child_link:
            links.add(joint.child_link)

    for joint_name in group.joint_names:
        add_child_link(joint_name)
    for chain in group.chains:
        if chain.tip_link:
            links.add(chain.tip_link)
        for joint_name in chain_joint_names(urdf, chain.base_link, chain.tip_link):
            add_child_link(joint_name)

    visiting.add(group.name)
    for subgroup_name in group.subgroups:
        if subgroup_name in visiting:
            continue
        subgroup = groups_by_name.get(subgroup_name)
        if subgroup is not None:
            links |= group_link_names(subgroup, urdf, groups_by_name, visiting)
    visiting.discard(group.name)
    return links


def links_are_adjacent(urdf: UrdfDocument, link: str, others: Set[str]) -> bool:
    """True when ``link`` shares a joint with any link in ``others``."""
    for joint in urdf.joints:
        if joint.parent_link == link and joint.child_link in others:
            return True
        if joint.child_link == link and joint.parent_link in others:
            return True
    return False


def _unique_names(values, context: str) -> Tuple[str, ...]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        name = str(value or "").strip()
        if not name:
            raise SrdfParseError(f"{context} cannot include empty values")
        if name in seen:
            raise SrdfParseError(f"{context} includes duplicate {name}")
        seen.add(name)
        result.append(name)
    return tuple(result)


def _group_tip_link(group: Optional[PlanningGroup]) -> str:
    if group is None:
        return ""
    if group.link_names:
        return group.link_names[-1]
    if group.chains:
        return group.chains[-1].tip_link
    return ""


def _parse_planning_groups(
    robot: ElementTree.Element, urdf: UrdfDocument
) -> Tuple[PlanningGroup, ...]:
    link_names = set(urdf.model.link_names)
    joint_names = set(urdf.model.joints_by_name)
    groups: List[PlanningGroup] = []
    seen: Set[str] = set()
    for element in _children(robot, "group"):
        name = (element.get("name") or "").strip()
        if not name:
            raise SrdfParseError("SRDF planning group name is required")
        if name in seen:
            raise SrdfParseError(f"Duplicate SRDF planning group: {name}")
        seen.add(name)

        group_joints = _unique_names(
            (joint.get("name") for joint in _children(element, "joint")),
            f"SRDF planning group {name} jointNames",
        )
        group_links = _unique_names(
            (link.get("name") for link in _children(element, "link")),
            f"SRDF planning group {name} linkNames",
        )
        for joint_name in group_joints:
            if joint_name not in joint_names:
                raise SrdfParseError(
                    f"SRDF planning group {name} references missing joint {joint_name}"
                )
        for link_name in group_links:
            if link_name not in link_names:
                raise SrdfParseError(
                    f"SRDF planning group {name} references missing link {link_name}"
                )

        chains: List[Chain] = []
        for chain_element in _children(element, "chain"):
            base_link = (chain_element.get("base_link") or "").strip()
            tip_link = (chain_element.get("tip_link") or "").strip()
            if base_link not in link_names or tip_link not in link_names:
                raise SrdfParseError(
                    f"SRDF planning group {name} chain references missing link"
                )
            chains.append(Chain(base_link=base_link, tip_link=tip_link))

        subgroups = _unique_names(
            (sub.get("name") for sub in _children(element, "group")),
            f"SRDF planning group {name} subgroups",
        )
        groups.append(
            PlanningGroup(
                name=name,
                joint_names=group_joints,
                link_names=group_links,
                chains=tuple(chains),
                subgroups=subgroups,
            )
        )

    if not groups:
        raise SrdfParseError("SRDF must define at least one planning group")
    for group in groups:
        for subgroup in group.subgroups:
            if subgroup not in seen:
                raise SrdfParseError(
                    f"SRDF planning group {group.name} references missing subgroup {subgroup}"
                )
    return tuple(groups)


def _parse_end_effectors(
    robot: ElementTree.Element,
    urdf: UrdfDocument,
    groups: Tuple[PlanningGroup, ...],
) -> Tuple[EndEffector, ...]:
    link_names = set(urdf.model.link_names)
    groups_by_name = {group.name: group for group in groups}
    seen: Set[str] = set()
    end_effectors: List[EndEffector] = []
    for element in _children(robot, "end_effector"):
        name = (element.get("name") or "").strip()
        if not name:
            raise SrdfParseError("SRDF end effector name is required")
        if name in seen:
            raise SrdfParseError(f"Duplicate SRDF end effector: {name}")
        seen.add(name)
        parent_link = (element.get("parent_link") or "").strip()
        group_name = (element.get("group") or "").strip()
        parent_group = (element.get("parent_group") or "").strip()
        if parent_link not in link_names:
            raise SrdfParseError(
                f"SRDF end effector {name} references missing parent_link "
                f"{parent_link or '(missing)'}"
            )
        if group_name not in groups_by_name:
            raise SrdfParseError(
                f"SRDF end effector {name} references missing group "
                f"{group_name or '(missing)'}"
            )
        if parent_group and parent_group not in groups_by_name:
            raise SrdfParseError(
                f"SRDF end effector {name} references missing parent_group {parent_group}"
            )

        link = _group_tip_link(groups_by_name.get(group_name)) or parent_link
        if link not in link_names:
            raise SrdfParseError(
                f"SRDF end effector {name} references missing link {link or '(missing)'}"
            )

        effector_links = group_link_names(
            groups_by_name[group_name], urdf, groups_by_name
        )
        if parent_group:
            parent_links = group_link_names(
                groups_by_name[parent_group], urdf, groups_by_name
            )
            overlap = sorted(effector_links & parent_links)
            if overlap:
                raise SrdfParseError(
                    f"SRDF end effector {name} group shares link(s) with parent_group: "
                    + ", ".join(overlap)
                )
            if parent_link not in parent_links:
                raise SrdfParseError(
                    f"SRDF end effector {name} parent_link is not in parent_group "
                    f"{parent_group}"
                )
        if (
            effector_links
            and parent_link not in effector_links
            and not links_are_adjacent(urdf, parent_link, effector_links)
        ):
            raise SrdfParseError(
                f"SRDF end effector {name} parent_link is not adjacent to its group"
            )
        end_effectors.append(
            EndEffector(
                name=name,
                parent_link=parent_link,
                group=group_name,
                parent_group=parent_group,
                link=link,
            )
        )
    return tuple(end_effectors)


def _parse_group_states(
    robot: ElementTree.Element,
    urdf: UrdfDocument,
    groups: Tuple[PlanningGroup, ...],
) -> Tuple[GroupState, ...]:
    groups_by_name = {group.name: group for group in groups}
    seen_states: Set[str] = set()
    states: List[GroupState] = []
    for element in _children(robot, "group_state"):
        name = (element.get("name") or "").strip()
        group_name = (element.get("group") or "").strip()
        if not name or group_name not in groups_by_name:
            raise SrdfParseError(
                f"SRDF group_state {name or '(missing)'} references missing group "
                f"{group_name or '(missing)'}"
            )
        state_key = f"{group_name}/{name}"
        if state_key in seen_states:
            raise SrdfParseError(f"Duplicate SRDF group_state {state_key}")
        seen_states.add(state_key)

        allowed = set(
            group_joint_names(groups_by_name[group_name], urdf, groups_by_name)
        )
        values: Dict[str, float] = {}
        for joint_element in _children(element, "joint"):
            joint_name = (joint_element.get("name") or "").strip()
            joint = urdf.model.joints_by_name.get(joint_name)
            try:
                value = float(joint_element.get("value"))
            except (TypeError, ValueError):
                value = float("nan")
            if joint is None or math.isnan(value) or math.isinf(value):
                raise SrdfParseError(
                    f"SRDF group_state {name} has invalid joint value "
                    f"{joint_name or '(missing)'}"
                )
            if joint_name in values:
                raise SrdfParseError(
                    f"SRDF group_state {name} includes duplicate joint {joint_name}"
                )
            if joint.type == "fixed" or joint.mimic is not None:
                raise SrdfParseError(
                    f"SRDF group_state {name} cannot set fixed or mimic joint {joint_name}"
                )
            if joint_name not in allowed:
                raise SrdfParseError(
                    f"SRDF group_state {name} joint {joint_name} is not in group {group_name}"
                )
            if joint.type != "continuous":
                # SRDF values are radians for revolute joints; the parsed URDF
                # keeps its limits in degrees, so convert the limits back.
                if joint.type == "revolute":
                    lower = math.radians(joint.min_value_deg)
                    upper = math.radians(joint.max_value_deg)
                else:
                    lower, upper = joint.min_value_deg, joint.max_value_deg
                if value < lower:
                    raise SrdfParseError(
                        f"SRDF group_state {name} joint {joint_name} is below its "
                        "URDF lower limit"
                    )
                if value > upper:
                    raise SrdfParseError(
                        f"SRDF group_state {name} joint {joint_name} is above its "
                        "URDF upper limit"
                    )
            values[joint_name] = value
        states.append(GroupState(name=name, group=group_name, joint_values_rad=values))
    return tuple(states)


def _parse_disabled_collisions(
    robot: ElementTree.Element, urdf: UrdfDocument
) -> Tuple[DisabledCollisionPair, ...]:
    link_names = set(urdf.model.link_names)
    seen: Set[str] = set()
    pairs: List[DisabledCollisionPair] = []
    for element in _children(robot, "disable_collisions"):
        link1 = (element.get("link1") or "").strip()
        link2 = (element.get("link2") or "").strip()
        reason = (element.get("reason") or "").strip()
        if link1 not in link_names or link2 not in link_names:
            raise SrdfParseError("SRDF disabled collision pair references missing link")
        if link1 == link2:
            raise SrdfParseError("SRDF disabled collision pair cannot repeat the same link")
        pair_key = "/".join(sorted((link1, link2)))
        if pair_key in seen:
            raise SrdfParseError(f"Duplicate SRDF disabled collision pair {pair_key}")
        seen.add(pair_key)
        if not reason:
            raise SrdfParseError(
                f"SRDF disabled collision pair {pair_key} requires a reason"
            )
        pairs.append(
            DisabledCollisionPair(
                link1=link1,
                link2=link2,
                reason=reason,
                source=classify_collision_reason(reason),
            )
        )
    return tuple(pairs)


def parse_srdf(text: str, urdf: UrdfDocument) -> SrdfDocument:
    """Parse an SRDF and validate every reference against ``urdf``."""
    try:
        robot = ElementTree.fromstring(text)
    except ElementTree.ParseError as error:
        raise SrdfParseError(f"Failed to parse SRDF XML: {error}") from error
    if _local_name(robot.tag) != "robot":
        raise SrdfParseError("SRDF root element must be <robot>")
    robot_name = (robot.get("name") or "").strip()
    if not robot_name:
        raise SrdfParseError("SRDF robot name is required")
    if urdf.name and urdf.name != robot_name:
        raise SrdfParseError("SRDF robot name must match the linked URDF robot name")

    groups = _parse_planning_groups(robot, urdf)
    return SrdfDocument(
        robot_name=robot_name,
        planning_groups=groups,
        end_effectors=_parse_end_effectors(robot, urdf, groups),
        group_states=_parse_group_states(robot, urdf, groups),
        disabled_collision_pairs=_parse_disabled_collisions(robot, urdf),
    )


def adjacent_collision_pairs(urdf: UrdfDocument) -> Tuple[Tuple[str, str], ...]:
    """Every parent/child link pair of the URDF, sorted -- the pairs a MoveIt
    setup assistant would disable with reason ``Adjacent``."""
    pairs = {
        tuple(sorted((joint.parent_link, joint.child_link)))
        for joint in urdf.joints
        if joint.parent_link and joint.child_link
    }
    return tuple(sorted(pairs))


def missing_adjacent_disables(
    srdf: SrdfDocument, urdf: UrdfDocument
) -> Tuple[Tuple[str, str], ...]:
    """Adjacent link pairs that the SRDF has *not* disabled -- these always
    self-collide in a naive checker and are the classic SRDF omission."""
    declared = {
        tuple(sorted((pair.link1, pair.link2)))
        for pair in srdf.disabled_collision_pairs
    }
    return tuple(
        pair for pair in adjacent_collision_pairs(urdf) if pair not in declared
    )
