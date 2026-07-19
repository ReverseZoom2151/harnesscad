"""Prompt-scope safety gate for text-to-CAD briefs.

The gate covers five blocked categories (weapons, medical/life-support,
automotive control, mains AC, and high-power battery) and returns a typed
``ScopeVerdict`` with its blocked flag, category, matched keyword, and message.

This is a pre-pipeline scope gate on the BRIEF, enforced *before* any agent
runs to keep the system constrained to low-voltage maker electronics. It is distinct from
``harnesscad.domain.programs.validate`` code safety, which audits *generated
code*; this module audits the incoming natural-language prompt instead.

Gap filled: HarnessCAD previously had no electronics/netlist IR at all, and
no scope policy for device-level electrical briefs; this gate supplies the
brief-time taxonomy that keeps the new electronics layer inside safe,
hobbyist, low-voltage territory.

Deterministic: keyword categories are checked in a fixed order and the first
match wins.

Usage::

    from harnesscad.domain.spec.safety_scope import check_safety_scope
    verdict = check_safety_scope("build a wifi plant waterer")
    if verdict.blocked:
        raise ValueError(verdict.message)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ScopeVerdict",
    "check_safety_scope",
    "blocked_categories",
    "main",
]


@dataclass(frozen=True)
class ScopeVerdict:
    """The gate's decision: blocked or not, and why."""

    blocked: bool = False
    category: Optional[str] = None
    matched_keyword: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocked": self.blocked,
            "category": self.category,
            "matched_keyword": self.matched_keyword,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScopeVerdict":
        return cls(
            blocked=bool(data.get("blocked", False)),
            category=data.get("category"),
            matched_keyword=data.get("matched_keyword"),
            message=data.get("message"),
        )


# Categories are checked in this order; the first keyword hit wins.
# (category name, keywords, message template with {word} placeholder)
_BLOCKED_CATEGORIES: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    (
        "weapons",
        (
            "weapon",
            "gun",
            "firearm",
            "missile",
            "explosive",
            "grenade",
            "bomb",
            "defense system",
            "tactical military",
            "ammunition",
            "artillery",
            "pistol",
        ),
        "Safety Block: Weapons-related projects ('{word}') are strictly "
        "blocked. This harness only supports educational, hobbyist, and safe "
        "IoT hardware prototypes.",
    ),
    (
        "medical",
        (
            "medical",
            "pacemaker",
            "ventilator",
            "life support",
            "implant",
            "clinical health",
            "surgical",
            "life-support",
            "dialysis",
            "biomedical",
        ),
        "Safety Block: Critical medical or life-support devices ('{word}') "
        "are strictly blocked. This harness only generates low-voltage "
        "educational prototypes and does not compile medical grade "
        "electronics.",
    ),
    (
        "automotive",
        (
            "automotive",
            "car system",
            "ecu control",
            "engine control",
            "vehicle safety",
            "brake control",
            "can-bus car",
            "throttle control",
            "autopilot car",
        ),
        "Safety Block: High-risk automotive vehicle control systems "
        "('{word}') are blocked to prevent unsafe driving automation "
        "prototypes.",
    ),
    (
        "mains_ac",
        (
            "mains ac",
            "110v",
            "220v",
            "ac mains",
            "outlet power",
            "wall plug ac",
            "high voltage ac",
            "240v",
            "ac outlet",
            "wall socket",
        ),
        "Safety Warning: Projects switching mains AC electricity (110V-240V) "
        "are explicitly blocked. Please modify your prompt to use low-voltage "
        "DC relays (e.g. switching 5V or 12V DC elements) for electrical "
        "safety.",
    ),
    (
        "high_power_battery",
        (
            "high-power battery",
            "high power battery",
            "tesla pack",
            "48v battery",
            "60v battery",
            "high voltage lithium",
            "ev battery",
            "electric vehicle battery",
        ),
        "Safety Warning: High-power battery packs and electric vehicle "
        "charging systems are blocked due to extreme fire and electrical "
        "hazards. Please focus on low-voltage battery setups (such as "
        "standard 3.7V LiPo or AA cells).",
    ),
)


def check_safety_scope(prompt: str) -> ScopeVerdict:
    """Check the brief against the five blocked categories.

    Categories are checked in the source's order (weapons, medical,
    automotive, mains AC, high-power battery) and the first keyword found
    inside the lowercased prompt wins. Returns a non-blocked ScopeVerdict
    when the prompt is in scope.
    """
    prompt_lower = prompt.lower()
    for category, keywords, message_template in _BLOCKED_CATEGORIES:
        for word in keywords:
            if word in prompt_lower:
                return ScopeVerdict(
                    blocked=True,
                    category=category,
                    matched_keyword=word,
                    message=message_template.format(word=word),
                )
    return ScopeVerdict(blocked=False)


def blocked_categories() -> List[Dict[str, Any]]:
    """List the taxonomy: category names and their keyword lists."""
    return [
        {"category": category, "keywords": list(keywords)}
        for category, keywords, _message in _BLOCKED_CATEGORIES
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` exercises one blocked prompt per
    category plus in-scope prompts, asserting categories, matched keywords,
    and precedence (weapons keyword wins over a later-category keyword)."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.domain.spec.safety_scope",
        description="Pre-pipeline prompt-scope safety gate for device briefs "
        ".",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="run one blocked prompt per category plus safe prompts.",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the verdicts as JSON."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="optional single prompt to check instead of the selfcheck suite.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.prompt is not None and not args.selfcheck:
        verdict = check_safety_scope(args.prompt)
        if args.json:
            print(json.dumps(verdict.to_dict(), indent=2, sort_keys=True))
        else:
            print("blocked: %s" % verdict.blocked)
            if verdict.blocked:
                print("category: %s" % verdict.category)
                print("message: %s" % verdict.message)
        return 1 if verdict.blocked else 0

    if not args.selfcheck:
        parser.print_help()
        return 0

    cases = [
        ("a 3d printed pistol frame", "weapons", "pistol"),
        ("a diy pacemaker monitor", "medical", "pacemaker"),
        ("a throttle control module for my car", "automotive", "throttle control"),
        ("a smart switch for 220v wall power", "mains_ac", "220v"),
        ("a 48v battery management charger", "high_power_battery", "48v battery"),
    ]
    safe_prompts = [
        "a wifi plant waterer with a soil sensor",
        "a pocket mp3 player with an oled display",
        "a desk thermostat display using a 3.7v lipo",
    ]

    verdicts = {prompt: check_safety_scope(prompt) for prompt, _c, _k in cases}
    safe_verdicts = {prompt: check_safety_scope(prompt) for prompt in safe_prompts}

    def _case_ok(prompt: str, category: str, keyword: str) -> bool:
        verdict = verdicts[prompt]
        if not (
            verdict.blocked
            and verdict.category == category
            and verdict.matched_keyword == keyword
            and verdict.message
        ):
            return False
        # The first three category messages quote the matched keyword; the
        # mains AC and battery messages are generic advisories (as in the
        # source, which had no '{word}' in those two strings).
        if category in ("weapons", "medical", "automotive"):
            return keyword in verdict.message.lower()
        return True

    blocked_ok = all(_case_ok(*case) for case in cases)
    safe_ok = all(not verdict.blocked for verdict in safe_verdicts.values())

    # Precedence: a prompt hitting both weapons and mains AC blocks as weapons.
    precedence = check_safety_scope("a 220v missile launcher")
    precedence_ok = precedence.blocked and precedence.category == "weapons"

    taxonomy = blocked_categories()
    taxonomy_ok = [entry["category"] for entry in taxonomy] == [
        "weapons",
        "medical",
        "automotive",
        "mains_ac",
        "high_power_battery",
    ] and all(entry["keywords"] for entry in taxonomy)

    if args.json:
        print(
            json.dumps(
                {
                    "blocked": {p: v.to_dict() for p, v in verdicts.items()},
                    "safe": {p: v.to_dict() for p, v in safe_verdicts.items()},
                    "taxonomy": taxonomy,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("safety_scope selfcheck:")
        for prompt, category, keyword in cases:
            verdict = verdicts[prompt]
            print(
                "  BLOCKED [%s via '%s']: %s"
                % (verdict.category, verdict.matched_keyword, prompt)
            )
        for prompt in safe_prompts:
            print("  ALLOWED: %s" % prompt)
        print("  taxonomy categories: %d" % len(taxonomy))

    if not (blocked_ok and safe_ok and precedence_ok and taxonomy_ok):
        print(
            "SELFCHECK FAILED: blocked=%s safe=%s precedence=%s taxonomy=%s"
            % (blocked_ok, safe_ok, precedence_ok, taxonomy_ok),
            file=sys.stderr,
        )
        return 1
    print(
        "safety_scope selfcheck OK: 5 categories block, safe prompts pass, "
        "weapons takes precedence"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
