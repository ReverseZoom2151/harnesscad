"""ShapeGraMM grammar + Markov rule selection.

From "ShapeGraMM: On the fly procedural generation of massive models for
real-time visualization" (Santos, Brazil, Raposo, 2023). ShapeGraMM extends
the CGA shape grammar. Production rules may share the same predecessor name
and the derivation picks among the alternatives.

This module is DISTINCT from ``procedural.shape_grammar`` (plain weighted
expansion where every alternative for a symbol has a fixed, context-free
weight). Here rule selection follows a *Markov model* on the derivation tree:
which alternative is chosen for a symbol depends on the rule that produced
that symbol (the parent rule id). Transition tables let the same predecessor
expand differently according to how it was reached, which is how ShapeGraMM
captures the non-random repetition patterns of real CAD models.

Deterministic: all randomness flows through ``random.Random(seed)``.
"""

from dataclasses import dataclass, field
import random


@dataclass(frozen=True)
class Rule:
    """A production rule ``source -> targets`` identified by ``rid``.

    ``weight`` is the context-free base weight, used when no Markov
    transition table applies to the parent rule.
    """

    rid: str
    source: str
    targets: tuple
    weight: float = 1.0

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("rule weight must be non-negative")


class MarkovGrammar:
    """Shape grammar whose alternative selection is a Markov chain.

    ``transitions`` maps a parent rule id to a ``{child_rule_id: weight}``
    table. When a symbol is expanded and the parent rule id has a table, only
    the child rules listed there (with weight > 0) are eligible, weighted by
    the table. When the parent rule id has no table (including the axiom,
    whose parent id is ``None``), the context-free ``Rule.weight`` values are
    used, matching a plain grammar as a fallback.
    """

    def __init__(self, rules, terminals, transitions=None):
        self.rules = tuple(rules)
        self.terminals = frozenset(terminals)
        self.transitions = {k: dict(v) for k, v in (transitions or {}).items()}
        self._by_source = {}
        self._by_id = {}
        for rule in self.rules:
            self._by_source.setdefault(rule.source, []).append(rule)
            if rule.rid in self._by_id:
                raise ValueError("duplicate rule id: %s" % rule.rid)
            self._by_id[rule.rid] = rule

    def candidates(self, symbol):
        """Return the eligible-in-principle rules for ``symbol`` (weight>0)."""
        return tuple(r for r in self._by_source.get(symbol, []) if r.weight > 0)

    def _weighted(self, symbol, parent_rid):
        """Return ``(rules, weights)`` for the Markov-conditioned choice."""
        cands = self.candidates(symbol)
        table = self.transitions.get(parent_rid)
        if table is None:
            return cands, [r.weight for r in cands]
        rules = []
        weights = []
        for r in cands:
            w = table.get(r.rid, 0.0)
            if w > 0:
                rules.append(r)
                weights.append(w)
        return tuple(rules), weights

    def select(self, symbol, parent_rid, rng):
        """Pick one rule for ``symbol`` given the parent rule id.

        Returns ``None`` when the symbol has no eligible successor (an
        unproductive/blocked expansion).
        """
        rules, weights = self._weighted(symbol, parent_rid)
        if not rules:
            return None
        if len(rules) == 1:
            return rules[0]
        return rng.choices(rules, weights=weights, k=1)[0]

    def expand(self, axiom, *, seed=0, max_depth=32, max_nodes=100000):
        """Derive the grammar from ``axiom`` until only terminals remain.

        Returns ``(terminals, trace, diagnostics)`` where ``terminals`` is a
        tuple of ``(symbol, depth)`` pairs in generation order, ``trace`` is a
        tuple of ``(symbol, rule_id, parent_rule_id)`` applications, and
        ``diagnostics`` is a sorted tuple of budget/blocking notices.
        """
        rng = random.Random(seed)
        # stack of (symbol, parent_rid, depth); LIFO gives a stable left-to-right
        # depth-first order when children are pushed reversed.
        stack = [(axiom, None, 0)]
        terminals = []
        trace = []
        diagnostics = set()
        nodes = 0
        while stack:
            symbol, parent_rid, depth = stack.pop()
            nodes += 1
            if nodes > max_nodes:
                diagnostics.add("node_budget")
                break
            if symbol in self.terminals:
                terminals.append((symbol, depth))
                continue
            if depth >= max_depth:
                diagnostics.add("depth_budget:%s" % symbol)
                continue
            rule = self.select(symbol, parent_rid, rng)
            if rule is None:
                diagnostics.add("blocked:%s" % symbol)
                continue
            trace.append((symbol, rule.rid, parent_rid))
            for child in reversed(rule.targets):
                stack.append((child, rule.rid, depth + 1))
        return tuple(terminals), tuple(trace), tuple(sorted(diagnostics))

    def transition_matrix(self, parent_rid):
        """Return the normalized ``{child_rule_id: probability}`` for a parent.

        Uses the Markov table if present, else the context-free weights of all
        rules (grouped by source has no meaning here, so this reflects the raw
        table). Returns an empty dict if the parent has no table.
        """
        table = self.transitions.get(parent_rid)
        if not table:
            return {}
        total = sum(w for w in table.values() if w > 0)
        if total <= 0:
            return {}
        return {k: v / total for k, v in table.items() if v > 0}
