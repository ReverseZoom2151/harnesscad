"""Rotational driver coupling for gear trains (CodeToCAD gear constraint).

CodeToCAD's ``apply_gear_constraint(obj, gear_obj, ratio)`` installs a *driver*
on the driven object's rotation:

    driven_rotation = -ratio * driver_rotation

i.e. the driven body's angle is a signed multiple of the driver's, evaluated
every time the driver moves.  The minus sign is the meshing reversal: external
gears turn opposite ways.

The harness's :mod:`geometry.cadgpt_gear_train` already computes gear ratios and
the spatial *placement* of meshing gears (centre distance, phase offset, helix
twist) -- but nothing there *propagates rotation*: a driver angle in never comes
out the far end of the train.  This module supplies that missing kinematic
layer:

* :class:`GearCoupling`  -- one driver->driven edge: ratio, mesh kind, and the
  rotation it induces (``drive``).
* :class:`GearTrain`     -- a set of couplings forming a directed graph; angles
  are propagated from a driven root outward (:meth:`GearTrain.propagate`), with
  cumulative ratios (:meth:`GearTrain.effective_ratio`), speeds
  (:meth:`GearTrain.speeds`), torques (:meth:`GearTrain.torques`), and cycle /
  conflict detection.

Mesh kinds
----------
``external``  meshing spur/helical gears -- reverse (sign ``-1``): the upstream
              constraint's default.
``internal``  ring gear / planet inside it -- same sense (sign ``+1``).
``shaft``     two gears keyed to the same shaft (a compound stage) -- ratio 1,
              same sense (sign ``+1``).

An idler (a gear whose only job is to bridge two others) therefore flips sign
but cancels out of the effective ratio, exactly as in a real train -- this falls
out of the products rather than being special-cased.

Angles are radians, speeds are radians per unit time.  Stdlib only,
deterministic (all traversal is over sorted names).
"""

from __future__ import annotations

__all__ = [
    "GearError",
    "EXTERNAL",
    "INTERNAL",
    "SHAFT",
    "MESH_SIGN",
    "ratio_from_teeth",
    "GearCoupling",
    "GearTrain",
]

EXTERNAL = "external"
INTERNAL = "internal"
SHAFT = "shaft"

MESH_SIGN = {EXTERNAL: -1.0, INTERNAL: 1.0, SHAFT: 1.0}

_TOL = 1e-9


class GearError(ValueError):
    """Raised for an invalid gear coupling or train."""


def ratio_from_teeth(driver_teeth: int, driven_teeth: int) -> float:
    """Speed ratio of a driver/driven pair: ``driver_teeth / driven_teeth``.

    A small driver turning a big driven wheel gives a ratio below 1 -- the
    driven wheel turns slower, which is a reduction.
    """
    driver_teeth = int(driver_teeth)
    driven_teeth = int(driven_teeth)
    if driver_teeth <= 0 or driven_teeth <= 0:
        raise GearError("tooth counts must be positive")
    return driver_teeth / driven_teeth


class GearCoupling(object):
    """One driver -> driven rotational coupling.

    ``drive(angle)`` returns ``sign * ratio * angle`` where ``sign`` comes from
    the mesh kind (``-1`` external, ``+1`` internal/shaft), reproducing the
    upstream scripted driver ``-ratio * gearRotation`` for the external case.
    """

    __slots__ = ("driver", "driven", "ratio", "mesh", "axis")

    def __init__(
        self,
        driver: str,
        driven: str,
        ratio: float = 1.0,
        mesh: str = EXTERNAL,
        axis: str = "z",
    ):
        if driver == driven:
            raise GearError("a gear cannot drive itself")
        if mesh not in MESH_SIGN:
            raise GearError(f"unknown mesh kind: {mesh!r}")
        ratio = float(ratio)
        if ratio <= 0.0:
            raise GearError("ratio must be positive; direction comes from mesh")
        if mesh == SHAFT and abs(ratio - 1.0) > _TOL:
            raise GearError("a shaft coupling has ratio 1")
        self.driver = str(driver)
        self.driven = str(driven)
        self.ratio = ratio
        self.mesh = mesh
        self.axis = str(axis)

    @classmethod
    def from_teeth(
        cls,
        driver: str,
        driven: str,
        driver_teeth: int,
        driven_teeth: int,
        mesh: str = EXTERNAL,
        axis: str = "z",
    ) -> "GearCoupling":
        return cls(
            driver,
            driven,
            ratio_from_teeth(driver_teeth, driven_teeth),
            mesh,
            axis,
        )

    @classmethod
    def on_shaft(cls, driver: str, driven: str, axis: str = "z") -> "GearCoupling":
        """Two gears keyed to one shaft: same angle, same sense."""
        return cls(driver, driven, 1.0, SHAFT, axis)

    @property
    def sign(self) -> float:
        return MESH_SIGN[self.mesh]

    @property
    def signed_ratio(self) -> float:
        """``sign * ratio`` -- the coefficient of the upstream scripted driver."""
        return self.sign * self.ratio

    def drive(self, driver_angle: float) -> float:
        """The driven angle induced by ``driver_angle`` (radians)."""
        return self.signed_ratio * float(driver_angle)

    def back_drive(self, driven_angle: float) -> float:
        """The driver angle that would induce ``driven_angle``."""
        return float(driven_angle) / self.signed_ratio

    def expression(self) -> str:
        """The scripted-driver expression, as the upstream adapter writes it."""
        return f"{self.signed_ratio} * {self.driver}Rotation"

    def inverse(self) -> "GearCoupling":
        """The coupling with driver and driven swapped."""
        return GearCoupling(
            self.driven, self.driver, 1.0 / self.ratio, self.mesh, self.axis
        )

    def as_dict(self):
        return {
            "driver": self.driver,
            "driven": self.driven,
            "ratio": self.ratio,
            "mesh": self.mesh,
            "axis": self.axis,
        }

    def __eq__(self, other):
        if not isinstance(other, GearCoupling):
            return NotImplemented
        return self.as_dict() == other.as_dict()

    def __hash__(self):
        return hash(tuple(sorted(self.as_dict().items())))

    def __repr__(self):
        return (
            f"GearCoupling(driver={self.driver!r}, driven={self.driven!r},"
            f" ratio={self.ratio!r}, mesh={self.mesh!r}, axis={self.axis!r})"
        )


class GearTrain(object):
    """A directed graph of :class:`GearCoupling` edges, driven from a root."""

    def __init__(self):
        self._couplings = []

    # -- building --------------------------------------------------------
    def add(self, coupling: GearCoupling) -> "GearTrain":
        for existing in self._couplings:
            if existing.driven == coupling.driven:
                raise GearError(
                    f"{coupling.driven!r} already has a driver"
                    f" ({existing.driver!r}); a gear takes one driver"
                )
        self._couplings.append(coupling)
        if self._has_cycle():
            self._couplings.pop()
            raise GearError("coupling would create a cycle in the gear train")
        return self

    def mesh(
        self,
        driver: str,
        driven: str,
        driver_teeth: int,
        driven_teeth: int,
        mesh: str = EXTERNAL,
        axis: str = "z",
    ) -> "GearTrain":
        return self.add(
            GearCoupling.from_teeth(
                driver, driven, driver_teeth, driven_teeth, mesh, axis
            )
        )

    def couple_shaft(self, driver: str, driven: str, axis: str = "z") -> "GearTrain":
        return self.add(GearCoupling.on_shaft(driver, driven, axis))

    # -- structure -------------------------------------------------------
    @property
    def couplings(self):
        return list(self._couplings)

    def gears(self):
        """All gear names, sorted."""
        names = set()
        for c in self._couplings:
            names.add(c.driver)
            names.add(c.driven)
        return sorted(names)

    def roots(self):
        """Gears with no driver -- the train's inputs, sorted."""
        driven = {c.driven for c in self._couplings}
        return [name for name in self.gears() if name not in driven]

    def leaves(self):
        """Gears that drive nothing -- the train's outputs, sorted."""
        drivers = {c.driver for c in self._couplings}
        return [name for name in self.gears() if name not in drivers]

    def _children(self, name):
        return sorted(
            (c for c in self._couplings if c.driver == name),
            key=lambda c: c.driven,
        )

    def _driver_of(self, name):
        for c in self._couplings:
            if c.driven == name:
                return c
        return None

    def _has_cycle(self) -> bool:
        colour = {}

        def visit(name):
            colour[name] = 1
            for c in self._children(name):
                state = colour.get(c.driven, 0)
                if state == 1:
                    return True
                if state == 0 and visit(c.driven):
                    return True
            colour[name] = 2
            return False

        for name in self.gears():
            if colour.get(name, 0) == 0 and visit(name):
                return True
        return False

    def path_to(self, name: str):
        """The couplings from the root down to ``name``, in drive order."""
        if name not in self.gears():
            raise GearError(f"unknown gear: {name!r}")
        chain = []
        cursor = name
        while True:
            coupling = self._driver_of(cursor)
            if coupling is None:
                break
            chain.append(coupling)
            cursor = coupling.driver
        chain.reverse()
        return chain

    # -- kinematics ------------------------------------------------------
    def effective_ratio(self, name: str) -> float:
        """Signed ratio from the train's root to ``name`` (1.0 for a root).

        The product of the signed ratios along the drive path: idlers flip the
        sign twice and cancel out of the magnitude.
        """
        total = 1.0
        for coupling in self.path_to(name):
            total *= coupling.signed_ratio
        return total

    def propagate(self, driver: str, angle: float):
        """Angles of every gear reachable from ``driver`` when it turns ``angle``.

        Returns a dict including ``driver`` itself.  Rotation propagates
        downstream only: gears the driver does not reach are absent.
        """
        if driver not in self.gears():
            raise GearError(f"unknown gear: {driver!r}")
        angles = {driver: float(angle)}
        stack = [driver]
        while stack:
            current = stack.pop()
            for coupling in self._children(current):
                angles[coupling.driven] = coupling.drive(angles[current])
                stack.append(coupling.driven)
        return angles

    def speeds(self, driver: str, speed: float):
        """Angular speeds downstream of ``driver`` (same law as angles)."""
        return self.propagate(driver, speed)

    def torques(self, driver: str, torque: float):
        """Torques downstream of ``driver``, assuming an ideal (lossless) train.

        Power is conserved, so torque scales inversely with speed:
        ``T_driven = T_driver / signed_ratio``.
        """
        if driver not in self.gears():
            raise GearError(f"unknown gear: {driver!r}")
        out = {driver: float(torque)}
        stack = [driver]
        while stack:
            current = stack.pop()
            for coupling in self._children(current):
                out[coupling.driven] = out[current] / coupling.signed_ratio
                stack.append(coupling.driven)
        return out

    def drivers_expressions(self):
        """The scripted-driver expression per driven gear, sorted by name."""
        return {
            c.driven: c.expression()
            for c in sorted(self._couplings, key=lambda c: c.driven)
        }

    def reduction(self, source: str, target: str) -> float:
        """Signed ratio from ``source`` to ``target`` (``target`` downstream).

        ``propagate(source, 1.0)[target]``, but computed from the path.
        """
        chain = self.path_to(target)
        names = [c.driver for c in chain] + [target]
        if source not in names:
            raise GearError(f"{target!r} is not downstream of {source!r}")
        start = names.index(source)
        total = 1.0
        for coupling in chain[start:]:
            total *= coupling.signed_ratio
        return total

    def __len__(self):
        return len(self._couplings)

    def __repr__(self):
        return f"GearTrain({self._couplings!r})"
