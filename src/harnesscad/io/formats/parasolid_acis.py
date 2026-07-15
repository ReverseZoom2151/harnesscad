"""Parasolid (X_T / X_B) and ACIS (SAT / SAB) -- DECLINED, licensed kernels only.

This module deliberately implements NOTHING. It exists to state, in code, why two
otherwise-obvious B-rep interchange formats are absent from the codec registry.

Parasolid (Siemens, ``.x_t`` text / ``.x_b`` binary) and ACIS (Spatial/Dassault,
``.sat`` text / ``.sab`` binary) are the native transmit formats of two proprietary
geometry kernels. Unlike STEP -- an open ISO 10303 exchange grammar anyone may
parse and emit -- an X_T or SAT file is a serialisation of a *specific kernel's*
internal B-rep, and reading or writing it faithfully requires that kernel:

* **Parasolid** interchange is defined and validated only by the licensed Parasolid
  kernel (or Siemens' PK toolkit). There is no open, complete X_T grammar; a
  stdlib-only "reader" would be a guess, and a "writer" would emit geometry no
  Parasolid consumer would trust.
* **ACIS** SAT/SAB is likewise the ACIS kernel's own dump. The header is legible,
  but the entity records encode kernel-version-specific topology that only the
  ACIS kernel (or a licensed 3D InterOp bridge) round-trips correctly.

Producing either from the harness would mean shipping a licensed kernel -- which we
do not, and this project's stdlib-first, deterministic constraint forbids. So these
formats are OUT OF SCOPE. The supported B-rep hand-off is **STEP** (AP203/214 via
:mod:`harnesscad.io.formats.step`, AP242 via
:mod:`harnesscad.io.formats.step_ap242`), the open ISO exchange schema; convert
through STEP when a Parasolid- or ACIS-based system is the destination.

No functions here read or write geometry, and no adapter is registered for them, so
the registry never offers a Parasolid/ACIS write path it cannot honour.
"""

from __future__ import annotations

from typing import Tuple

__all__ = [
    "LicensedKernelRequired",
    "DECLINED",
    "requires_licensed_kernel",
]


class LicensedKernelRequired(NotImplementedError):
    """Raised if anything tries to treat Parasolid/ACIS as an implemented codec."""


#: The declined formats and the kernel each one requires.
DECLINED: Tuple[Tuple[str, str, str], ...] = (
    ("Parasolid", ".x_t/.x_b", "Siemens Parasolid kernel (PK)"),
    ("ACIS", ".sat/.sab", "Spatial/Dassault ACIS kernel (3D ACIS Modeler)"),
)


def requires_licensed_kernel(fmt: str = "") -> "LicensedKernelRequired":
    """Return the exception explaining why ``fmt`` is not implemented here.

    Callers that reach for a Parasolid/ACIS codec should ``raise`` this; it names
    the licensed kernel the format needs and points at STEP as the open route.
    """
    detail = ", ".join(f"{name} ({exts}) needs {kernel}"
                       for name, exts, kernel in DECLINED)
    which = f" ({fmt})" if fmt else ""
    return LicensedKernelRequired(
        f"Parasolid/ACIS interchange{which} requires a licensed geometry kernel "
        f"and is out of scope for this stdlib-only codec set: {detail}. "
        f"Use STEP (AP203/214 or AP242) as the open B-rep hand-off instead.")
