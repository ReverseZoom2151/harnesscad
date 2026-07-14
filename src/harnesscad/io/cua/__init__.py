"""CUA — computer-use control of a real CAD GUI, grounded in the a11y tree.

Layout (import order matters — see :mod:`frames`):

* :mod:`frames`   — DPI awareness + the ONE immutable coordinate frame.
* :mod:`quantity` — locale-correct numeric entry with a MANDATORY read-back.
* :mod:`uia`      — the Windows UIAutomation driver; every action returns a
                    VERIFIED outcome or an error.
* :mod:`guardrails` — resolve-before-click, so we can REFUSE.
* :mod:`bindings_freecad` — the CISP-op -> UIA-control table, as DATA.
* :mod:`environment_freecad` — a FreeCAD GUI :class:`~harnesscad.core.environment.Environment`.

Every module here is import-safe without its optional dependencies: they degrade
to ``available() is False`` and the tests SKIP. Nothing hangs, nothing raises at
import time.
"""
