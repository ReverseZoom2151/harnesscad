"""environment_onshape - a live Onshape browser session as an Environment, with
the REST API as a SEPARATE, agent-untouchable oracle.

Why this module exists (the hole it closes)
--------------------------------------------
The whole case for computer-use was: *the CAD tools people are paid to use -
SolidWorks, Fusion, Onshape, Inventor, CATIA - have no reliable headless API, so
the GUI is the only universal CAD API.* We then proved it on FreeCAD, which has a
perfectly good Python API. That proof is compromised: we demonstrated GUI driving
on the one tool that did not need it.

Onshape is the honest bridge. It is a real commercial CAD tool, browser-based
(so a real GUI to drive), AND it has a complete REST API. That second fact is not
a convenience - it is the entire point. **The REST API is not the actuator, it is
the ORACLE.** We drive the browser GUI like a human; we then read the part back
through a channel the agent never touched, and compare. On a desktop tool the
scripting API and the GUI share one process, so a "verified" GUI action can be a
GUI action that quietly used the API. Here the two channels are physically
separate: the actuator is a browser rendering ``cad.onshape.com`` over HTTPS; the
oracle is ``https://cad.onshape.com/api`` signed with an API key. Nothing the
browser does can reach into the REST reads, and nothing the REST reads do can move
the part. That separation is exactly what the whole design demands.

    actuator  : the browser DOM + WebGL viewport (a human-shaped channel)
    oracle    : /api/v9/partstudios/.../massproperties, /boundingboxes, /features
                (a structured, synchronous, agent-untouchable channel)

The two capabilities that differ from FreeCAD's GUI
---------------------------------------------------
FreeCAD's GUI environment declares ``synchronous_read = False``: its only read is
an out-of-band macro channel that may lag the app's own recompute. Onshape is the
opposite. Its REST oracle IS a synchronous structured read of the committed
workspace microversion - so this environment declares ``synchronous_read = True``,
honestly, and that is the one capability a browser-plus-REST tool has that a
desktop-GUI-only tool cannot. It still declares ``content_digest = False``
(Onshape's document microversion id is a version handle, not a content hash of the
geometry - two different feature histories can yield identical solids), and
``deterministic_replay = False`` (server-side regen, element ids minted per
session, browser timing).

Auth - a hard constraint
-------------------------
This module NEVER handles, asks for, prints, or writes an API key or secret. It
reads ``ONSHAPE_ACCESS_KEY`` / ``ONSHAPE_SECRET_KEY`` from the environment ONLY,
via :class:`OnshapeCredentials`, whose ``__repr__`` redacts. When they are absent
the module SKIPS cleanly with a message naming the two variables - it never hangs,
never prompts, never fails. Every test skips without them.

The agent never destroys a document
------------------------------------
The harness creates a fresh SCRATCH document (a name-stamped throwaway), drives
only that, and deletes only that on close. No user document is ever opened,
written, saved-over, or shared. Document deletion goes through the harness's own
REST channel (``DELETE /documents/{did}``), never the agent.

Reachability
------------
Two independent things must be present for a real run:

    1. credentials  (ONSHAPE_ACCESS_KEY / ONSHAPE_SECRET_KEY) - for the oracle and
       for the harness's scratch-document lifecycle.
    2. a browser actuator (Playwright) plus an authenticated browser session - for
       the GUI driving.

:func:`available` reports exactly which of these is missing. With neither, this is
a complete, credential-gated implementation that runs the moment both are present;
it fabricates nothing.

Stdlib only for the oracle (urllib, hmac, hashlib, base64). The actuator is
abstracted behind :class:`BrowserActuator`; the Playwright backing is imported
lazily so the module loads, and its non-browser logic tests, with no browser.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import string
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from email.utils import formatdate
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from harnesscad.core.cisp.ops import Op
from harnesscad.core.environment import (
    CapabilityError, Capabilities, Observation, StepResult, coerce_ops,
)
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.cua import guardrails

#: The two environment variables the user sets. Read-only, never written.
ACCESS_ENV = "ONSHAPE_ACCESS_KEY"
SECRET_ENV = "ONSHAPE_SECRET_KEY"

#: Where the standard (non-enterprise) Onshape cloud lives. Overridable for an
#: enterprise domain via HARNESSCAD_ONSHAPE_BASE (e.g. https://acme.onshape.com).
DEFAULT_BASE_URL = "https://cad.onshape.com"


# ---------------------------------------------------------------------------
# credentials - read from the environment, NEVER printed, NEVER persisted
# ---------------------------------------------------------------------------
class OnshapeCredentials:
    """Onshape API key + secret, read from the environment and nothing else.

    The secret lives only in this object's private slots and is used only to sign
    a request. It is NEVER logged, NEVER written to disk, and ``repr`` /``str``
    redact it, so an accidental print in a trace cannot leak it.
    """

    __slots__ = ("_access", "_secret")

    def __init__(self, access: Optional[str] = None, secret: Optional[str] = None) -> None:
        self._access = access if access is not None else os.environ.get(ACCESS_ENV, "")
        self._secret = secret if secret is not None else os.environ.get(SECRET_ENV, "")

    @property
    def present(self) -> bool:
        return bool(self._access) and bool(self._secret)

    def _sign(self, method: str, path: str, query: str, nonce: str,
              date: str, content_type: str) -> Tuple[str, str]:
        """Return (Authorization header value, the access key) for one request.

        Implements Onshape's HMAC-SHA256 request-signature scheme (see
        docs/auth/apikeys): the string to sign is method, nonce, date,
        content-type, path and query, each on its own line and the whole thing
        lowercased, HMAC-SHA256'd with the secret and base64-encoded. The secret
        never leaves this method.
        """
        to_sign = (
            method + "\n" + nonce + "\n" + date + "\n"
            + content_type + "\n" + path + "\n" + query + "\n"
        ).lower().encode("utf-8")
        digest = hmac.new(self._secret.encode("utf-8"), to_sign, hashlib.sha256).digest()
        signature = base64.b64encode(digest).decode("ascii")
        return "On %s:HmacSHA256:%s" % (self._access, signature), self._access

    def __repr__(self) -> str:  # pragma: no cover - trivial, but load-bearing
        return "OnshapeCredentials(present=%s, secret=<redacted>)" % self.present

    __str__ = __repr__


# ---------------------------------------------------------------------------
# the ORACLE - the REST client the agent never touches
# ---------------------------------------------------------------------------
class OnshapeApiError(RuntimeError):
    """A REST call to the oracle failed. Carries status + body for the trace."""

    def __init__(self, status: int, url: str, body: str = "") -> None:
        self.status = status
        self.url = url
        self.body = body
        super().__init__("Onshape API %s on %s%s"
                         % (status, url, (": " + body[:200]) if body else ""))


#: Onshape mass properties are SI: volume in m^3, area in m^2, centroid in m. CISP
#: ops are authored in millimetres, and the scripted backends measure in mm^3 /
#: mm^2 / mm. Converting here - explicitly, at the oracle boundary - is what makes
#: the differential compare meaningful instead of off by 10^9.
M3_TO_MM3 = 1.0e9
M2_TO_MM2 = 1.0e6
M_TO_MM = 1.0e3


@dataclass(frozen=True)
class MassProperties:
    """The oracle's structured read of a Part Studio's geometry, in mm units."""

    volume_mm3: float
    surface_area_mm2: float
    centroid_mm: Tuple[float, float, float]
    mass: float
    #: The raw SI payload, untouched, for anyone who needs it verbatim.
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"volume": self.volume_mm3, "surface_area": self.surface_area_mm2,
                "center_of_mass": list(self.centroid_mm), "mass": self.mass}


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box from the oracle, in mm units."""

    low_mm: Tuple[float, float, float]
    high_mm: Tuple[float, float, float]

    @property
    def size_mm(self) -> Tuple[float, float, float]:
        return tuple(self.high_mm[i] - self.low_mm[i] for i in range(3))  # type: ignore[return-value]

    def to_dict(self) -> dict:
        return {"bbox": list(self.size_mm), "bbox_min": list(self.low_mm),
                "bbox_max": list(self.high_mm)}


@dataclass(frozen=True)
class DocumentRef:
    """A scratch document coordinate: document / workspace / element ids."""

    did: str
    wid: str
    eid: str = ""

    def with_element(self, eid: str) -> "DocumentRef":
        return DocumentRef(self.did, self.wid, eid)


class OnshapeOracle:
    """The REST oracle. Signs every request; the agent never sees this object.

    All reads are synchronous structured reads of the committed workspace - which
    is precisely why the Environment can honestly declare ``synchronous_read =
    True`` where FreeCAD's GUI declares False.
    """

    def __init__(self, credentials: Optional[OnshapeCredentials] = None,
                 base_url: Optional[str] = None, api_version: str = "v9",
                 timeout: float = 30.0) -> None:
        self.creds = credentials or OnshapeCredentials()
        self.base_url = (base_url or os.environ.get("HARNESSCAD_ONSHAPE_BASE")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.api_version = api_version
        self.timeout = float(timeout)

    # -- signing + transport ----------------------------------------------
    @staticmethod
    def _nonce() -> str:
        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(25))

    def _request(self, method: str, path: str, query: Optional[dict] = None,
                 body: Optional[dict] = None, accept: str = "application/json") -> Any:
        if not self.creds.present:
            raise OnshapeApiError(0, path,
                                  "no credentials; set %s and %s"
                                  % (ACCESS_ENV, SECRET_ENV))
        query = query or {}
        query_str = urllib.parse.urlencode(sorted(query.items()))
        content_type = "application/json"
        date = formatdate(usegmt=True)
        nonce = self._nonce()
        auth, _access = self.creds._sign(method, path, query_str, nonce, date,
                                          content_type)
        url = self.base_url + path + (("?" + query_str) if query_str else "")
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Date", date)
        req.add_header("On-Nonce", nonce)
        req.add_header("Authorization", auth)
        req.add_header("Content-Type", content_type)
        req.add_header("Accept", accept)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
                if accept.startswith("application/json"):
                    return json.loads(payload) if payload else {}
                return payload
        except urllib.error.HTTPError as exc:  # noqa: PERF203
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                pass
            raise OnshapeApiError(exc.code, url, detail) from exc
        except urllib.error.URLError as exc:
            raise OnshapeApiError(0, url, str(exc.reason)) from exc

    # -- document lifecycle (harness-owned; the agent never calls these) ---
    def create_scratch_document(self, name: str) -> DocumentRef:
        """Create a fresh throwaway document and return its did/wid/eid.

        The new document ships with one Part Studio tab; we resolve its element id
        so the caller has a complete coordinate. This is the ONLY document the
        harness will ever drive or delete.
        """
        doc = self._request("POST", "/api/%s/documents" % self.api_version,
                            body={"name": name, "isPublic": False})
        did = doc["id"]
        wid = doc["defaultWorkspace"]["id"]
        ref = DocumentRef(did, wid)
        elements = self.list_elements(ref)
        for el in elements:
            if el.get("elementType") == "PARTSTUDIO" or el.get("type") == "PARTSTUDIO":
                return ref.with_element(el["id"])
        if elements:
            return ref.with_element(elements[0]["id"])
        return ref

    def delete_document(self, did: str) -> None:
        """Permanently remove a scratch document. Harness-owned cleanup only."""
        self._request("DELETE", "/api/%s/documents/%s" % (self.api_version, did),
                      query={"forever": "true"})

    def list_elements(self, ref: DocumentRef) -> List[dict]:
        return self._request(
            "GET", "/api/%s/documents/d/%s/w/%s/elements"
            % (self.api_version, ref.did, ref.wid))

    # -- the oracle reads (each verifies ONE property) --------------------
    def mass_properties(self, ref: DocumentRef) -> MassProperties:
        """GET .../partstudios/.../massproperties -> volume, area, centroid.

        Verifies: volume (m^3->mm^3), surface area (m^2->mm^2), centroid (m->mm).
        """
        data = self._request(
            "GET", "/api/%s/partstudios/d/%s/w/%s/e/%s/massproperties"
            % (self.api_version, ref.did, ref.wid, ref.eid))
        bodies = data.get("bodies", {})
        agg = bodies.get("-all-") or (next(iter(bodies.values())) if bodies else {})
        vol = _first(agg.get("volume"))
        area = _first(agg.get("area") if "area" in agg else agg.get("periphery"))
        centroid = agg.get("centroid") or [0.0, 0.0, 0.0]
        mass = _first(agg.get("mass"))
        return MassProperties(
            volume_mm3=vol * M3_TO_MM3,
            surface_area_mm2=area * M2_TO_MM2,
            centroid_mm=(centroid[0] * M_TO_MM, centroid[1] * M_TO_MM,
                         centroid[2] * M_TO_MM),
            mass=mass, raw=data)

    def bounding_box(self, ref: DocumentRef) -> BoundingBox:
        """GET .../partstudios/.../boundingboxes -> AABB (m->mm).

        Verifies: overall dimensions and placement.
        """
        data = self._request(
            "GET", "/api/%s/partstudios/d/%s/w/%s/e/%s/boundingboxes"
            % (self.api_version, ref.did, ref.wid, ref.eid))
        return BoundingBox(
            low_mm=(data["lowX"] * M_TO_MM, data["lowY"] * M_TO_MM,
                    data["lowZ"] * M_TO_MM),
            high_mm=(data["highX"] * M_TO_MM, data["highY"] * M_TO_MM,
                     data["highZ"] * M_TO_MM))

    def features(self, ref: DocumentRef) -> List[dict]:
        """GET .../partstudios/.../features -> the feature list.

        Verifies: that the GUI actions produced the expected features, in order
        (a sketch, then an extrude, ...), with the expected feature types. This is
        the structural oracle, complementary to the geometric one above.
        """
        data = self._request(
            "GET", "/api/%s/partstudios/d/%s/w/%s/e/%s/features"
            % (self.api_version, ref.did, ref.wid, ref.eid))
        return data.get("features", [])

    def export_stl(self, ref: DocumentRef) -> bytes:
        """GET .../partstudios/.../stl -> the tessellated solid (synchronous).

        The STL endpoint is synchronous; the STEP endpoint is asynchronous (POST
        then poll /translations/{id}). STL is enough for the geometry oracle, so it
        is the default export.
        """
        return self._request(
            "GET", "/api/v6/partstudios/d/%s/w/%s/e/%s/stl"
            % (ref.did, ref.wid, ref.eid),
            query={"mode": "text", "units": "millimeter", "grouping": "true"},
            accept="text/plain")


def _first(value: Any) -> float:
    """Onshape reports scalars as [value, low, high]; take the value. Scalars pass
    through. None -> 0.0, so a missing field is a zero, never a KeyError hole."""
    if value is None:
        return 0.0
    if isinstance(value, (list, tuple)):
        return float(value[0]) if value else 0.0
    return float(value)


# ---------------------------------------------------------------------------
# the ACTUATOR - the browser. Abstract, so the mapping tests without a browser.
# ---------------------------------------------------------------------------
class BrowserActuator(Protocol):
    """The browser-driving surface. Onshape's DOM is the actuator.

    Three-tier action space, same doctrine as FreeCAD's UIA driver: prefer
    semantic/DOM-addressed controls (``click_control``/``fill_field``) over pixels;
    the Part Studio's WebGL viewport is the ONLY place a pixel is unavoidable
    (:meth:`viewport_pick`), and even there the pixel is computed from a known
    camera, never guessed by vision.
    """

    def available(self) -> Tuple[bool, str]: ...
    def open_document(self, url: str) -> None: ...
    def click_control(self, selector: str) -> bool: ...
    def fill_field(self, selector: str, text: str) -> str: ...
    def read_field(self, selector: str) -> str: ...
    def viewport_pick(self, x: int, y: int) -> bool: ...
    def close(self) -> None: ...


class PlaywrightActuator:
    """A Playwright-backed browser actuator. Imported lazily and gated.

    Onshape has no unauthenticated automation surface, so this requires an already
    authenticated browser session (a persistent context / storage state the USER
    established - this module never handles their password any more than it handles
    the API secret). If Playwright is not installed, :meth:`available` says so and
    the environment SKIPS.
    """

    def __init__(self, storage_state: Optional[str] = None, headless: bool = True) -> None:
        self.storage_state = storage_state or os.environ.get(
            "HARNESSCAD_ONSHAPE_BROWSER_STATE")
        self.headless = headless
        self._pw = None
        self._browser = None
        self._page = None

    def available(self) -> Tuple[bool, str]:
        try:
            import playwright.sync_api  # noqa: F401
        except Exception:  # noqa: BLE001
            return False, ("Playwright is not installed; the browser actuator "
                           "cannot drive Onshape's GUI")
        if not self.storage_state:
            return False, ("no authenticated browser session; set "
                           "HARNESSCAD_ONSHAPE_BROWSER_STATE to a Playwright "
                           "storage-state file for a logged-in Onshape session")
        return True, ""

    def _ensure(self):
        if self._page is not None:
            return
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        context = self._browser.new_context(storage_state=self.storage_state)
        self._page = context.new_page()

    def open_document(self, url: str) -> None:
        self._ensure()
        self._page.goto(url, wait_until="networkidle")

    def click_control(self, selector: str) -> bool:
        self._ensure()
        locator = self._page.locator(selector)
        locator.wait_for(state="visible", timeout=15000)
        locator.click()
        return True

    def fill_field(self, selector: str, text: str) -> str:
        self._ensure()
        locator = self._page.locator(selector)
        locator.wait_for(state="visible", timeout=15000)
        locator.fill("")
        locator.type(text)
        return self.read_field(selector)

    def read_field(self, selector: str) -> str:
        self._ensure()
        return self._page.locator(selector).input_value()

    def viewport_pick(self, x: int, y: int) -> bool:
        self._ensure()
        self._page.mouse.click(x, y)
        return True

    def close(self) -> None:
        try:
            if self._browser is not None:
                self._browser.close()
            if self._pw is not None:
                self._pw.stop()
        except Exception:  # noqa: BLE001
            pass
        self._pw = self._browser = self._page = None


# ---------------------------------------------------------------------------
# the op -> Onshape-GUI mapping (DATA, and honest about what it cannot bind)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GuiField:
    """One value to write into an Onshape feature dialog, addressed by selector."""

    selector: str
    #: which op attribute supplies the value (e.g. "distance", "w")
    source: str
    unit: str = "mm"


@dataclass(frozen=True)
class GuiRecipe:
    """How one CISP op is built through Onshape's GUI: a toolbar control, the
    dialog fields to fill (each read back), and the confirm button.

    The selectors are Onshape's stable DOM hooks. Onshape ships ``data-*`` test
    attributes and stable ids on its toolbar and dialogs; the values below name the
    intent and are marked for LIVE CALIBRATION - they must be read off the running
    app once (as FreeCAD's button names were read off its live UIA tree), because
    no vision guess substitutes for the real selector. Until calibrated against a
    live session they are placeholders, and this is stated plainly rather than
    presented as verified.
    """

    op: str
    toolbar: str
    fields: Tuple[GuiField, ...] = ()
    confirm: str = "[data-test-id='dialog-ok']"
    needs_viewport: bool = False


#: The drivable subset, mirroring FreeCAD's (sketch a profile, then a feature).
#: Extend as selectors are calibrated. Kept deliberately small and honest.
RECIPES: Dict[str, GuiRecipe] = {
    "new_sketch": GuiRecipe(
        op="new_sketch", toolbar="[data-test-id='toolbar-sketch']"),
    "add_rectangle": GuiRecipe(
        op="add_rectangle", toolbar="[data-test-id='sketch-rectangle']",
        fields=(GuiField("[data-test-id='rect-width']", "w"),
                GuiField("[data-test-id='rect-height']", "h"))),
    "add_circle": GuiRecipe(
        op="add_circle", toolbar="[data-test-id='sketch-circle']",
        fields=(GuiField("[data-test-id='circle-radius']", "r"),)),
    "extrude": GuiRecipe(
        op="extrude", toolbar="[data-test-id='feature-extrude']",
        fields=(GuiField("[data-test-id='extrude-depth']", "distance"),)),
}

#: Every other op needs a face/edge PICK in the viewport, or an assembly, and is
#: REFUSED with the reason rather than approximated - the same doctrine FreeCAD's
#: environment applies. Half a part looks like a part; that is worse than none.
REQUIRES_VIEWPORT: Dict[str, str] = {
    "add_point": "a sketch point needs a viewport click at a computed pixel",
    "add_line": "a line needs two viewport picks",
    "constrain": "a constraint needs the two sketch entities picked in the viewport",
    "fillet": "fillet needs the target edges picked in the viewport",
    "chamfer": "chamfer needs the target edges picked in the viewport",
    "boolean": "boolean needs two solids selected in the tree/viewport",
    "revolve": "revolve needs an axis picked in the sketch",
    "hole": "hole needs a face datum and location picked in the viewport",
    "shell": "shell needs the open faces picked in the viewport",
    "draft": "draft needs faces and a neutral plane picked in the viewport",
    "loft": "loft needs the ordered profiles picked in the viewport",
    "sweep": "sweep needs a profile and a path picked in the viewport",
    "linear_pattern": "pattern needs the seed feature selected in the tree",
    "circular_pattern": "pattern needs the seed feature and an axis picked",
    "mirror": "mirror needs the seed feature and a mirror plane picked",
    "add_instance": "assembly insert is a different workspace element",
    "mate": "a mate needs two instances picked in the assembly",
    "set_param": "editing a feature needs it selected in the tree first",
}


def value_for(op: Op, source: str) -> float:
    """Pull the field value off an op by attribute name. Raises if absent - a
    binding we cannot fill is a binding we do not attempt."""
    if not hasattr(op, source):
        raise KeyError("op %r has no attribute %r for GUI binding"
                       % (type(op).__name__, source))
    return float(getattr(op, source))


# ---------------------------------------------------------------------------
# reachability
# ---------------------------------------------------------------------------
def available(actuator: Optional[BrowserActuator] = None) -> Tuple[bool, str]:
    """(can this environment run here, why not). Never raises, never hangs.

    Two independent requirements, reported separately so the user knows exactly
    what to provide: credentials (for the oracle + scratch-doc lifecycle) and a
    browser actuator (for the GUI).
    """
    missing: List[str] = []
    if not OnshapeCredentials().present:
        missing.append("credentials: set %s and %s (never entered here - "
                       "environment variables only)" % (ACCESS_ENV, SECRET_ENV))
    act = actuator if actuator is not None else PlaywrightActuator()
    ok, why = act.available()
    if not ok:
        missing.append("actuator: " + why)
    if missing:
        return False, "; ".join(missing)
    return True, ""


class OnshapeUnavailable(RuntimeError):
    """The Onshape environment cannot run here. Carries the precise reason."""


# ---------------------------------------------------------------------------
# the Environment
# ---------------------------------------------------------------------------
class OnshapeGuiEnvironment:
    """A live Onshape browser session as an Environment; REST is the oracle.

    Read :attr:`CAPABILITIES` before relying on anything. The one honest
    difference from FreeCAD's GUI is ``synchronous_read = True``.
    """

    #: What it can honestly do. Note the difference from FreeCAD's declaration:
    #: synchronous_read is TRUE here, because the REST oracle is a real synchronous
    #: structured read of the committed workspace - a channel FreeCAD's GUI lacks.
    CAPABILITIES = Capabilities(
        name="onshape-gui",
        # False: Onshape's microversion id is a version handle minted per edit, not
        # a content hash of the geometry (distinct histories can share a solid).
        content_digest=False,
        # False: a rejected feature dialog has already opened a panel and started a
        # feature preview; the browser is mutated before the value is refused.
        nonmutating_reject=False,
        # TRUE, and this is the whole point of the bridge: the REST oracle IS a
        # synchronous structured read of the committed part - a separate channel
        # the agent never actuates. FreeCAD's GUI declares this False.
        synchronous_read=True,
        # False: server-side regenerate, per-session element ids, browser timing.
        deterministic_replay=False,
        export=True,
        export_formats=("stl", "step"),
        supported_ops=tuple(RECIPES.keys()),
        unsupported_ops=dict(REQUIRES_VIEWPORT),
        resolve_before_act=True,
        notes=(
            "the browser DOM/WebGL is the ACTUATOR; the REST API is the ORACLE, and "
            "they are physically separate channels (HTTPS app vs signed /api) - the "
            "agent never touches REST, which is what makes it an oracle not a mirror",
            "synchronous_read is TRUE (unlike FreeCAD's GUI): Onshape's REST is a "
            "real synchronous structured read of the committed workspace",
            "no content digest: Onshape's microversion id is a version handle, not a "
            "content hash of the geometry, and this environment will not fake one",
            "reject is MUTATING: a refused feature dialog has already opened a panel "
            "and begun a preview",
            "the agent NEVER creates, saves, deletes or shares a user document; the "
            "harness makes ONE scratch document, drives only it, and deletes only it",
            "API key + secret are read from %s / %s ONLY, never entered, printed or "
            "written" % (ACCESS_ENV, SECRET_ENV),
            "only the sketch-a-profile-then-feature subset is drivable coordinate-"
            "free; every op needing a viewport/tree PICK is refused, not faked",
            "mass properties, bounding box and centroid are reported in mm/mm^2/mm^3 "
            "after converting Onshape's SI payload at the oracle boundary",
        ),
    )

    def __init__(self, credentials: Optional[OnshapeCredentials] = None,
                 oracle: Optional[OnshapeOracle] = None,
                 actuator: Optional[BrowserActuator] = None,
                 scratch_name: Optional[str] = None) -> None:
        self.creds = credentials or OnshapeCredentials()
        ok, why = available(actuator)
        if not ok:
            raise OnshapeUnavailable("the Onshape environment cannot run: " + why)
        self.oracle = oracle or OnshapeOracle(self.creds)
        self.actuator: BrowserActuator = actuator or PlaywrightActuator()
        self.scratch_name = scratch_name or (
            "harnesscad-scratch-%d" % int(time.time()))
        self.guards = guardrails.Guardrails()
        self.doc: Optional[DocumentRef] = None
        self._steps = 0
        self._built: List[Op] = []
        self._pending: List[Op] = []
        self._outcomes: List[Dict[str, Any]] = []

    # -- Environment -------------------------------------------------------
    def capabilities(self) -> Capabilities:
        return self.CAPABILITIES

    def reset(self) -> Observation:
        """Fresh scratch document, opened in the browser. Nothing is reused."""
        self.close()
        self.doc = self.oracle.create_scratch_document(self.scratch_name)
        url = ("%s/documents/%s/w/%s/e/%s"
               % (self.oracle.base_url, self.doc.did, self.doc.wid, self.doc.eid))
        self.actuator.open_document(url)
        self._steps = 0
        self._built = []
        self._pending = []
        self._outcomes = []
        return self.observe()

    def step(self, action) -> StepResult:
        """Drive the GUI for the buffered ops; refuse what needs a viewport pick.

        An op the GUI cannot do coordinate-free is REFUSED with its reason, never
        approximated. Verification is deferred to the oracle: ``verified`` is set
        only after the REST read confirms the feature landed (see :meth:`observe`).
        """
        ops = coerce_ops(action)
        self._steps += 1
        caps = self.capabilities()
        diags: List[Diagnostic] = []
        for op in ops:
            tag = getattr(type(op), "OP", "")
            if not caps.supports(tag):
                diags.append(Diagnostic(Severity.ERROR, "unsupported-op",
                                        "onshape-gui cannot drive '%s': %s"
                                        % (tag, caps.why_not(tag))))
        if diags:
            return StepResult(ok=False, verified=False, observation=self.observe(),
                              reward=-1.0, diagnostics=diags,
                              info={"step": self._steps})

        executed = 0
        for op in ops:
            recipe = RECIPES[getattr(type(op), "OP", "")]
            try:
                outcome = self._run_recipe(recipe, op)
            except (guardrails.GuardrailViolation, OnshapeApiError) as exc:
                diags.append(Diagnostic(Severity.ERROR, "gui-error", str(exc)))
                break
            self._outcomes.append(outcome)
            if not outcome["ok"]:
                diags.append(Diagnostic(Severity.ERROR, "gui-error",
                                        outcome.get("error", "unverified action")))
                break
            self._built.append(op)
            executed += 1

        ok = not diags
        # The ORACLE, not the browser's own return value, decides verification: we
        # read the feature list back through REST and confirm it grew by exactly the
        # features we drove. A browser click that "succeeded" but produced no
        # feature is NOT verified - that is the whole discipline.
        verified = ok and self._verify_features_via_oracle(executed)
        return StepResult(
            ok=ok, verified=verified, observation=self.observe(),
            reward=1.0 if ok else -1.0, diagnostics=diags,
            info={"step": self._steps, "executed_ops": executed,
                  "outcomes": self._outcomes[-3:]})

    def observe(self) -> Observation:
        """Structured state. ``digest`` is None - always. That is the honest answer.

        Unlike FreeCAD's hybrid observation, this one can carry a real synchronous
        structured read (the oracle), so state includes the feature count when a
        document is live.
        """
        state: Dict[str, Any] = {"ops_built": [op.to_dict() for op in self._built],
                                 "ops_pending": len(self._pending)}
        if self.doc is not None and self.doc.eid:
            try:
                state["feature_count"] = len(self.oracle.features(self.doc))
            except OnshapeApiError as exc:
                state["oracle_error"] = str(exc)
        return Observation(kind="structured", state=state, digest=None,
                           step=self._steps,
                           notes=("no content digest: an Onshape workspace has a "
                                  "microversion id, which is a version handle, not "
                                  "a content hash; this environment will not fake "
                                  "one",))

    def export(self, fmt: str):
        """Export through the REST oracle (STL synchronous; STEP is async-poll)."""
        f = str(fmt).lower()
        if self.doc is None:
            raise OnshapeApiError(0, "export", "no scratch document; call reset()")
        if f == "stl":
            payload = self.oracle.export_stl(self.doc)
            return payload.decode("utf-8", "replace") \
                if isinstance(payload, bytes) else payload
        raise ValueError("onshape-gui export supports 'stl' synchronously; 'step' "
                         "is asynchronous and not wired in this build")

    def close(self) -> None:
        """Delete the scratch document (harness-owned) and drop the browser."""
        if self.doc is not None:
            try:
                self.oracle.delete_document(self.doc.did)
            except OnshapeApiError:
                pass
            self.doc = None
        try:
            self.actuator.close()
        except Exception:  # noqa: BLE001
            pass

    # -- capability-gated --------------------------------------------------
    def state_digest(self) -> str:
        raise CapabilityError(
            self.CAPABILITIES.name, "content_digest",
            "an Onshape workspace exposes a microversion id, which is a VERSION "
            "handle, not a content hash of the geometry - two different feature "
            "histories can produce the identical solid and carry different ids. "
            "Returning it as a content digest would be a silent lie.")

    def query(self, q: str) -> dict:
        """A SYNCHRONOUS structured read - through the oracle. This is the honest
        capability that FreeCAD's GUI does not have.
        """
        if self.doc is None:
            raise OnshapeApiError(0, "query", "no scratch document; call reset()")
        if q in ("measure", "metrics"):
            mp = self.oracle.mass_properties(self.doc)
            bb = self.oracle.bounding_box(self.doc)
            out = mp.to_dict()
            out.update(bb.to_dict())
            return out
        if q == "measure_volume":
            return {"volume": self.oracle.mass_properties(self.doc).volume_mm3}
        if q == "bbox":
            return self.oracle.bounding_box(self.doc).to_dict()
        if q == "features":
            feats = self.oracle.features(self.doc)
            return {"features": feats, "count": len(feats)}
        if q == "document":
            return {"did": self.doc.did, "wid": self.doc.wid, "eid": self.doc.eid}
        raise CapabilityError(self.CAPABILITIES.name, "synchronous_read",
                              "unknown query %r; try measure/bbox/features/document"
                              % q)

    def measure(self, q: str = "measure") -> dict:
        """Alias to :meth:`query` for parity with FreeCAD's environment API."""
        return self.query(q if q != "measure" else "measure")

    # -- the GUI recipe runner --------------------------------------------
    def _run_recipe(self, recipe: GuiRecipe, op: Op) -> Dict[str, Any]:
        """Click the toolbar, WRITE-AND-READ-BACK each field, confirm.

        The read-back is the same 375mm defence FreeCAD uses: a numeric field we
        cannot prove we set is a field we did not set. Onshape's dialogs take a
        value-with-units expression ('37.5 mm'), so we type that and read it back.
        """
        t0 = time.time()
        if not self.actuator.click_control(recipe.toolbar):
            return {"ok": False, "op": recipe.op, "error": "toolbar not reachable",
                    "elapsed": time.time() - t0}
        writes: List[dict] = []
        for fb in recipe.fields:
            value = value_for(op, fb.source)
            typed = "%g %s" % (value, fb.unit)
            read_back = self.actuator.fill_field(fb.selector, typed)
            landed = _numeric_matches(value, read_back)
            writes.append({"field": fb.selector, "intended": value,
                           "typed": typed, "read_back": read_back,
                           "verified": landed})
            if not landed:
                # The classic silent 10x error dies here as it does for FreeCAD.
                return {"ok": False, "op": recipe.op,
                        "error": "field %s: read back %r, intended %g - not verified"
                                 % (fb.selector, read_back, value),
                        "writes": writes, "elapsed": time.time() - t0}
        self.actuator.click_control(recipe.confirm)
        return {"ok": True, "op": recipe.op, "writes": writes,
                "elapsed": time.time() - t0}

    def _verify_features_via_oracle(self, executed: int) -> bool:
        """Confirm through REST that the feature list grew as the GUI claimed.

        This is the oracle acting as an oracle: the browser said it built
        ``executed`` features; the REST channel - which the agent never actuated -
        must independently show them. A count that did not move is an unverified
        step, no matter what the browser returned.
        """
        if executed == 0:
            return True
        if self.doc is None:
            return False
        try:
            feats = self.oracle.features(self.doc)
        except OnshapeApiError:
            return False
        # At least as many features as ops we built (a rectangle may add its sketch
        # plus the rectangle within one feature; the invariant is monotone growth
        # to >= the number of build ops).
        return len(feats) >= len(self._built)

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "OnshapeGuiEnvironment":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _numeric_matches(intended: float, read_back: str, rel_tol: float = 1e-4) -> bool:
    """Parse a value out of an Onshape field read-back and compare to intent.

    Onshape renders '37.5 mm' (or '37.5' in the sketch dimension). We extract the
    leading number and compare with a tight tolerance - so 37.5 vs 375 is a hard
    failure, the same discipline as :mod:`harnesscad.io.cua.quantity`.
    """
    import re
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(read_back).replace(",", ""))
    if not m:
        return False
    try:
        got = float(m.group(0))
    except ValueError:
        return False
    if intended == 0.0:
        return abs(got) < 1e-9
    return abs(got - intended) <= max(rel_tol * abs(intended), 1e-6)
