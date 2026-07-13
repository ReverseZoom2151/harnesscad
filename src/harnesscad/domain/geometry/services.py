"""Geometry service registry -- the fleet of geometry operations, made callable.

The repo carries ~150 geometry modules and ~35 numeric ones. A handful are wired
into the F-rep backend (SDF primitives, combinators, marching cubes, half-edge,
winding number, quadrature, chord tolerance); the rest were correct, tested and
*unreachable* -- there was no surface through which anything could invoke a gear
generator, a screw-thread sweep, a TPMS infill field, a NURBS knot insertion or a
BVH.

This module is that surface. It mirrors the pattern of
:mod:`harnesscad.eval.verifiers.registry` and :mod:`harnesscad.io.formats.registry`:

*   **Discovery, not assertion.** Every operation names the module it drives, and
    the entry is only published if the static capability registry
    (:mod:`harnesscad.registry`) agrees that the module exists and really exports
    the symbol the operation binds. This surface therefore cannot advertise an
    operation that is not there -- :func:`missing` reports any that fell out.
*   **Adapters, never rewrites.** An :class:`Operation` binds a module function
    directly and forwards its arguments untouched; the geometry modules are not
    modified and their semantics are not reinterpreted here.
*   **Capability dispatch.** Operations carry the same tag vocabulary the
    registry uses (``gears``, ``threads``, ``curves``, ``sdf``, ``meshing``,
    ``acceleration``, ...), so a caller can ask for *what it needs* rather than
    for a module path: ``services.find(tag="threads")``.
*   **Deterministic.** Operations are sorted by name; nothing here reads a clock
    or a random source.

Usage::

    from harnesscad.domain.geometry import services

    geo = services.call("gear.involute.geometry", module=2.0, teeth=20)
    pts = services.call("curve.catmull_rom.points", points=[...], subdivisions=8)
    services.find(tag="sdf")
    services.report()

Everything is stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad import registry as capabilities

# -- drawings (the read/QA side; the export side lives in io.drawing) ---------
from harnesscad.domain.drawings import annotation_mapping
from harnesscad.domain.drawings import canvas_layout
from harnesscad.domain.drawings import dimension_lines
from harnesscad.domain.drawings import gdt
from harnesscad.domain.drawings import iso_ortho_consistency
from harnesscad.domain.drawings import manufacturing_spec
from harnesscad.domain.drawings import svg_view_metrics

# -- geometry: assembly ------------------------------------------------------
from harnesscad.domain.geometry.assembly import box_contact
from harnesscad.domain.geometry.assembly import explode_offsets
from harnesscad.domain.geometry.assembly import exploded_view
from harnesscad.domain.geometry.assembly import instancing
from harnesscad.domain.geometry.assembly import placement
from harnesscad.domain.geometry.assembly import quadtree
from harnesscad.domain.geometry.assembly import scene_validity
from harnesscad.domain.geometry.assembly import split_layout
from harnesscad.domain.geometry.assembly import voxel_parts

# -- geometry: features ------------------------------------------------------
from harnesscad.domain.geometry.features import airfoil
from harnesscad.domain.geometry.features import cap_references
from harnesscad.domain.geometry.features import enclosure
from harnesscad.domain.geometry.features import feature_model
from harnesscad.domain.geometry.features import sketch_extrude
from harnesscad.domain.geometry.features import height_patterns
from harnesscad.domain.geometry.features import holes
from harnesscad.domain.geometry.features import keyframes
from harnesscad.domain.geometry.features import revolve as revolve_feature
from harnesscad.domain.geometry.features import screw_thread
from harnesscad.domain.geometry.features import serpentine
from harnesscad.domain.geometry.features import teardrop
from harnesscad.domain.geometry.features import thread_profile

# -- geometry: kinematics ----------------------------------------------------
from harnesscad.domain.geometry.kinematics import bevel_gear
from harnesscad.domain.geometry.kinematics import gear_coupling
from harnesscad.domain.geometry.kinematics import gear_modules
from harnesscad.domain.geometry.kinematics import gear_train
from harnesscad.domain.geometry.kinematics import involute_gear
from harnesscad.domain.geometry.kinematics import joint_limits
from harnesscad.domain.geometry.kinematics import joint_motion

# -- geometry: mesh ----------------------------------------------------------
from harnesscad.domain.geometry.mesh import bvh
from harnesscad.domain.geometry.mesh import colorize as mesh_colorize
from harnesscad.domain.geometry.mesh import integer_geometry
from harnesscad.domain.geometry.mesh import sampling as mesh_sampling
from harnesscad.domain.geometry.mesh import segmentation as mesh_segmentation
from harnesscad.domain.geometry.mesh import smoothing as mesh_smoothing
from harnesscad.domain.geometry.mesh import template_deform
from harnesscad.domain.geometry.mesh import triangle_intersect

# -- geometry: parametric ----------------------------------------------------
from harnesscad.domain.geometry.parametric import analytic_surfaces
from harnesscad.domain.geometry.parametric import beauty_functionals
from harnesscad.domain.geometry.parametric import catmull_rom
from harnesscad.domain.geometry.parametric import chord_tolerance
from harnesscad.domain.geometry.parametric import facets
from harnesscad.domain.geometry.parametric import hybrid_representation
from harnesscad.domain.geometry.parametric import knot_insertion
from harnesscad.domain.geometry.parametric import path_offset
from harnesscad.domain.geometry.parametric import simplify as polyline_simplify
from harnesscad.domain.geometry.parametric import solid_lines
from harnesscad.domain.geometry.parametric import surface_fit
from harnesscad.domain.geometry.parametric import surface_metrics

# -- geometry: sdf -----------------------------------------------------------
from harnesscad.domain.geometry.sdf import cam_profile
from harnesscad.domain.geometry.sdf import csg_bounds
from harnesscad.domain.geometry.sdf import developability
from harnesscad.domain.geometry.sdf import developable_detect
from harnesscad.domain.geometry.sdf import extra_shapes
from harnesscad.domain.geometry.sdf import spiral
from harnesscad.domain.geometry.sdf import tpms

# -- geometry: sketch --------------------------------------------------------
from harnesscad.domain.geometry.sketch import constraints as sketch_constraints
from harnesscad.domain.geometry.sketch import construction_validity
from harnesscad.domain.geometry.sketch import loop_validity
from harnesscad.domain.geometry.sketch import primitive_fit
from harnesscad.domain.geometry.sketch import symmetry as sketch_symmetry

# -- geometry: topology ------------------------------------------------------
from harnesscad.domain.geometry.topology import edge_convexity
from harnesscad.domain.geometry.topology import entity_selector
from harnesscad.domain.geometry.topology import explorer as topo_explorer
from harnesscad.domain.geometry.topology import face_adjacency
from harnesscad.domain.geometry.topology import region_selectors
from harnesscad.domain.geometry.topology import relative_dimensions
from harnesscad.domain.geometry.topology import selector_dsl
from harnesscad.domain.geometry.topology import selector_grammar
from harnesscad.domain.geometry.topology import synthetic_brep
from harnesscad.domain.geometry.topology import topological_naming

# -- geometry: transforms ----------------------------------------------------
from harnesscad.domain.geometry.transforms import dataset_normalize
from harnesscad.domain.geometry.transforms import grid_normalize
from harnesscad.domain.geometry.transforms import orientation
from harnesscad.domain.geometry.transforms import plane_frame
from harnesscad.domain.geometry.transforms import principal_axes
from harnesscad.domain.geometry.transforms import ransac_pose

# -- geometry: views ---------------------------------------------------------
from harnesscad.domain.geometry.views import camera
from harnesscad.domain.geometry.views import camera_rig
from harnesscad.domain.geometry.views import edge_detection
from harnesscad.domain.geometry.views import plane_detection
from harnesscad.domain.geometry.views import spoke_points
from harnesscad.domain.geometry.views import wireframe_field

# -- geometry: volumes -------------------------------------------------------
from harnesscad.domain.geometry.volumes import dmtet
from harnesscad.domain.geometry.volumes import dual_contouring
from harnesscad.domain.geometry.volumes import edge_sensitivity
from harnesscad.domain.geometry.volumes import occupancy
from harnesscad.domain.geometry.volumes import partition_of_unity
from harnesscad.domain.geometry.volumes import sparse_subdivision
from harnesscad.domain.geometry.volumes import split_signal
from harnesscad.domain.geometry.volumes import triplane_grid
from harnesscad.domain.geometry.volumes import tsdf

# -- numeric -----------------------------------------------------------------
from harnesscad.domain.numeric import assembly_dof
from harnesscad.domain.numeric import compression_metrics
from harnesscad.domain.numeric import constraint_solver
from harnesscad.domain.numeric import parameter_expressions
from harnesscad.domain.numeric import persistent_homology
from harnesscad.domain.numeric import quadrature
from harnesscad.domain.numeric import sequence_complexity
from harnesscad.domain.numeric import sphere_square_map
from harnesscad.domain.numeric import sphere_tracing

__all__ = [
    "Operation",
    "operations",
    "names",
    "find",
    "get",
    "call",
    "modules",
    "missing",
    "capability_matrix",
    "report",
    "UnknownOperationError",
]


class UnknownOperationError(KeyError):
    """No operation is registered under that name."""


@dataclass(frozen=True)
class Operation:
    """One dispatchable geometry operation bound to one module function."""

    name: str
    dotted: str            # the module that actually does the work
    symbol: str            # the public function/class it binds
    tags: Tuple[str, ...]
    summary: str
    fn: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dotted": self.dotted,
            "symbol": self.symbol,
            "tags": list(self.tags),
            "summary": self.summary,
        }


_G = "harnesscad.domain.geometry."
_N = "harnesscad.domain.numeric."
_W = "harnesscad.domain.drawings."

# ---------------------------------------------------------------------------
# The operation table.
#
# (name, dotted module, bound callable, tags, one-line purpose)
#
# Each row is a *real* operation: it takes geometric input and returns geometric
# output. The callable is the module's own public function -- nothing is
# reimplemented, reinterpreted or wrapped here, so an operation cannot drift from
# the module it names.
# ---------------------------------------------------------------------------

_TABLE: Tuple[Tuple[str, str, Callable[..., Any], Tuple[str, ...], str], ...] = (
    # ---- threads / fasteners ----------------------------------------------
    ("thread.profile.iso", _G + "features.thread_profile", thread_profile.iso_thread,
     ("mechanical", "features"), "ISO 60-degree thread tooth cross-section."),
    ("thread.profile.acme", _G + "features.thread_profile", thread_profile.acme_thread,
     ("mechanical", "features"), "ACME 29-degree trapezoidal thread tooth section."),
    ("thread.profile.buttress", _G + "features.thread_profile",
     thread_profile.ansi_buttress_thread, ("mechanical", "features"),
     "ANSI buttress thread tooth section (asymmetric load flank)."),
    ("thread.section.default", _G + "features.screw_thread",
     screw_thread.default_thread_section, ("mechanical", "features"),
     "A default tooth outline for a helical sweep."),
    ("thread.helix", _G + "features.screw_thread", screw_thread.thread,
     ("mechanical", "features", "sweep"),
     "Sweep a tooth outline helically: the actual screw-thread solid."),

    # ---- gears -------------------------------------------------------------
    ("gear.involute.geometry", _G + "kinematics.involute_gear",
     involute_gear.gear_geometry, ("mechanical", "kinematics"),
     "Every radius of an involute spur gear from (module, teeth)."),
    ("gear.involute.point", _G + "kinematics.involute_gear",
     involute_gear.involute_point, ("mechanical", "kinematics", "curves"),
     "A point on the involute of a base circle at a roll angle."),
    ("gear.involute.rack", _G + "kinematics.involute_gear", involute_gear.rack_profile,
     ("mechanical", "kinematics"), "The generating rack profile of a spur gear."),
    ("gear.module.nearest", _G + "kinematics.gear_modules", gear_modules.nearest_module,
     ("mechanical", "kinematics"), "Snap a computed module to the standard series."),
    ("gear.module.is_standard", _G + "kinematics.gear_modules",
     gear_modules.is_standard_module, ("mechanical", "kinematics"),
     "Is this module value a standard one?"),
    ("gear.train.center_distance", _G + "kinematics.gear_train",
     gear_train.center_distance, ("mechanical", "kinematics", "assembly"),
     "Centre distance of a meshing gear pair."),
    ("gear.train.place_driven", _G + "kinematics.gear_train", gear_train.place_driven_gear,
     ("mechanical", "kinematics", "assembly"),
     "Placement (position + phase) of the driven gear of a pair."),
    ("gear.train.ratio", _G + "kinematics.gear_train", gear_train.gear_ratio,
     ("mechanical", "kinematics"), "Gear ratio of a tooth-count pair."),
    ("gear.bevel.pair", _G + "kinematics.bevel_gear", bevel_gear.pitch_cone_angles,
     ("mechanical", "kinematics"), "Pitch cone angles of a bevel-gear pair."),
    ("gear.bevel.spherical_involute", _G + "kinematics.bevel_gear",
     bevel_gear.spherical_involute, ("mechanical", "kinematics", "curves"),
     "A point on the spherical involute of a bevel-gear tooth."),
    ("gear.coupling.ratio", _G + "kinematics.gear_coupling", gear_coupling.ratio_from_teeth,
     ("mechanical", "kinematics"), "Rotational coupling ratio of a gear constraint."),

    # ---- holes / features --------------------------------------------------
    ("hole.simple", _G + "features.holes", holes.simple_hole,
     ("features",), "A plain drilled hole feature (profile + volume)."),
    ("hole.counterbore", _G + "features.holes", holes.counterbore_hole,
     ("features",), "A counterbored hole (stepped cylindrical recess)."),
    ("hole.countersink", _G + "features.holes", holes.countersink_hole,
     ("features",), "A countersunk hole (conical recess)."),
    ("hole.breaks_wall", _G + "features.holes", holes.hole_breaks_wall,
     ("features", "verify"), "Would this hole break through the wall?"),
    ("hole.teardrop", _G + "features.teardrop", teardrop.teardrop_profile,
     ("features", "fabrication"),
     "Self-supporting (teardrop) hole profile for FDM printing."),
    ("hole.teardrop.self_supporting", _G + "features.teardrop",
     teardrop.is_self_supporting, ("features", "fabrication", "verify"),
     "Does a profile stay within the printable overhang limit?"),
    ("feature.airfoil.polygon", _G + "features.airfoil", airfoil.airfoil_polygon,
     ("features", "curves"), "NACA 4-digit airfoil section polygon."),
    ("feature.airfoil.scale", _G + "features.airfoil", airfoil.scale_polygon,
     ("features", "curves"), "Scale a unit-chord airfoil to a real chord."),
    ("feature.serpentine", _G + "features.serpentine", serpentine.serpentine_polyline,
     ("features", "mechanical"), "Meander (serpentine) spring centre-line."),
    ("feature.enclosure.plan", _G + "features.enclosure", enclosure.plan_enclosure,
     ("features",), "Parametric enclosure + lid recipe from a spec."),
    ("feature.revolve.pappus_volume", _G + "features.revolve",
     revolve_feature.pappus_volume, ("features", "revolve"),
     "Volume of a solid of revolution by Pappus's theorem (exact, meshless)."),
    ("feature.revolve.pappus_area", _G + "features.revolve",
     revolve_feature.pappus_surface_area, ("features", "revolve"),
     "Lateral surface area of a solid of revolution by Pappus's theorem."),
    ("feature.revolve.profile_area", _G + "features.revolve",
     revolve_feature.profile_area, ("features", "revolve"),
     "Signed area of a revolve profile (its sweep generator)."),
    ("feature.height_pattern.wave", _G + "features.height_patterns",
     height_patterns.draw_wave, ("features",),
     "A sinusoidal height field on a pin-grid / height-map surface."),
    ("feature.height_pattern.cone", _G + "features.height_patterns",
     height_patterns.draw_cone, ("features",), "A conical height field."),
    ("feature.keyframe.tween", _G + "features.keyframes", keyframes.tween,
     ("features",), "Interpolate two height fields at a keyframe fraction."),

    # ---- curves / parametric ----------------------------------------------
    ("curve.catmull_rom.points", _G + "parametric.catmull_rom",
     catmull_rom.catmull_rom_points, ("curves",),
     "Catmull-Rom spline through control points (interpolating)."),
    ("curve.catmull_rom.prism", _G + "parametric.catmull_rom",
     catmull_rom.catmull_rom_prism, ("curves", "features", "loft"),
     "LOFT: a solid prism lofted through a stack of Catmull-Rom sections."),
    ("curve.catmull_rom.patch", _G + "parametric.catmull_rom",
     catmull_rom.catmull_rom_patch, ("curves", "surface"),
     "A lofted surface patch between two Catmull-Rom curves."),
    ("curve.simplify", _G + "parametric.simplify", polyline_simplify.simplify,
     ("curves",), "Ramer-Douglas-Peucker polyline decimation to a tolerance."),
    ("curve.simplify.deviation", _G + "parametric.simplify",
     polyline_simplify.max_deviation, ("curves", "verify"),
     "Worst deviation a decimation introduced (the honesty check on it)."),
    ("curve.offset", _G + "parametric.path_offset", path_offset.offset_points,
     ("curves", "features"), "Mitred 2D path offset (inward or outward)."),
    ("curve.stroke", _G + "parametric.path_offset", path_offset.path_2d,
     ("curves", "features"), "Stroke a polyline into a closed 2D outline of a width."),
    ("curve.round_polygon", _G + "parametric.path_offset", path_offset.round_polygon,
     ("curves", "features"), "FILLET (2D): round every corner of a polygon."),
    ("curve.fillet_corner", _G + "parametric.path_offset", path_offset.fillet_corner,
     ("curves", "features"), "Fillet arc replacing a single polyline corner."),
    ("curve.arc.approximate", _G + "parametric.chord_tolerance",
     chord_tolerance.approximate_arc, ("curves", "meshing"),
     "Tessellate an arc to a chord (sagitta) tolerance."),
    ("curve.circle.approximate", _G + "parametric.chord_tolerance",
     chord_tolerance.approximate_circle, ("curves", "meshing"),
     "Tessellate a circle to a chord tolerance."),
    ("curve.chord.segments", _G + "parametric.chord_tolerance",
     chord_tolerance.segments_for_tolerance, ("curves", "meshing"),
     "Segments an arc needs to hold a chord tolerance."),
    ("curve.chord.error", _G + "parametric.chord_tolerance", chord_tolerance.chord_error,
     ("curves", "meshing", "verify"), "Chord error of an N-segment arc approximation."),
    ("curve.facets.count", _G + "parametric.facets", facets.get_fragments_from_r,
     ("curves", "meshing"), "OpenSCAD $fn/$fs/$fa facet-count resolution."),
    ("curve.facets.for_error", _G + "parametric.facets", facets.fragments_for_chord_error,
     ("curves", "meshing"), "Facets a circle needs for a chord-error budget."),
    ("curve.nurbs.insert_knot", _G + "parametric.knot_insertion",
     knot_insertion.insert_knot, ("curves",),
     "NURBS knot insertion (shape-preserving refinement)."),
    ("curve.nurbs.refine", _G + "parametric.knot_insertion", knot_insertion.refine_knots,
     ("curves",), "Insert a whole knot vector at once."),
    ("curve.nurbs.to_bezier", _G + "parametric.knot_insertion",
     knot_insertion.decompose_span_to_bezier, ("curves",),
     "Decompose a NURBS span into Bezier form."),
    ("curve.line.solid", _G + "parametric.solid_lines", solid_lines.line,
     ("curves", "features"), "A solid (swept) line segment."),
    ("curve.bending_energy", _G + "parametric.beauty_functionals",
     beauty_functionals.bending_energy, ("curves", "quality"),
     "Discrete bending energy: the fairness of a curve."),
    ("curve.curvature", _G + "parametric.beauty_functionals",
     beauty_functionals.discrete_curvature, ("curves", "quality"),
     "Discrete curvature at each vertex of a polyline."),

    # ---- surfaces ----------------------------------------------------------
    ("surface.sample", _G + "parametric.analytic_surfaces",
     analytic_surfaces.sample_surface, ("surface", "meshing"),
     "Sample an analytic surface (plane/cylinder/cone/sphere/torus) on a UV grid."),
    ("surface.fit.best", _G + "parametric.surface_fit", surface_fit.fit_best,
     ("surface", "reconstruction"),
     "Fit the best analytic surface primitive to a point set."),
    ("surface.fit.plane", _G + "parametric.surface_fit", surface_fit.fit_plane,
     ("surface", "reconstruction"), "Least-squares plane fit."),
    ("surface.fit.cylinder", _G + "parametric.surface_fit", surface_fit.fit_cylinder,
     ("surface", "reconstruction"), "Least-squares cylinder fit."),
    ("surface.fit.sphere", _G + "parametric.surface_fit", surface_fit.fit_sphere,
     ("surface", "reconstruction"), "Least-squares sphere fit."),

    # ---- sdf ---------------------------------------------------------------
    ("sdf.infill.gyroid", _G + "sdf.tpms", tpms.gyroid,
     ("sdf", "fabrication"), "Gyroid TPMS field -- lattice infill for a solid."),
    ("sdf.infill.schwarz_p", _G + "sdf.tpms", tpms.schwarz_p,
     ("sdf", "fabrication"), "Schwarz-P TPMS field."),
    ("sdf.infill.schwarz_d", _G + "sdf.tpms", tpms.schwarz_d,
     ("sdf", "fabrication"), "Schwarz-D TPMS field."),
    ("sdf.infill.neovius", _G + "sdf.tpms", tpms.neovius,
     ("sdf", "fabrication"), "Neovius TPMS field."),
    ("sdf.spiral", _G + "sdf.spiral", spiral.ArcSpiral,
     ("sdf", "curves"), "Exact 2D SDF of an Archimedean spiral."),
    ("sdf.cam.flat_flank", _G + "sdf.cam_profile", cam_profile.make_flat_flank_cam,
     ("sdf", "mechanical"), "Exact 2D SDF of a flat-flank mechanical cam."),
    ("sdf.cam.three_arc", _G + "sdf.cam_profile", cam_profile.make_three_arc_cam,
     ("sdf", "mechanical"), "Exact 2D SDF of a three-arc cam."),
    ("sdf.shape.box_frame", _G + "sdf.extra_shapes", extra_shapes.box_frame,
     ("sdf",), "SDF of a hollow box frame (its 12 edges)."),
    ("sdf.shape.capped_torus", _G + "sdf.extra_shapes", extra_shapes.capped_torus,
     ("sdf",), "SDF of a capped torus (a torus arc)."),
    ("sdf.shape.hex_prism", _G + "sdf.extra_shapes", extra_shapes.hexagonal_prism,
     ("sdf",), "SDF of a hexagonal prism."),
    ("sdf.shape.tri_prism", _G + "sdf.extra_shapes", extra_shapes.triangular_prism,
     ("sdf",), "SDF of a triangular prism."),
    ("sdf.shape.link", _G + "sdf.extra_shapes", extra_shapes.link,
     ("sdf",), "SDF of a chain link."),
    ("sdf.bounds", _G + "sdf.csg_bounds", csg_bounds.bounding_box,
     ("sdf", "csg", "acceleration"),
     "Propagate a bounding box through a typed CSG tree (no kernel)."),
    ("sdf.fits_within", _G + "sdf.csg_bounds", csg_bounds.fits_within,
     ("sdf", "csg", "fabrication"), "Does a CSG tree fit the build volume?"),
    ("sdf.provably_empty", _G + "sdf.csg_bounds", csg_bounds.is_provably_empty,
     ("sdf", "csg", "verify"), "Is this CSG tree provably an empty solid?"),
    ("sdf.developable.classify", _G + "sdf.developable_detect",
     developable_detect.classify_developability, ("sdf", "surface", "fabrication"),
     "Classify a point of an SDF surface as developable / doubly curved."),
    ("sdf.raycast", _N + "sphere_tracing", sphere_tracing.sphere_trace,
     ("sdf", "acceleration"),
     "Sphere-trace (ray-march) an SDF: the pick / raycast query."),
    ("sdf.normal.finite_difference", _N + "sphere_tracing",
     sphere_tracing.estimate_normal, ("sdf",),
     "Finite-difference normal of an arbitrary SDF callable."),

    # ---- meshing / isosurface ---------------------------------------------
    ("mesh.contour_2d", _G + "volumes.dual_contouring", dual_contouring.dual_contour_2d,
     ("meshing", "isosurface", "sdf"),
     "Dual-contour an f-rep graph in 2D with QEF vertex placement. NOTE: this is "
     "a 2D contourer, not a 3D rival of marching cubes -- see report()."),
    ("mesh.smooth.laplacian", _G + "mesh.smoothing", mesh_smoothing.laplacian_smooth,
     ("meshing",), "Uniform Laplacian mesh smoothing."),
    ("mesh.smooth.taubin", _G + "mesh.smoothing", mesh_smoothing.taubin_smooth,
     ("meshing",), "Taubin lambda/mu smoothing (volume-preserving)."),
    ("mesh.sample", _G + "mesh.sampling", mesh_sampling.sample_mesh,
     ("meshing", "pointcloud"), "Area-weighted deterministic surface sampling."),
    ("mesh.components", _G + "mesh.segmentation", mesh_segmentation.connected_components,
     ("meshing", "topology"), "Connected components of a triangle mesh."),
    ("mesh.part_count", _G + "mesh.segmentation", mesh_segmentation.part_count,
     ("meshing", "topology"), "How many disconnected bodies is this mesh?"),
    ("mesh.self_intersects", _G + "mesh.triangle_intersect",
     triangle_intersect.triangles_intersect, ("meshing", "verify"),
     "Do two triangles intersect? (the mesh-boolean / validity substrate)"),
    ("mesh.bvh.build", _G + "mesh.bvh", bvh.BVH, ("acceleration", "meshing"),
     "Build a bounding-volume hierarchy over triangle boxes."),
    ("mesh.bvh.boxes", _G + "mesh.bvh", bvh.boxes_of_triangles,
     ("acceleration", "meshing"), "Per-triangle AABBs, the leaves of a BVH."),
    ("mesh.weld.integer", _G + "mesh.integer_geometry", integer_geometry.VertexRegistry,
     ("meshing",), "Fixed-point vertex welding on a shared integer grid."),

    # ---- volumes -----------------------------------------------------------
    ("volume.occupancy.surface", _G + "volumes.occupancy", occupancy.surface_occupancy,
     ("voxel", "sdf"), "Surface-occupancy shell of a signed-distance grid."),
    ("volume.occupancy.iou", _G + "volumes.occupancy", occupancy.occupancy_iou,
     ("voxel", "benchmark"), "IoU of two occupancy grids."),
    ("volume.tsdf.grid", _G + "volumes.tsdf", tsdf.TSDFGrid,
     ("voxel", "sdf", "csg"), "Voxelised truncated SDF with Boolean algebra."),
    ("volume.tsdf.iou", _G + "volumes.tsdf", tsdf.voxel_iou,
     ("voxel", "benchmark"), "Voxel IoU of two TSDF grids."),

    # ---- sketch ------------------------------------------------------------
    ("sketch.constraint.enforce", _G + "sketch.constraints", sketch_constraints.enforce,
     ("sketch", "constraints"), "Enforce one structural sketch constraint."),
    ("sketch.solve", _N + "constraint_solver", constraint_solver.solve,
     ("sketch", "constraints", "solver"),
     "Gauss-Newton solve of a constrained sketch."),
    ("sketch.diagnose", _N + "constraint_solver", constraint_solver.diagnose,
     ("sketch", "constraints", "solver"),
     "Under/over-constrained diagnosis of a sketch constraint graph."),
    ("sketch.loop.valid", _G + "sketch.loop_validity", loop_validity.check_loop,
     ("sketch", "verify"), "Is a sketch loop closed, simple and non-degenerate?"),
    ("sketch.constructible", _G + "sketch.construction_validity",
     construction_validity.check_sequence, ("sketch", "verify"),
     "Is a construction sequence buildable (no self-intersection, no short edges)?"),
    ("sketch.fit.primitive", _G + "sketch.primitive_fit", primitive_fit.fit_best,
     ("sketch", "reconstruction"), "Fit the best sketch primitive to 2D points."),
    ("sketch.symmetry.axis", _G + "sketch.symmetry", sketch_symmetry.symmetry_axis,
     ("sketch",), "The symmetry axis of a sketch loop."),
    ("sketch.symmetry.reflect", _G + "sketch.symmetry", sketch_symmetry.reflect_loop,
     ("sketch",), "MIRROR (2D): reflect a sketch loop about an axis."),

    # ---- topology ----------------------------------------------------------
    ("topology.select", _G + "topology.selector_dsl", selector_dsl.select,
     ("topology",), "Evaluate a CadQuery string selector against entities."),
    ("topology.selector.parse", _G + "topology.selector_grammar",
     selector_grammar.parse_selector, ("topology", "parsing"),
     "Compile a CadQuery selector string to its object form."),
    ("topology.region.select", _G + "topology.region_selectors", region_selectors.select,
     ("topology",), "Volumetric (region) selection: keep shapes inside a solid region."),
    ("topology.edge_convexity", _G + "topology.edge_convexity",
     edge_convexity.classify_edge_convexity, ("topology", "brep"),
     "Classify a B-rep edge as convex / concave / smooth."),
    ("topology.aag", _G + "topology.edge_convexity", edge_convexity.build_aag,
     ("topology", "brep", "graph"),
     "Attributed adjacency graph of a B-rep (the feature-recognition substrate)."),
    ("topology.synthetic_brep", _G + "topology.synthetic_brep", synthetic_brep.build_topology,
     ("topology", "brep"), "Synthetic B-rep topology (faces/edges) of an analytic primitive."),
    ("topology.explore", _G + "topology.explorer", topo_explorer.topology_summary,
     ("topology", "brep"), "Kernel-free TopoDS-style topology summary."),
    ("topology.naming.fingerprint", _G + "topology.topological_naming",
     topological_naming.fingerprint, ("topology", "parametric"),
     "Stable face fingerprint -- the topological-naming problem."),
    ("topology.naming.match", _G + "topology.topological_naming",
     topological_naming.match_topology, ("topology", "parametric"),
     "Match faces across a parametric rebuild."),
    ("topology.relative_dimension", _G + "topology.relative_dimensions",
     relative_dimensions.resolve_relative_size, ("topology", "parametric"),
     "Resolve a relative dimension ('min + 2mm', '50%') against a bound."),

    # ---- transforms --------------------------------------------------------
    ("transform.orientation", _G + "transforms.orientation", orientation.resolve_orientation,
     ("transform",), "Resolve an orientation directive to a rotation matrix."),
    ("transform.rotation", _G + "transforms.orientation", orientation.rotation_about,
     ("transform",), "Rotation matrix about an axis."),
    ("transform.principal_frame", _G + "transforms.principal_axes",
     principal_axes.principal_frame, ("transform", "pointcloud"),
     "Inertia / principal-axis frame of a point set (canonical pose)."),
    ("transform.inertia_tensor", _G + "transforms.principal_axes",
     principal_axes.inertia_tensor, ("transform", "pointcloud"),
     "Inertia tensor of a point set about its centroid."),
    ("transform.align_clouds", _G + "transforms.principal_axes",
     principal_axes.align_point_clouds, ("transform", "pointcloud"),
     "Correspondence-free alignment of two point clouds."),
    ("transform.ransac_pose", _G + "transforms.ransac_pose", ransac_pose.ransac_rigid_pose,
     ("transform", "pointcloud"), "RANSAC rigid pose from noisy 3D-3D correspondences."),
    ("transform.normalize_grid", _G + "transforms.grid_normalize",
     grid_normalize.center_and_scale_solid, ("transform",),
     "Centre and unit-scale a solid's UV grids."),

    # ---- assembly ----------------------------------------------------------
    ("assembly.contact", _G + "assembly.box_contact", box_contact.classify_boxes,
     ("assembly", "verify"), "Tri-state contact / gap / interference of two boxes."),
    ("assembly.protrusions", _G + "assembly.box_contact", box_contact.scan_protrusions,
     ("assembly", "verify"), "Where does one box protrude out of another?"),
    ("assembly.scene_check", _G + "assembly.scene_validity", scene_validity.check_scene,
     ("assembly", "verify"), "Collision / floating / containment audit of a layout."),
    ("assembly.place", _G + "assembly.placement", placement.resolve_placement,
     ("assembly",), "Resolve an align/offset/polar placement clause."),
    ("assembly.explode.order", _G + "assembly.explode_offsets", explode_offsets.removal_order,
     ("assembly",), "Outside-in disassembly order of an assembly."),
    ("assembly.explode.layout", _G + "assembly.exploded_view", exploded_view.solve_exploded_view,
     ("assembly",), "Exploded-view part positions at a progress fraction."),
    ("assembly.instancing.share", _G + "assembly.instancing", instancing.share_geometries,
     ("assembly",), "Deduplicate redundant geometry into shared instances."),
    ("assembly.quadtree", _G + "assembly.quadtree", quadtree.QuadTreeSpace,
     ("acceleration", "assembly"), "AABB quadtree spatial index over placed parts."),
    ("assembly.split.planar", _G + "assembly.split_layout", split_layout.split_body_planar,
     ("assembly", "fabrication"), "Split a body with a plane (with dowel holes)."),
    ("assembly.split.grid", _G + "assembly.split_layout", split_layout.distribute_in_grid,
     ("assembly", "fabrication"), "Lay parts out on a build plate grid."),
    ("assembly.voxel_parts", _G + "assembly.voxel_parts", voxel_parts.connected_parts,
     ("assembly", "voxel"), "Decompose a voxel solid into connected parts."),
    ("assembly.dof", _N + "assembly_dof", assembly_dof.AssemblyDOF,
     ("assembly", "constraints"), "6-DOF well-posedness of an assembly constraint set."),
    ("joint.limits", _G + "kinematics.joint_limits", joint_limits.revolute,
     ("kinematics", "assembly"), "The 6-DOF limit box of a revolute joint."),
    ("joint.motion", _G + "kinematics.joint_motion", joint_motion.sample_joint_motion,
     ("kinematics", "assembly"), "Sample the motion a joint permits."),
    ("joint.free_dof", _G + "kinematics.joint_motion", joint_motion.joint_free_dof,
     ("kinematics", "assembly"), "Degrees of freedom a joint type leaves free."),

    # ---- views / cameras ---------------------------------------------------
    ("view.camera.extrinsic", _G + "views.camera", camera.extrinsic_matrix,
     ("render",), "Camera extrinsic matrix from a pose."),
    ("view.camera.intrinsic", _G + "views.camera", camera.intrinsic_matrix,
     ("render",), "Pinhole camera intrinsics."),
    ("view.camera.three_view", _G + "views.camera", camera.three_view_extrinsic,
     ("render", "drawings"), "Extrinsics of the three standard orthographic views."),
    ("view.camera.rig", _G + "views.camera_rig", camera_rig.camera_positions,
     ("render",), "Object-framing camera rig around a bounding box."),
    ("view.edges.sobel", _G + "views.edge_detection", edge_detection.sobel_magnitude,
     ("vision",), "Sobel edge magnitude of a 2D view image."),
    ("view.planes.detect", _G + "views.plane_detection", plane_detection.ransac_planes,
     ("reconstruction", "surface"), "RANSAC planar-region detection in a point set."),
    ("view.spokes", _G + "views.spoke_points", spoke_points.process_spoke_points,
     ("curves",), "Group and order spoke edge points for spline generation."),

    # ---- numeric backing ---------------------------------------------------
    ("numeric.quadrature.nodes", _N + "quadrature", quadrature.nodes_and_weights,
     ("numeric",), "Gauss-Legendre nodes and weights (backs exact mass properties)."),
    ("numeric.quadrature.integrate", _N + "quadrature", quadrature.integrate,
     ("numeric",), "Gauss-Legendre integration of a 1D function."),
    ("numeric.parameters.table", _N + "parameter_expressions",
     parameter_expressions.build_table, ("parametric", "spec"),
     "Safe parametric-expression table (a parameter set with dependencies)."),
    ("numeric.parameters.evaluate", _N + "parameter_expressions",
     parameter_expressions.evaluate, ("parametric", "spec"),
     "Evaluate one parametric expression against a table."),
    ("numeric.homology.betti", _N + "persistent_homology", persistent_homology.betti_curve,
     ("topology", "sdf"),
     "Sublevel-set persistent homology of an SDF grid (how many components/voids)."),
    ("numeric.homology.persistence", _N + "persistent_homology",
     persistent_homology.persistence_pairs, ("topology", "sdf"),
     "Persistence pairs of a scalar/SDF grid."),
    # ---- feature model / SSR ----------------------------------------------
    ("feature.sketch_extrude.interpret", _G + "features.sketch_extrude",
     sketch_extrude.Interpreter, ("features", "sketch", "parametric"),
     "Interpret a global-coordinate sketch-and-extrude program into solids."),
    ("feature.ssr.model", _G + "features.feature_model", feature_model.SSRModel,
     ("features", "parametric"),
     "The SSR (Sketch, Sketch-based feature, Refinement) design triple."),
    ("feature.ssr.refinement_targets", _G + "features.cap_references",
     cap_references.build_refinement_entities, ("features", "parametric"),
     "Cap-type reference entities a refinement (fillet/chamfer) can target."),

    # ---- mesh (continued) --------------------------------------------------
    ("mesh.normals", _G + "mesh.template_deform", template_deform.vertex_normals,
     ("meshing",), "Area-weighted vertex normals of a triangle mesh."),
    ("mesh.icosphere", _G + "mesh.template_deform", template_deform.icosphere,
     ("meshing",), "Subdivided icosphere: the canonical template mesh."),
    ("mesh.displace", _G + "mesh.template_deform",
     template_deform.apply_normal_displacement, ("meshing",),
     "Displace a mesh along its vertex normals (template deformation)."),
    ("mesh.color.average", _G + "mesh.colorize", mesh_colorize.mesh_average_color,
     ("meshing", "material"), "Average surface colour of a coloured mesh."),
    ("mesh.color.sample", _G + "mesh.colorize", mesh_colorize.sample_surface_color,
     ("meshing", "material"), "Barycentric colour sample on a mesh face."),

    # ---- surfaces (continued) ----------------------------------------------
    ("surface.representation.choose", _G + "parametric.hybrid_representation",
     hybrid_representation.choose_representation, ("surface", "reconstruction"),
     "Choose analytic vs NURBS representation by Chamfer fidelity."),
    ("surface.metrics.chamfer", _G + "parametric.surface_metrics",
     surface_metrics.chamfer_distance, ("surface", "benchmark"),
     "Chamfer distance between a surface and its point supervision."),
    ("surface.metrics.hausdorff", _G + "parametric.surface_metrics",
     surface_metrics.hausdorff_distance, ("surface", "benchmark"),
     "Hausdorff distance between two point sets."),
    ("sdf.developable.energy", _G + "sdf.developability",
     developability.developability_energy, ("sdf", "surface", "optimization"),
     "Zero-Gaussian-curvature developability energy of an SDF surface."),

    # ---- topology (continued) ----------------------------------------------
    ("topology.entity_select", _G + "topology.entity_selector",
     entity_selector.EntitySelector, ("topology",),
     "Fluent entity selection (nearest / farthest / directional) over a shape."),
    ("topology.face_adjacency", _G + "topology.face_adjacency",
     face_adjacency.FaceAdjacencyGraph, ("topology", "brep", "graph"),
     "Face-adjacency graph and its segmentation of a B-rep."),

    # ---- transforms (continued) --------------------------------------------
    ("transform.plane", _G + "transforms.plane_frame", plane_frame.Plane,
     ("transform",), "CadQuery named-preset plane frame algebra (XY, front, ...)."),
    ("transform.normalize_dataset", _G + "transforms.dataset_normalize",
     dataset_normalize.global_normalize, ("transform", "pointcloud", "dataset"),
     "Dataset-level point-cloud normalisation (one shared centre and scale)."),

    # ---- volumes (continued) -----------------------------------------------
    ("volume.dmtet", _G + "volumes.dmtet", dmtet.DMTet,
     ("voxel", "meshing", "sdf"),
     "Deformable tetrahedral grid encoding of a mesh (marching-tets substrate)."),
    ("volume.dmtet.interpolate", _G + "volumes.dmtet", dmtet.interpolate_sdf_in_tet,
     ("voxel", "sdf"), "Barycentric SDF interpolation inside a tetrahedron."),
    ("volume.edge_sensitivity", _G + "volumes.edge_sensitivity",
     edge_sensitivity.edge_crossing_sensitivity, ("voxel", "meshing", "quality"),
     "How sensitive an iso-surface crossing is to noise on its edge."),
    ("volume.mpu_blend", _G + "volumes.partition_of_unity", partition_of_unity.mpu_blend,
     ("voxel", "sdf"), "Multi-level partition-of-unity blending of local implicits."),
    ("volume.octree.encode", _G + "volumes.split_signal", split_signal.encode_split_signals,
     ("voxel", "acceleration"),
     "Encode an octree's split signal (its subdivision pattern)."),
    ("volume.octree.decode", _G + "volumes.split_signal", split_signal.decode_split_signals,
     ("voxel", "acceleration"), "Rebuild an octree from its split signal."),
    ("volume.subdivide", _G + "volumes.sparse_subdivision", sparse_subdivision.subdivide,
     ("voxel", "sdf"), "Two-stage sparse-voxel subdivision of an SDF shell."),
    ("volume.triplane", _G + "volumes.triplane_grid", triplane_grid.TriplaneGrid,
     ("voxel",), "Triplane (three axis-aligned feature planes) 3D representation."),
    ("view.wireframe_field", _G + "views.wireframe_field", wireframe_field.build_field,
     ("curves", "reconstruction"),
     "Closed-form geometric vector field encoding a wireframe's segments."),

    # ---- numeric (continued) -----------------------------------------------
    ("numeric.sphere_square", _N + "sphere_square_map", sphere_square_map.sphere_to_square,
     ("numeric", "transform"), "Equal-area sphere-to-square parametrisation."),
    ("numeric.grid.iou", _N + "compression_metrics", compression_metrics.occupancy_iou,
     ("numeric", "voxel", "benchmark"), "Occupancy IoU of two SDF grids."),
    ("numeric.grid.rmse", _N + "compression_metrics", compression_metrics.rmse,
     ("numeric", "benchmark"), "RMSE between two scalar fields."),
    ("numeric.sequence.complexity", _N + "sequence_complexity",
     sequence_complexity.sequence_complexity, ("numeric", "parametric"),
     "Complexity of a parametric CAD command sequence."),

    # ---- drawings: the READ / QA side (the export side is io.drawing) -------
    ("drawing.metrics", _W + "svg_view_metrics", svg_view_metrics.analyze_svg_text,
     ("drawings", "benchmark"),
     "Measure a drawing SVG: view labels, path count, components, sheet size."),
    ("drawing.gdt.validate", _W + "gdt", gdt.validate_frames,
     ("drawings", "tolerancing", "verify"),
     "Validate GD&T feature-control frames attached to a drawing."),
    ("drawing.dimensions.detect", _W + "dimension_lines", dimension_lines.detect_dimensions,
     ("drawings",), "Recover dimension / extension lines from raw drawing segments."),
    ("drawing.iso_ortho.consistent", _W + "iso_ortho_consistency",
     iso_ortho_consistency.check_iso_ortho_consistency, ("drawings", "verify"),
     "Do the isometric and the orthographic views agree on the part's extents?"),
    ("drawing.layout", _W + "canvas_layout", canvas_layout.layout_program,
     ("drawings",), "Arrange a program's orthographic views on a fixed canvas."),
    ("drawing.annotation.assign", _W + "annotation_mapping",
     annotation_mapping.assign_features, ("drawings",),
     "Map 2D drawing entities onto the 3D features they annotate."),
    ("drawing.spec.build", _W + "manufacturing_spec", manufacturing_spec.build_spec,
     ("drawings", "spec"),
     "Assemble a unified manufacturing specification from a drawing."),
)


# ---------------------------------------------------------------------------
# Discovery / validation against the static capability index
# ---------------------------------------------------------------------------

_OPS: Optional[Tuple[Operation, ...]] = None
_MISSING: List[dict] = []


def _symbol_name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__name__", type(fn).__name__)


def _build() -> Tuple[Operation, ...]:
    """Publish the table, but only the rows the capability index corroborates."""
    global _MISSING
    out: List[Operation] = []
    missing: List[dict] = []
    for (name, dotted, fn, tags, summary) in _TABLE:
        symbol = _symbol_name(fn)
        try:
            entry = capabilities.get(dotted)
        except KeyError:
            missing.append({"name": name, "dotted": dotted, "symbol": symbol,
                            "reason": "module is not in the capability index"})
            continue
        if symbol not in entry.symbols:
            missing.append({"name": name, "dotted": dotted, "symbol": symbol,
                            "reason": "module does not export that symbol"})
            continue
        # the module's own registry tags travel with the operation
        merged = tuple(sorted(set(tags) | set(entry.tags)))
        out.append(Operation(name=name, dotted=dotted, symbol=symbol, tags=merged,
                             summary=summary, fn=fn))
    out.sort(key=lambda o: o.name)
    _MISSING = missing
    return tuple(out)


def operations(refresh: bool = False) -> Tuple[Operation, ...]:
    """Every published operation, sorted by name (cached)."""
    global _OPS
    if refresh or _OPS is None:
        _OPS = _build()
    return _OPS


def missing() -> List[dict]:
    """Table rows the capability index refused to corroborate (should be empty)."""
    operations()
    return list(_MISSING)


def names() -> List[str]:
    return [o.name for o in operations()]


def get(name: str) -> Operation:
    for o in operations():
        if o.name == name:
            return o
    raise UnknownOperationError(
        "no geometry operation named %r (%d registered)" % (name, len(operations())))


def call(name: str, *args: Any, **kwargs: Any) -> Any:
    """Dispatch by capability name. Arguments go straight to the module function."""
    return get(name)(*args, **kwargs)


def find(tag: Optional[str] = None, package: Optional[str] = None,
         prefix: Optional[str] = None) -> List[Operation]:
    """Operations matching a capability tag, a source package, or a name prefix."""
    out = []
    for o in operations():
        if tag is not None and tag not in o.tags:
            continue
        if package is not None and ("." + package + ".") not in o.dotted:
            continue
        if prefix is not None and not o.name.startswith(prefix):
            continue
        out.append(o)
    return out


def modules() -> List[str]:
    """The distinct modules this surface makes reachable."""
    return sorted({o.dotted for o in operations()})


def tags() -> List[str]:
    seen: Dict[str, int] = {}
    for o in operations():
        for t in o.tags:
            seen[t] = seen.get(t, 0) + 1
    return sorted(seen)


def capability_matrix() -> List[dict]:
    return [o.to_dict() for o in operations()]


def report() -> dict:
    """Machine-readable summary of the geometry fleet behind this surface."""
    ops = operations()
    by_tag: Dict[str, int] = {}
    for o in ops:
        for t in o.tags:
            by_tag[t] = by_tag.get(t, 0) + 1
    return {
        "operations": len(ops),
        "modules": len(modules()),
        "by_tag": dict(sorted(by_tag.items(), key=lambda kv: (-kv[1], kv[0]))),
        "missing": missing(),
        "notes": {
            "mesh.contour_2d":
                "volumes.dual_contouring only implements dual_contour_2d: it is a "
                "2D contourer, NOT a 3D iso-surface extractor. It is therefore not "
                "a rival of marching cubes and is not offered as a 3D mesher. The "
                "3D rivals live in io.backends.frep.MESHERS "
                "(marching_cubes | surface_nets).",
        },
    }
