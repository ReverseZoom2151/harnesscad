"""Deterministic catalogue of the KCL stdlib, engine ops and file formats.

Static reference data for the KCL toolchain:

*   :data:`KCL_STD_FUNCTIONS` -- the KCL standard-library function set, grouped by
    module (the ``std_fn`` dispatch table). These are the callable modeling
    primitives a KCL program uses.
*   :data:`ENGINE_OPS` -- the engine modeling-command set (the ``ModelingCmd``
    discriminated union). These are the low-level ops the KCL executor sends to
    the engine.
*   :data:`IMPORT_FORMATS` / :data:`EXPORT_FORMATS_3D` / :data:`EXPORT_FORMATS_2D`
    and :data:`CONVERSION_MATRIX` -- the file-conversion format matrix. This
    tells a codec author which formats matter.
*   :data:`EXTENSION_TO_IMPORT_FORMAT` -- filename-extension resolution.

Everything here is inert data + pure query helpers; nothing is imported eagerly,
no API is called. It exists so a backend / codec author has one checked place
to read what the engine actually supports without re-deriving it.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

__all__ = [
    "KCL_STD_FUNCTIONS",
    "ENGINE_OPS",
    "IMPORT_FORMATS",
    "EXPORT_FORMATS_3D",
    "EXPORT_FORMATS_2D",
    "CONVERSION_MATRIX",
    "EXTENSION_TO_IMPORT_FORMAT",
    "std_function_names",
    "std_modules",
    "is_engine_op",
    "can_convert",
    "import_format_for_extension",
    "conversions_from",
    "conversions_to",
]

# ---------------------------------------------------------------------------
# KCL standard library (from kcl-lib/src/std/mod.rs std_fn table).
# Keyed by KCL module; values are the snake_case function identifiers.
# ---------------------------------------------------------------------------

KCL_STD_FUNCTIONS: Dict[str, Tuple[str, ...]] = {
    "sketch": (
        "start_sketch_on", "start_profile", "profile_start", "profile_start_x",
        "profile_start_y", "line", "x_line", "y_line", "angled_line",
        "angled_line_that_intersects", "arc", "tangential_arc", "bezier_curve",
        "circle", "circle_three_point", "ellipse", "elliptic", "elliptic_point",
        "conic", "hyperbolic", "hyperbolic_point", "parabolic", "parabolic_point",
        "involute_circular", "polygon", "rectangle", "region", "close",
        "subtract_2d", "extrude", "revolve", "loft", "sweep",
    ),
    "solid": (
        "extrude", "revolve", "loft", "sweep", "shell", "hollow", "fillet",
        "chamfer", "blend", "appearance", "subtract", "union", "intersect",
        "split", "delete_face", "flip_surface", "join_surfaces", "is_solid",
        "is_surface", "pattern_linear_3d", "pattern_circular_3d", "pattern_transform",
    ),
    "csg": ("union", "subtract", "intersect", "split"),
    "extrude": ("extrude",),
    "revolve": ("revolve",),
    "loft": ("loft",),
    "sweep": ("sweep",),
    "shell": ("shell", "hollow"),
    "fillet": ("fillet",),
    "chamfer": ("chamfer",),
    "helix": ("helix",),
    "shapes": ("circle", "circle_three_point", "ellipse", "polygon", "rectangle"),
    "patterns": (
        "pattern_linear_2d", "pattern_linear_3d", "pattern_circular_2d",
        "pattern_circular_3d", "pattern_transform", "pattern_transform_2d",
    ),
    "transform": (
        "translate", "rotate", "scale", "mirror_2d", "mirror_3d", "hide", "delete",
    ),
    "mirror": ("mirror_2d", "mirror_3d"),
    "planes": ("offset_plane", "plane_of"),
    "segment": (
        "segment_start", "segment_start_x", "segment_start_y", "segment_end",
        "segment_end_x", "segment_end_y", "segment_length", "segment_angle",
        "last_segment_x", "last_segment_y", "tangent_to_end",
    ),
    "edge": (
        "get_next_adjacent_edge", "get_previous_adjacent_edge", "get_opposite_edge",
        "get_common_edge", "get_bounded_edge",
    ),
    "faces": ("face_of",),
    "constraints": (
        "coincident", "distance", "horizontal_distance", "vertical_distance",
        "horizontal", "vertical", "parallel", "perpendicular", "tangent", "angle",
        "equal_length", "equal_radius", "radius", "diameter", "midpoint", "symmetric",
        "point", "line", "arc", "circle", "control_point_spline",
    ),
    "solver": (
        "coincident", "distance", "horizontal_distance", "vertical_distance",
        "horizontal", "vertical", "parallel", "perpendicular", "tangent", "angle",
        "equal_length", "equal_radius", "radius", "diameter", "midpoint", "symmetric",
        "point", "line", "arc", "circle", "control_point_spline",
    ),
    "gdt": (
        "datum", "flatness", "straightness", "circularity", "cylindricity",
        "concentricity", "symmetry", "runout", "angularity", "perpendicularity",
        "parallelism", "annotation", "note", "distance", "profile", "profile_line",
        "profile_surface", "position",
    ),
    "array": ("concat", "flatten", "map", "pop", "push", "reduce", "slice"),
    "math": (
        "cos", "sin", "tan", "acos", "asin", "atan", "atan2", "sqrt", "abs", "rem",
        "round", "floor", "ceil", "min", "max", "pow", "log", "log2", "log10", "ln",
        "leg_len", "leg_angle_x", "leg_angle_y",
    ),
    "appearance": ("appearance", "hex_string"),
    "assert": ("assert", "assert_is"),
    "clone": ("clone",),
    "runtime": ("exit",),
}

# ---------------------------------------------------------------------------
# Engine op set (the ModelingCmd discriminated union).
# The low-level commands the KCL executor issues to the modeling engine.
# ---------------------------------------------------------------------------

ENGINE_OPS: Tuple[str, ...] = (
    "add_hole_from_offset", "boolean_imprint", "boolean_intersection",
    "boolean_subtract", "boolean_union", "bounding_box", "camera_drag_end",
    "camera_drag_move", "camera_drag_start", "center_of_mass", "close_path",
    "closest_edge", "create_region", "create_region_from_query_point",
    "curve_get_control_points", "curve_get_end_points", "curve_get_type",
    "curve_set_constraint", "default_camera_center_to_scene",
    "default_camera_center_to_selection", "default_camera_focus_on",
    "default_camera_get_settings", "default_camera_get_view",
    "default_camera_look_at", "default_camera_perspective_settings",
    "default_camera_set_orthographic", "default_camera_set_perspective",
    "default_camera_set_view", "default_camera_zoom", "density", "disable_dry_run",
    "edge_get_length", "edge_lines_visible", "enable_dry_run", "enable_sketch_mode",
    "engine_util_evaluate_path", "entity_circular_pattern", "entity_clone",
    "entity_delete_children", "entity_fade", "entity_get_all_child_uuids",
    "entity_get_child_uuid", "entity_get_distance", "entity_get_index",
    "entity_get_num_children", "entity_get_parent_id", "entity_get_primitive_index",
    "entity_get_sketch_paths", "entity_linear_pattern",
    "entity_linear_pattern_transform", "entity_make_helix",
    "entity_make_helix_from_edge", "entity_make_helix_from_params", "entity_mirror",
    "entity_mirror_across", "entity_mirror_across_edge", "entity_set_opacity",
    "export", "extend_path", "extrude", "extrude_to_reference", "face_get_center",
    "face_get_gradient", "face_get_position", "face_is_planar", "get_entity_type",
    "get_num_objects", "get_sketch_mode_plane", "handle_mouse_drag_end",
    "handle_mouse_drag_move", "handle_mouse_drag_start", "highlight_set_entities",
    "highlight_set_entity", "import_files", "loft", "make_axes_gizmo",
    "make_offset_path", "make_plane", "mass", "mouse_click", "mouse_move",
    "move_path_pen", "new_annotation", "object_bring_to_front",
    "object_set_material_params_pbr", "object_set_name", "object_visible",
    "offset_surface", "orient_to_face", "path_get_curve_uuid",
    "path_get_curve_uuids_for_vertices", "path_get_info",
    "path_get_sketch_target_uuid", "path_get_vertex_uuids",
    "plane_intersect_and_project", "plane_set_color", "project_entity_to_plane",
    "project_points_to_plane", "query_entity_type", "query_entity_type_with_point",
    "reconfigure_stream", "region_get_query_point",
    "region_get_resolvable_intersection_info", "remove_scene_objects", "revolve",
    "revolve_about_edge", "scene_clear_all", "scene_get_entity_ids", "select_add",
    "select_clear", "select_entity", "select_get", "select_region_from_point",
    "select_remove", "select_replace", "select_with_point", "send_object",
    "set_background_color", "set_current_tool_properties",
    "set_default_system_properties", "set_grid_auto_scale",
    "set_grid_reference_plane", "set_grid_scale", "set_object_transform",
    "set_order_independent_transparency", "set_scene_units", "set_selection_filter",
    "set_selection_type", "set_tool", "sketch_mode_disable", "start_path",
    "surface_area", "surface_blend", "sweep", "take_snapshot", "twist_extrude",
    "update_annotation", "view_isometric", "volume", "zoom_to_fit",
)

# ---------------------------------------------------------------------------
# File-conversion format matrix.
# ---------------------------------------------------------------------------

#: Formats Zoo can read (FileImportFormat).
IMPORT_FORMATS: Tuple[str, ...] = (
    "acis", "catia", "creo", "fbx", "gltf", "inventor", "nx", "obj", "parasolid",
    "ply", "sldprt", "step", "stl",
)

#: 3D formats Zoo can write (FileExportFormat).
EXPORT_FORMATS_3D: Tuple[str, ...] = ("fbx", "glb", "gltf", "obj", "ply", "step", "stl")

#: 2D formats Zoo can write (OutputFormat2d) -- currently DXF only.
EXPORT_FORMATS_2D: Tuple[str, ...] = ("dxf",)

#: All (src, dst) pairs Zoo's file-conversion endpoint accepts (src != dst).
CONVERSION_MATRIX: Tuple[Tuple[str, str], ...] = tuple(
    (src, dst)
    for src in IMPORT_FORMATS
    for dst in EXPORT_FORMATS_3D
    if src != dst
)

#: Filename extension -> import format (from diff-viewer ``extensionToSrcFormat``).
#: ``dae`` is intentionally absent (disabled in the new format API).
EXTENSION_TO_IMPORT_FORMAT: Dict[str, str] = {
    "fbx": "fbx", "gltf": "gltf", "obj": "obj", "ply": "ply", "sldprt": "sldprt",
    "stp": "step", "step": "step", "stl": "stl",
}


# ---------------------------------------------------------------------------
# Query helpers.
# ---------------------------------------------------------------------------

def std_modules() -> List[str]:
    """Sorted KCL stdlib module names."""
    return sorted(KCL_STD_FUNCTIONS)


def std_function_names(module: str = None) -> List[str]:
    """Sorted ``module::function`` names, optionally restricted to one module."""
    out: List[str] = []
    for mod, fns in KCL_STD_FUNCTIONS.items():
        if module is not None and mod != module:
            continue
        out.extend(f"{mod}::{fn}" for fn in fns)
    return sorted(set(out))


def is_engine_op(name: str) -> bool:
    """True if ``name`` is a known engine modeling command."""
    return name in ENGINE_OPS


def import_format_for_extension(filename_or_ext: str) -> str:
    """Resolve a filename or bare extension to an import format, or '' if unsupported."""
    ext = filename_or_ext.rsplit(".", 1)[-1].lower()
    return EXTENSION_TO_IMPORT_FORMAT.get(ext, "")


def can_convert(src: str, dst: str) -> bool:
    """True if Zoo can convert from import format ``src`` to export format ``dst``."""
    return src in IMPORT_FORMATS and dst in EXPORT_FORMATS_3D and src != dst


def conversions_from(src: str) -> List[str]:
    """Export formats reachable from import format ``src``."""
    return sorted(dst for dst in EXPORT_FORMATS_3D if can_convert(src, dst))


def conversions_to(dst: str) -> List[str]:
    """Import formats that can produce export format ``dst``."""
    return sorted(src for src in IMPORT_FORMATS if can_convert(src, dst))
