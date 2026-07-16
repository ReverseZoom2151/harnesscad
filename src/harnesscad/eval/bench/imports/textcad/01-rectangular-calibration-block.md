# 1. Rectangular Calibration Block With Four Holes

![Rectangular calibration block orbit gif](benchmark_01_rectangular_calibration_block.gif)

## Prompt

Create a single solid STEP model in millimeters. The part is a rectangular block, 100 mm long in X, 60 mm wide in Y, and 20 mm tall in Z. Center the block on the XY origin, with the bottom face at Z = 0.

Add four vertical through-holes, each 8 mm in diameter, located at X = +/-35 mm and Y = +/-20 mm.

Add a 2 mm chamfer to the top perimeter edges only. Do not chamfer the holes.

Export as a STEP file.

## Test Cases

| Test | Expected result |
| --- | --- |
| STEP import | File imports successfully |
| Solid count | Exactly one watertight solid |
| Bounding box | 100 x 60 x 20 mm |
| Block placement | Centered on XY origin; bottom at Z = 0 |
| Hole count | Four through-holes |
| Hole diameter | 8 mm each |
| Hole axes | Parallel to Z |
| Hole locations | X = +/-35 mm, Y = +/-20 mm |
| Hole depth | Fully through the 20 mm block |
| Chamfer | 2 mm chamfer on top outer perimeter only |
| Negative check | No chamfers on holes; no extra bosses, slots, text, or decorative geometry |
