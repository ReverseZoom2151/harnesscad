# 4. Stepped Shaft With Keyway

![Stepped shaft orbit gif](benchmark_04_stepped_shaft_keyway.gif)

## Prompt

Create a single solid stepped shaft STEP model in millimeters.

The shaft axis runs along X. The total length is 120 mm. The left end center is at X = 0, Y = 0, Z = 0.

From X = 0 to X = 30, the shaft diameter is 20 mm. From X = 30 to X = 90, the shaft diameter is 30 mm. From X = 90 to X = 120, the shaft diameter is 20 mm.

Add a 1 mm chamfer to both end edges.

Add a rectangular keyway slot on the top of the 30 mm diameter middle section. The keyway is 6 mm wide in Y, 3 mm deep in Z, and runs from X = 40 to X = 80.

Export as a STEP file.

## Test Cases

| Test | Expected result |
| --- | --- |
| STEP import | File imports successfully |
| Solid count | Exactly one watertight solid |
| Shaft axis | Along X, not Z |
| Total length | 120 mm |
| Left section | Diameter 20 mm, length 30 mm |
| Middle section | Diameter 30 mm, length 60 mm |
| Right section | Diameter 20 mm, length 30 mm |
| Coaxiality | All three cylindrical sections share same X-axis |
| End chamfers | 1 mm chamfer at both ends |
| Keyway location | On top of middle section |
| Keyway length | From X = 40 to X = 80 |
| Keyway width | 6 mm in Y |
| Keyway depth | 3 mm downward into shaft |
| Negative check | Keyway should not pass through the shaft; shaft should not be vertical |
