# 2. Circular Flange With Bolt-Hole Pattern

![Circular flange orbit gif](benchmark_02_circular_flange.gif)

## Prompt

Create a single solid circular flange as a STEP model in millimeters. The flange is a cylinder with an outside diameter of 80 mm and a thickness of 10 mm. Its axis is vertical along Z, with the bottom face at Z = 0 and the center at X = 0, Y = 0.

Add a central vertical through-bore with diameter 30 mm.

Add six equally spaced vertical through-holes, each 6 mm in diameter, on a 60 mm bolt-circle diameter.

Add a 1.5 mm fillet to the top and bottom outside circular edges.

Export as a STEP file.

## Test Cases

| Test | Expected result |
| --- | --- |
| STEP import | File imports successfully |
| Solid count | Exactly one watertight solid |
| Outer diameter | 80 mm |
| Thickness | 10 mm |
| Central bore | Diameter 30 mm vertical through-hole |
| Coaxiality | Central bore coaxial with flange outer cylinder |
| Bolt-hole count | Six |
| Bolt-hole diameter | 6 mm each |
| Bolt circle | Hole centers lie on 60 mm diameter circle |
| Angular spacing | 60 degrees between adjacent bolt holes |
| Fillets | 1.5 mm fillets on top and bottom outside circular edges |
| Negative check | No missing bolt holes; no misplaced non-circular holes; no extra patterns |
