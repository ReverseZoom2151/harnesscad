# 7. Radial-Engine-Style Cylinder With Cooling Fins

![Radial-engine-style cylinder orbit gif](benchmark_07_radial_engine_cylinder.gif)

## Prompt

Create a single solid radial-engine-style cylinder as a STEP model in millimeters.

The main cylinder axis is vertical along Z and centered at the origin.

Create a central barrel with diameter 36 mm and height 70 mm, bottom at Z = 0.

Around the barrel, add 12 horizontal circular cooling fins. Each fin is 2 mm thick in Z, has outside diameter 62 mm, and is spaced every 5 mm from Z = 10 mm to Z = 65 mm.

Add a thicker base flange at the bottom, outside diameter 70 mm and thickness 8 mm, with six vertical mounting holes of diameter 5 mm on a 56 mm bolt circle.

Add a top cap cylinder, diameter 44 mm and height 8 mm, from Z = 70 mm to Z = 78 mm.

Add an angled spark-plug boss protruding from the side of the top cap. The boss is a cylinder of diameter 12 mm and length 24 mm, angled upward at 35 degrees from horizontal, with its axis pointing outward in the positive X direction.

Add a 5 mm diameter hole through the boss along its own axis.

Add small 1 mm fillets to the outer fin edges and base flange edges.

Export as a STEP file.

## Test Cases

| Test | Expected result |
| --- | --- |
| STEP import | File imports successfully |
| Solid count | One fused watertight solid |
| Main axis | Vertical along Z |
| Barrel | Diameter 36 mm, 70 mm tall |
| Barrel placement | Centered at origin; bottom at Z = 0 |
| Cooling fin count | 12 |
| Fin diameter | Diameter 62 mm |
| Fin thickness | 2 mm each |
| Fin spacing | Nominally 5 mm from Z = 10 to Z = 65 |
| Base flange | Diameter 70 mm, 8 mm thick |
| Mounting hole count | Six |
| Mounting hole diameter | Diameter 5 mm |
| Mounting bolt circle | Diameter 56 mm |
| Mounting hole spacing | 60 degrees between holes |
| Top cap | Diameter 44 mm, 8 mm tall, Z = 70 to 78 mm |
| Spark-plug boss | Diameter 12 mm cylinder, 24 mm long |
| Boss orientation | Points outward in +X direction |
| Boss angle | 35 degrees upward from horizontal |
| Boss bore | Diameter 5 mm through-hole along boss axis |
| Fillets | 1 mm outer fin-edge and flange-edge fillets |
| Negative check | Boss bore must be coaxial with angled boss; fins must not be missing or merged into one thick cylinder |
