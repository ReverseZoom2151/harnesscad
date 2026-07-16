# 9. Spiral Staircase With Helical Handrail

![Spiral staircase orbit gif](benchmark_09_spiral_staircase.gif)

## Prompt

Create a single STEP model of a miniature spiral staircase in millimeters.

The staircase is centered on the origin and rises along Z.

Add a central vertical column, diameter 14 mm and height 140 mm, with its bottom at Z = 0.

Add 20 identical wedge-shaped stair treads arranged helically around the column. Each tread is 4 mm thick, has an inner radius of 10 mm, an outer radius of 62 mm, and subtends 24 degrees in plan view. The first tread is at Z = 4 mm, and each subsequent tread rises by 6 mm and rotates by 18 degrees around Z.

Add a helical outer handrail tube of diameter 5 mm following radius 66 mm, starting at Z = 14 mm and ending at Z = 130 mm, making one full revolution around the staircase.

Add 20 vertical balusters, each diameter 3 mm, connecting the outer end of each tread to the handrail.

Add a circular base disk, diameter 90 mm and thickness 5 mm.

Export as a STEP file.

## Test Cases

| Test | Expected result |
| --- | --- |
| STEP import | File imports successfully |
| Main structure | Valid solid bodies or valid fused assembly |
| Central column | Diameter 14 mm, 140 mm tall |
| Column placement | Centered on Z-axis; bottom at Z = 0 |
| Base disk | Diameter 90 mm, 5 mm thick |
| Tread count | 20 |
| Tread thickness | 4 mm |
| Tread inner radius | R10 mm |
| Tread outer radius | R62 mm |
| Tread angular width | 24 degrees |
| Tread rise | 6 mm per step |
| Tread rotation | 18 degrees per step |
| Helical progression | Treads rise and rotate together, not merely radial-patterned flat |
| Handrail | Diameter 5 mm tube at R66 mm |
| Handrail vertical range | Z = 14 mm to Z = 130 mm |
| Handrail revolution | Approximately one full 360-degree turn |
| Baluster count | 20 |
| Baluster diameter | Diameter 3 mm |
| Baluster relation | Each connects tread outer region to handrail |
| Negative check | No missing rail; no flat circular stair pattern with no rise |
