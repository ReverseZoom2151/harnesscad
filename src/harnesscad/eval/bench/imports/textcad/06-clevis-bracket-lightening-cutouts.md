# 6. Aerospace-Style Clevis Bracket With Lightening Cutouts

![Clevis bracket orbit gif](benchmark_06_clevis_bracket_lightening_cutouts.gif)

## Prompt

Create a single solid aerospace-style clevis bracket as a STEP model in millimeters.

The part is symmetric about the XZ plane.

Start with a base plate 120 mm long in X, 60 mm wide in Y, and 10 mm thick in Z, centered on the XY origin, with bottom face at Z = 0.

Add two vertical clevis lugs rising from the top of the base near the center. Each lug is 18 mm thick in Y, 42 mm tall above the base, and extends 36 mm along X. The two lugs are separated by a 16 mm central gap in Y. The top of each lug has a semicircular rounded profile with radius 18 mm when viewed from the side.

Add a horizontal through-hole of diameter 14 mm through both lugs along the Y direction, centered at X = 0 and Z = 34 mm.

Add four base mounting holes, diameter 7 mm, through the base plate, located at X = +/-45 mm and Y = +/-20 mm.

Add two triangular lightening cutouts through the base web, one on each side of the clevis, each with rounded corners of radius 3 mm.

Add two diagonal reinforcing ribs from the base to the outer faces of the lugs, one on each side, thickness 6 mm.

Add 3 mm fillets to the base perimeter and 2 mm fillets at lug-to-base transitions.

Export as a STEP file.

## Test Cases

| Test | Expected result |
| --- | --- |
| STEP import | File imports successfully |
| Solid count | One watertight fused solid |
| Symmetry | Symmetric about XZ plane |
| Base dimensions | 120 x 60 x 10 mm |
| Lug count | Two |
| Lug thickness | 18 mm in Y each |
| Lug height | 42 mm above base |
| Lug X extent | 36 mm |
| Lug gap | 16 mm central gap in Y |
| Lug top profile | Semicircular/rounded top, radius approximately 18 mm |
| Clevis hole | Diameter 14 mm horizontal through-hole |
| Clevis hole axis | Along Y |
| Clevis hole location | Centered at X = 0, Z = 34 mm |
| Base mounting holes | Four 7 mm diameter vertical through-holes |
| Base hole locations | X = +/-45 mm, Y = +/-20 mm |
| Lightening cutouts | Two triangular cutouts through base/web region |
| Cutout corner radius | Approximately 3 mm rounded corners |
| Reinforcing ribs | Two diagonal ribs, 6 mm thick |
| Fillets | 3 mm base perimeter; 2 mm lug-to-base |
| Negative check | Clevis gap must be open; lugs/ribs must not be detached |
