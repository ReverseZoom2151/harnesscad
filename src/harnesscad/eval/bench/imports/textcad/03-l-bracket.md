# 3. L-Bracket With Gussets and Two Hole Directions

![L-bracket orbit gif](benchmark_03_l_bracket.gif)

## Prompt

Create a single solid L-bracket STEP model in millimeters.

The bracket has a horizontal base plate 80 mm long in X, 50 mm wide in Y, and 8 mm thick in Z. Center the base plate on the XY origin, with its bottom at Z = 0.

Add a vertical back plate along the rear long edge of the base. The back plate is 80 mm long in X, 8 mm thick in Y, and 50 mm tall in Z, rising from the top of the base plate. The back plate should sit along the rear edge at positive Y.

Add two vertical through-holes in the base plate, each 6 mm in diameter, located at X = +/-25 mm and Y = -10 mm.

Add two horizontal through-holes in the vertical plate, each 6 mm in diameter, located at X = +/-25 mm and Z = 30 mm, passing through the 8 mm thickness of the vertical plate.

Add two triangular gussets, each 8 mm thick in X, located at X = +/-20 mm. Each gusset should connect the base plate to the back plate with a right-triangle side profile 30 mm tall and 30 mm deep.

Add 2 mm fillets to the outside corner where the base and back plate meet.

Export as a STEP file.

## Test Cases

| Test | Expected result |
| --- | --- |
| STEP import | File imports successfully |
| Solid count | Exactly one fused watertight solid |
| Base dimensions | 80 x 50 x 8 mm |
| Back plate dimensions | 80 x 8 x 50 mm |
| Back plate placement | Along positive-Y rear edge of base |
| Base hole count | Two |
| Base hole geometry | 6 mm diameter vertical through-holes |
| Base hole locations | X = +/-25 mm, Y = -10 mm |
| Back hole count | Two |
| Back hole geometry | 6 mm diameter horizontal through-holes through Y thickness |
| Back hole locations | X = +/-25 mm, Z = 30 mm |
| Gusset count | Two |
| Gusset geometry | Right-triangle side profile, approximately 30 x 30 mm |
| Gusset placement | At X = +/-20 mm |
| Fillets | 2 mm fillets at base/back outside transition |
| Negative check | No floating gussets; no unfused plates; holes must actually cut through |
