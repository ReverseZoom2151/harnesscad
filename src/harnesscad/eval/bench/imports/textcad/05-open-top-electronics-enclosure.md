# 5. Open-Top Electronics Enclosure With Bosses

![Open-top electronics enclosure orbit gif](benchmark_05_open_top_electronics_enclosure.gif)

## Prompt

Create a single solid open-top electronics enclosure base as a STEP model in millimeters.

The outer shape is a rectangular box 100 mm long in X, 70 mm wide in Y, and 30 mm tall in Z. Center it on the XY origin, with the bottom face at Z = 0.

The enclosure is open at the top. The wall thickness is 3 mm and the bottom floor thickness is 3 mm.

Add four internal cylindrical standoffs rising from the inside floor. Each standoff has an outside diameter of 10 mm and a height of 12 mm above the inside floor. Place the standoffs at X = +/-35 mm and Y = +/-25 mm.

Add a centered blind hole in each standoff, 3 mm in diameter and 8 mm deep from the top of the standoff.

Add 2 mm radius fillets to the four outside vertical corners of the enclosure.

Export as a STEP file.

## Test Cases

| Test | Expected result |
| --- | --- |
| STEP import | File imports successfully |
| Solid count | One solid body representing the enclosure material |
| Outer dimensions | 100 x 70 x 30 mm |
| Placement | Centered on XY origin; bottom at Z = 0 |
| Top condition | Open top, not capped |
| Wall thickness | 3 mm |
| Floor thickness | 3 mm |
| Interior cavity | Correctly hollowed, with continuous walls and floor |
| Standoff count | Four |
| Standoff diameter | Diameter 10 mm |
| Standoff height | 12 mm above inside floor |
| Standoff locations | X = +/-35 mm, Y = +/-25 mm |
| Blind hole count | Four |
| Blind hole diameter | Diameter 3 mm |
| Blind hole depth | 8 mm from standoff top |
| Corner fillets | 2 mm radius on four outside vertical corners |
| Negative check | Blind holes must not break through floor; standoffs must not float |
