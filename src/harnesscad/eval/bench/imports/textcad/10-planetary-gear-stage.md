# 10. Simplified Planetary Gear Stage

![Planetary gear stage orbit gif](benchmark_10_planetary_gear_stage.gif)

## Prompt

Create a visually clear simplified planetary gear assembly as a STEP model in millimeters.

The assembly lies flat in the XY plane with gear axes along Z. Use separate solid bodies for the sun gear, three planet gears, ring gear, carrier plate, and three planet pins.

All gears are 8 mm thick. Use simplified straight-sided trapezoidal teeth rather than true involute teeth.

The sun gear has 24 external teeth, pitch diameter 48 mm, root diameter 42 mm, and outside diameter 54 mm.

The three planet gears each have 18 external teeth, pitch diameter 36 mm, root diameter 31 mm, and outside diameter 41 mm. Place the planet gear centers on a 42 mm radius circle, equally spaced every 120 degrees.

The ring gear is concentric with the sun gear, has 60 internal teeth, internal pitch diameter 120 mm, internal root diameter 126 mm, internal tooth-tip diameter 114 mm, and outside diameter 140 mm.

Add a thin circular carrier plate below the gears, diameter 105 mm and thickness 4 mm, located from Z = -5 mm to Z = -1 mm.

Add three vertical planet pins, each diameter 6 mm and height 14 mm, centered under the planet gears.

Add a central sun bore of diameter 10 mm.

Export as a STEP file.

## Test Cases

| Test | Expected result |
| --- | --- |
| STEP import | File imports successfully |
| Body count | Separate bodies for sun, 3 planets, ring, carrier, and 3 pins |
| Gear axes | All gear axes parallel to Z |
| Gear thickness | 8 mm for sun, planet, and ring gears |
| Sun tooth count | 24 external teeth |
| Sun diameters | Pitch approximately 48 mm diameter; root 42 mm diameter; outside 54 mm diameter |
| Sun bore | 10 mm diameter central through-bore |
| Planet count | Three planet gears |
| Planet tooth count | 18 teeth each |
| Planet diameters | Pitch approximately 36 mm diameter; root 31 mm diameter; outside 41 mm diameter |
| Planet placement | Centers on R42 mm circle |
| Planet spacing | 120 degrees apart |
| Ring gear | 140 mm outside diameter |
| Ring internal teeth | 60 internal teeth |
| Ring diameters | Internal pitch approximately 120 mm diameter; internal root 126 mm diameter; tooth-tip 114 mm diameter |
| Carrier plate | Diameter 105 mm, 4 mm thick, Z = -5 to -1 mm |
| Pins | Three 6 mm diameter pins aligned with planet centers |
| Negative check | Ring teeth must face inward; planet gears should not be fused into sun or ring |
