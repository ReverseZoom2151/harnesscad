# 8. Centrifugal Impeller With Backward-Curved Blades

![Centrifugal impeller orbit gif](benchmark_08_centrifugal_impeller.gif)

## Prompt

Create a single solid centrifugal impeller as a STEP model in millimeters.

The impeller axis is vertical along Z and centered at the origin.

Add a circular backplate disk with outside diameter 90 mm and thickness 6 mm, with its bottom face at Z = 0.

Add a central hub cylinder on top of the backplate, diameter 26 mm and height 22 mm above the backplate.

Add a vertical through-bore of diameter 8 mm through the entire part.

Add 12 identical backward-curved blades on top of the backplate, equally spaced around the hub. Each blade begins at radius 18 mm and ends at radius 43 mm. Each blade is 3 mm thick, 16 mm tall above the backplate, and curves backward by approximately 45 degrees from root to tip. The blade tips should lean opposite the direction of rotation when viewed from above.

Add 1 mm fillets at the blade roots where they meet the backplate and hub. Add a 1.5 mm fillet to the top and bottom outer circular edges of the backplate.

Export as a STEP file.

## Test Cases

| Test | Expected result |
| --- | --- |
| STEP import | File imports successfully |
| Solid count | One fused watertight solid |
| Backplate diameter | Diameter 90 mm |
| Backplate thickness | 6 mm |
| Hub diameter | Diameter 26 mm |
| Hub height | 22 mm above backplate |
| Through-bore | Diameter 8 mm through entire part |
| Coaxiality | Backplate, hub, and bore share Z-axis |
| Blade count | 12 |
| Blade spacing | 30 degrees between adjacent blades |
| Blade radial span | Approximately R18 to R43 |
| Blade height | 16 mm above backplate |
| Blade thickness | Approximately 3 mm |
| Blade curvature | Approximately 45 degrees backward from root to tip |
| Blade fusion | Blades joined to backplate and hub |
| Fillets | 1 mm blade-root fillets; 1.5 mm backplate edge fillets |
| Negative check | No straight radial-only blades unless curvature requirement is otherwise approximated; no floating blades |
