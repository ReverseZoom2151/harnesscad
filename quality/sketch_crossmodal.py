"""Primitive raster versus supplied-image occupancy evidence."""
def crossmodal_consistency(primitives,image,rasterizer):
    expected=set(rasterizer(primitives));actual=set(image)
    union=expected|actual;intersection=expected&actual
    return {"iou":len(intersection)/len(union) if union else 1.0,
            "missing":tuple(sorted(expected-actual)),
            "extra":tuple(sorted(actual-expected)),
            "consistent":expected==actual}
