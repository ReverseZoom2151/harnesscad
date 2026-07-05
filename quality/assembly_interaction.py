"""Separate allowed contact, clearance violations, and forbidden collisions."""
def classify_interactions(interactions,*,allowed_contacts=(),minimum_clearance=0):
    allowed={tuple(sorted(x)) for x in allowed_contacts};out=[]
    for item in interactions:
        pair=tuple(sorted(item["faces"]));distance=float(item["distance"])
        if pair in allowed:
            kind="allowed_contact" if distance<=minimum_clearance else "contact_gap"
        else:
            kind="forbidden_collision" if distance<0 else (
                "clearance_violation" if distance<minimum_clearance else "clear")
        out.append({**item,"classification":kind})
    return tuple(out)
