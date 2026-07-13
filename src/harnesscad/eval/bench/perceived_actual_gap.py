"""Keep perceived ratings distinct from verified engineering feasibility."""
def feasibility_gap(*,perceived,actual_verified):
    if not 0<=perceived<=1:raise ValueError("perceived must be normalized")
    return {"perceived":perceived,"actual_verified":actual_verified,
            "gap":None if actual_verified is None else perceived-float(actual_verified),
            "claim_scope":"perceived_only" if actual_verified is None else "perceived_and_verified"}
