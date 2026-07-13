def assess(raw,upscaled,perceptual,depth_drift,max_depth_drift):
 appearance=perceptual(raw,upscaled); drift=depth_drift(raw,upscaled)
 return {"appearance_distance":appearance,"depth_drift":drift,
 "accepted":drift<=max_depth_drift,"reasons":() if drift<=max_depth_drift else ("geometry_drift",)}
