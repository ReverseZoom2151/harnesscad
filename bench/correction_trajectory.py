def score(steps):
 valid=[bool(x["valid"]) for x in steps];align=[x.get("alignment",0) for x in steps]
 first=next((i for i,x in enumerate(valid) if x),None)
 return {"first_pass":bool(valid and valid[0]),"iterations_to_valid":first,"recovered":first is not None and first>0,
 "regressions":sum(valid[i-1] and not valid[i] for i in range(1,len(valid))),
 "monotonic":all(b>=a for a,b in zip(align,align[1:]))}
