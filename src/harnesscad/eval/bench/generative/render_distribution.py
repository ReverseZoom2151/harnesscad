def summarize(items,feature_distance):
 if not items:return {"count":0,"quality":None,"prompt":None,"diversity":None}
 ordered=sorted(items,key=lambda x:x["id"])
 pairs=[feature_distance(a["feature"],b["feature"]) for i,a in enumerate(ordered) for b in ordered[i+1:]]
 return {"count":len(items),"quality":sum(x["quality"] for x in items)/len(items),
 "prompt":sum(x["prompt"] for x in items)/len(items),
 "diversity":sum(pairs)/len(pairs) if pairs else 0.0}
