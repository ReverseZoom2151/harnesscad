def consistency(views,distance):
 keys=sorted(views); pairs=[(a,b,distance(views[a],views[b])) for i,a in enumerate(keys) for b in keys[i+1:]]
 return {"pairs":tuple(pairs),"mean":sum(x[2] for x in pairs)/len(pairs) if pairs else None,
 "worst":max(pairs,key=lambda x:x[2]) if pairs else None}
