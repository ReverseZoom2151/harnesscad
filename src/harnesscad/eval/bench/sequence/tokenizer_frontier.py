def evaluate(name,source_tokens,encoded_tokens,reconstructed,invalid=False):
 return {"name":name,"compression":len(source_tokens)/len(encoded_tokens) if encoded_tokens else None,"reconstruction":sum(a==b for a,b in zip(source_tokens,reconstructed))/max(len(source_tokens),len(reconstructed),1),"invalid":int(invalid)}
def frontier(rows):
 return tuple(r for r in sorted(rows,key=lambda x:x["name"]) if not any((o["compression"] or 0)>=(r["compression"] or 0) and o["reconstruction"]>=r["reconstruction"] and o["invalid"]<=r["invalid"] and o!=r for o in rows))
