def metrics(actual,expected,order=False):
 ids=sorted(set(actual)|set(expected)); tp=na=ne=strict=0
 for i in ids:
  a=list(actual.get(i,()));e=list(expected.get(i,()));na+=len(a);ne+=len(e)
  if order:n=sum(x==y for x,y in zip(a,e))
  else:
   rem=list(e);n=0
   for x in a:
    if x in rem:n+=1;rem.remove(x)
  tp+=n;strict+=a==e if order else sorted(map(str,a))==sorted(map(str,e))
 p=tp/na if na else float(not ne);r=tp/ne if ne else float(not na)
 return {"precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0,"sketch_accuracy":strict/len(ids) if ids else 1}
