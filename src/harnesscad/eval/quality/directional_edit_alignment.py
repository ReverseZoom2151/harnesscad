from math import sqrt
def alignment(source_text,edited_text,source_image,edited_image):
 dt=[b-a for a,b in zip(source_text,edited_text)]; di=[b-a for a,b in zip(source_image,edited_image)]
 den=sqrt(sum(x*x for x in dt)*sum(x*x for x in di))
 return sum(a*b for a,b in zip(dt,di))/den if den else None
def rank(items): return tuple(sorted(items,key=lambda x:(not x["valid"],-(x["score"] if x["score"] is not None else -2),x["id"])))
