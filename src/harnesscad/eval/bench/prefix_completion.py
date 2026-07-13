def cuts(sequence,ratios=(.2,.4,.6,.8)):
 return tuple((r,tuple(sequence[:max(1,min(len(sequence),round(len(sequence)*r)))]),tuple(sequence[max(1,min(len(sequence),round(len(sequence)*r))):])) for r in ratios)
def auc(points):
 p=sorted(points);return sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(p,p[1:]))
