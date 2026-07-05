"""Canonical LCS masking, deterministic infill, and immutable-context guards."""
from __future__ import annotations
MASK="<mask>"

def _lcs(a,b):
    n,m=len(a),len(b); dp=[[()]*(m+1) for _ in range(n+1)]
    for i in range(n-1,-1,-1):
        for j in range(m-1,-1,-1):
            if a[i]==b[j]:dp[i][j]=((i,j),)+dp[i+1][j+1]
            else:
                left,right=dp[i+1][j],dp[i][j+1]
                dp[i][j]=left if len(left)>=len(right) else right
    return dp[0][0]

def locate_mask(original,edited):
    a,b=tuple(original),tuple(edited); matches=_lcs(a,b); out=[]; ai=bi=0
    for mi,mj in matches+((len(a),len(b)),):
        if ai<mi or bi<mj:
            if not out or out[-1]!=MASK:out.append(MASK)
        if mi<len(a):out.append(a[mi])
        ai,bi=mi+1,mj+1
    return tuple(out)

def infill(masked,replacements):
    replacements=iter(replacements); out=[]
    for token in masked:
        if token==MASK:out.extend(tuple(next(replacements)))
        else:out.append(token)
    try:next(replacements);raise ValueError("too many replacements")
    except StopIteration:return tuple(out)

def context_preserved(masked,edited):
    immutable=tuple(x for x in masked if x!=MASK); cursor=iter(edited)
    return all(any(y==x for y in cursor) for x in immutable)
