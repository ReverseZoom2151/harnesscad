from math import cos,sin,tau
def linear(count,spacing): return tuple((i*spacing,0,0) for i in range(count))
def grid(rows,cols,spacing): return tuple((c*spacing[0],r*spacing[1],0) for r in range(rows) for c in range(cols))
def radial(count,radius): return tuple((radius*cos(tau*i/count),radius*sin(tau*i/count),0) for i in range(count))
def pipe(points):
 if len(points)<2: raise ValueError("two points required")
 return tuple((points[i],points[i+1]) for i in range(len(points)-1))
