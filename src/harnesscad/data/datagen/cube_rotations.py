"""The 24 proper cube rotations and coordinate-aware call rewriting."""
from __future__ import annotations
import itertools

def cube_rotations():
    out=[]
    for perm in itertools.permutations(range(3)):
        parity = 1 if perm in ((0,1,2),(1,2,0),(2,0,1)) else -1
        for signs in itertools.product((-1,1), repeat=3):
            if parity*signs[0]*signs[1]*signs[2] == 1:
                matrix=tuple(tuple(signs[row] if perm[row]==col else 0
                                   for col in range(3)) for row in range(3))
                out.append(matrix)
    return tuple(sorted(set(out)))

def apply_rotation(matrix, vector):
    return tuple(sum(matrix[i][j]*vector[j] for j in range(3)) for i in range(3))

def inverse_rotation(matrix):
    return tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3))

def rewrite_calls(calls, matrix, *, workplane_kinds=("workplane",),
                  global_kinds=("translate_global","rotate_axis","center_global")):
    out=[]
    for call in calls:
        item=dict(call); args=dict(item.get("args", {}))
        if item.get("kind") in workplane_kinds:
            for key in ("origin","normal","x_direction"):
                if key in args: args[key]=apply_rotation(matrix, tuple(args[key]))
        elif item.get("kind") in global_kinds:
            for key in ("vector","axis","point"):
                if key in args: args[key]=apply_rotation(matrix, tuple(args[key]))
        item["args"]=args; out.append(item)
    return tuple(out)

def rotation_variant(calls, *, seed):
    rotations=cube_rotations(); index=seed % len(rotations); matrix=rotations[index]
    return {"index":index,"matrix":matrix,"inverse":inverse_rotation(matrix),
            "calls":rewrite_calls(calls,matrix)}
