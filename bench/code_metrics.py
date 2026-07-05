"""Non-executing Python AST metrics for generated CAD scripts."""

from __future__ import annotations
import ast
from collections import Counter


def parsing_rate(sources) -> float|None:
    values=tuple(sources)
    if not values:return None
    return sum(_parse(x) is not None for x in values)/len(values)


def _parse(source):
    try:return ast.parse(source)
    except (SyntaxError,TypeError):return None


def function_calls(source):
    tree=_parse(source)
    if tree is None:return None
    names=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            fn=node.func
            names.append(fn.id if isinstance(fn,ast.Name) else
                         fn.attr if isinstance(fn,ast.Attribute) else "<dynamic>")
    return Counter(names)


def function_accuracy(expected, actual) -> dict:
    a,b=function_calls(expected),function_calls(actual)
    if a is None or b is None:return {"available":False}
    matched=sum((a&b).values()); na=sum(a.values()); nb=sum(b.values())
    p=matched/nb if nb else float(not na); r=matched/na if na else float(not nb)
    return {"available":True,"precision":p,"recall":r,
            "f1":2*p*r/(p+r) if p+r else 0.0,"exact":a==b}


def _calls(source):
    tree=_parse(source)
    if tree is None:return []
    out=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Call):
            name=n.func.id if isinstance(n.func,ast.Name) else getattr(n.func,"attr","<dynamic>")
            args=tuple(ast.dump(x,include_attributes=False) for x in n.args)
            kws=tuple(sorted((x.arg,ast.dump(x.value,include_attributes=False)) for x in n.keywords))
            out.append((name,args,kws))
    return out


def parameter_accuracy(expected, actual) -> dict:
    a,b=_calls(expected),_calls(actual); matched=total=0
    for left,right in zip(a,b):
        if left[0]!=right[0]:continue
        lp=left[1]+left[2]; rp=right[1]+right[2]; total+=len(lp)
        matched+=sum(x==y for x,y in zip(lp,rp))
    return {"matched":matched,"total":total,"accuracy":matched/total if total else None}


def annotation_accuracy(expected, actual) -> dict:
    """Annotation records are (kind, value); report kind and exact-value errors."""
    a,b=tuple(expected),tuple(actual); total=max(len(a),len(b)); kinds=values=0
    for left,right in zip(a,b):
        kinds+=left[0]==right[0]; values+=left==right
    return {"accuracy":values/total if total else 1.0,
            "type_error_rate":(total-kinds)/total if total else 0.0,
            "data_error_rate":(kinds-values)/total if total else 0.0}
