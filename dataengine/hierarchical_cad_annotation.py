def validate(components,captions,summary_refs):
 ids=set(components); issues=[]
 if len(captions)!=len(set(captions)):issues.append("duplicate_caption")
 issues += [f"missing:{x}" for x in sorted(ids-set(captions))]
 issues += [f"orphan:{x}" for x in sorted(set(captions)-ids)]
 issues += [f"unreferenced:{x}" for x in sorted(ids-set(summary_refs))]
 issues += [f"stale:{x}" for x in sorted(set(summary_refs)-ids)]
 return tuple(issues)
