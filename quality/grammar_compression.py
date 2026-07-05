def compression(instances,templates,rule_bytes=0,instance_bytes=64,template_bytes=1024):
 explicit=len(instances)*template_bytes; compact=len(templates)*template_bytes+len(instances)*instance_bytes+rule_bytes
 return {"explicit_bytes":explicit,"compact_bytes":compact,
 "ratio":explicit/compact if compact else 0,"reuse":len(instances)/len(templates) if templates else 0}
