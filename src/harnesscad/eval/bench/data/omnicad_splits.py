"""Parent-lineage leakage and modality-coverage audits."""
def audit_omnicad_splits(records,required=("text","image","point")):
    owners={};leaks=[];missing=[]
    for record in records:
        key=record.parent_id or record.id
        old=owners.setdefault(key,record.split)
        if old!=record.split:leaks.append(key)
        absent=set(required)-set(record.modalities)
        if absent:missing.append((record.id,tuple(sorted(absent))))
    return {"lineage_leakage":tuple(sorted(set(leaks))),"missing_modalities":tuple(sorted(missing)),
            "ok":not leaks and not missing}
