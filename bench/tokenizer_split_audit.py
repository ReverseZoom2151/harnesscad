def audit(tokenizer_train,backbone_train):
 t,b=set(tokenizer_train),set(backbone_train)
 return {"nested":t<=b,"missing_from_backbone":tuple(sorted(t-b)),"heldout_exposure":tuple(sorted(b-t)),"has_heldout_exposure":bool(b-t)}
