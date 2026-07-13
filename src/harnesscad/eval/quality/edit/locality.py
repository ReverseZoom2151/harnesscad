def locality(intended,actual,all_entities):
 intended,actual,all_entities=map(set,(intended,actual,all_entities))
 collateral=actual-intended; untouched=all_entities-intended
 preserved=len(untouched-collateral)/len(untouched) if untouched else 1
 return {"precision":len(actual&intended)/len(actual) if actual else float(not intended),"collateral":tuple(sorted(collateral)),"preservation":preserved}
