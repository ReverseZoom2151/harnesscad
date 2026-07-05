def metrics(normal_pairs,parameter_errors,relationships,hanging_faces,total_faces):
 nc=sum(abs(sum(a*b for a,b in zip(x,y))) for x,y in normal_pairs)/len(normal_pairs) if normal_pairs else None
 return {"normal_consistency":nc,
 "parameter_error":sum(parameter_errors)/len(parameter_errors) if parameter_errors else None,
 "relationship_satisfaction":sum(relationships)/len(relationships) if relationships else None,
 "hanging_face_rate":hanging_faces/total_faces if total_faces else 0}
