def classify(*,components,loops,curves,type_diversity,feature_depth,repeated=0):
 score=components+loops+curves/5+type_diversity+2*feature_depth-repeated*.5
 level=min(5,max(1,int(score//8)+1))
 return {"level":level,"score":score,"evidence":{"components":components,"loops":loops,"curves":curves,"type_diversity":type_diversity,"feature_depth":feature_depth,"repeated":repeated}}
