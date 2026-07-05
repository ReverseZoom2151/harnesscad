import hashlib,json
def cache_key(**parts):
 return hashlib.sha256(json.dumps(parts,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
class PriorCache:
 def __init__(self):self._data={}
 def put(self,key,value,metadata=None):
  if key in self._data: raise ValueError("immutable cache entry")
  self._data[key]=(value,dict(metadata or {}))
 def get(self,key):return self._data.get(key)
