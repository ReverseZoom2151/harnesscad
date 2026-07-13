from dataclasses import dataclass
@dataclass(frozen=True)
class CodeError: category:str; operation:str; parameter:str|None=None; expected:str|None=None; hint:str|None=None
def normalize(exc,operation="",signature=None):
 if isinstance(exc,TypeError):cat="type"
 elif isinstance(exc,SyntaxError):cat="syntax"
 elif isinstance(exc,ValueError):cat="value"
 else:cat="kernel"
 return CodeError(cat,operation,expected=signature,hint=f"check {signature}" if signature else "inspect operation")
