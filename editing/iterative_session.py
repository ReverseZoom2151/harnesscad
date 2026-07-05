"""Append-only iterative edit provenance with rollback."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib

@dataclass(frozen=True)
class EditRevision:
    index: int; instruction: str; before_digest: str; after_digest: str; result: object

class IterativeEditSession:
    def __init__(self,initial):
        self.initial=initial;self._revisions=[]
    @staticmethod
    def digest(value):return hashlib.sha256(repr(value).encode()).hexdigest()
    @property
    def current(self):return self._revisions[-1].result if self._revisions else self.initial
    @property
    def revisions(self):return tuple(self._revisions)
    def apply(self,instruction,editor):
        before=self.current;after=editor(before,instruction)
        self._revisions.append(EditRevision(len(self._revisions),instruction,
            self.digest(before),self.digest(after),after))
        return after
    def rollback(self,index):
        if index<0 or index>len(self._revisions):raise IndexError(index)
        del self._revisions[index:]
        return self.current
