"""Confirmation-preserving lifecycle for opaque host script proposals."""
from __future__ import annotations
from dataclasses import dataclass, field, replace

@dataclass(frozen=True)
class HostProposal:
    id: str
    script: object
    request: str
    status: str = "proposed"
    revision: int = 0
    lineage: tuple[str, ...] = ()
    evidence: tuple[dict, ...] = field(default_factory=tuple)

def preview(proposal, host):
    if proposal.status!="proposed": raise ValueError("proposal is not previewable")
    evidence=dict(host.preview(proposal.script))
    return replace(proposal,status="previewed",evidence=proposal.evidence+(evidence,))

def confirm(proposal):
    if proposal.status!="previewed": raise ValueError("proposal must be previewed")
    return replace(proposal,status="confirmed")

def execute(proposal, host):
    if proposal.status!="confirmed": raise PermissionError("fresh confirmation required")
    result=host.execute(proposal.script)
    return replace(proposal,status="executed" if result.ok else "failed",
                   evidence=proposal.evidence+({"ok":result.ok,"message":result.message},))

def refine(proposal, script):
    return HostProposal(f"{proposal.id}-r{proposal.revision+1}",script,proposal.request,
                        revision=proposal.revision+1,
                        lineage=proposal.lineage+(proposal.id,),evidence=proposal.evidence)
