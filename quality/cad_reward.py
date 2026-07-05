"""Execution-first CAD reward with exact paper thresholds and format gate."""
from __future__ import annotations
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class CADReward:
    total: float; geometric: float; format: float; cd: float|None; executable: bool; reason: str

def geometric_reward(cd):
    if cd is None or cd>0.5:return 0.0
    if cd<1e-5:return 1.0
    return 1.0-(cd-1e-5)*(0.99/(0.5-1e-5))

def format_reward(text):
    return float(bool(re.search(r"<think>.*?</think>",text,re.S|re.I) and
                      re.search(r"```python\s+.+?```",text,re.S|re.I)))

def score_candidate(text, *, execute, sample, target_points, distance,
                    geometry_weight=1.0, format_weight=1.0):
    fmt=format_reward(text)
    try: shape=execute(text)
    except Exception:return CADReward(format_weight*fmt,0,fmt,None,False,"execution-failed")
    if shape is None:return CADReward(format_weight*fmt,0,fmt,None,False,"execution-failed")
    try: cd=distance(sample(shape),target_points)
    except Exception:return CADReward(format_weight*fmt,0,fmt,None,True,"geometry-failed")
    geo=geometric_reward(cd)
    return CADReward(geometry_weight*geo+format_weight*fmt,geo,fmt,cd,True,"ok")
