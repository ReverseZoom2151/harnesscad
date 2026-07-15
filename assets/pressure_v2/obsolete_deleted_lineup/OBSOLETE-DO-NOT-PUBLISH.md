# OBSOLETE -- DO NOT PUBLISH

These files (`part1.json`, `part2.json`, `part3.json`, `results.json`) are the
partial output of running the v2 pressure CODE against a **now-deleted**
six-model lineup (qwen2.5-coder and friends). A previous agent produced them and
was stopped before merging.

They are kept here for forensic reference ONLY. They must never be:

* merged into `../results.json`,
* rendered into a report,
* quoted as a v2 number, or
* compared against v1 as a before/after.

The v2 run will be executed on the FRONTIER lineup -- qwen3.6:27b, qwen3.6:35b,
ornith:9b, ornith:35b -- and only those results may be published. `../merge.py`
refuses any shard whose models are not the frontier set, so it cannot pick these
up by accident.
