"""Exercise the harnesscad.sdk facade for real -- not just import it.

The facade adds no behaviour, so every test here drives the RE-EXPORTED component:
an Environment reset/step, a bench score, an op round-trip, an MCP handshake, and
the lazy-import contract that keeps `import harnesscad` kernel-free.

NOTE: constructing an Environment pulls the (lazy) render hook, which imports
cadquery; cadquery is known to segfault at interpreter teardown (exit 139). Read
the PRINTED pytest results, not the process exit code.
"""

import subprocess
import sys

import harnesscad
from harnesscad import sdk


def test_top_level_names_resolve_to_facade():
    # The headline names are importable at top level and are the same objects the
    # facade exposes (lazy package __getattr__).
    assert harnesscad.Op is sdk.Op
    assert harnesscad.parse_op is sdk.parse_op
    assert harnesscad.score is sdk.score
    assert harnesscad.Environment is sdk.Environment


def test_parse_op_load_ops_round_trip():
    ops = sdk.load_ops([
        {"op": "new_sketch", "plane": "XY"},
        {"op": "add_circle", "sketch": "s0", "cx": 1.0, "cy": 2.0, "r": 3.0},
    ])
    assert [o.OP for o in ops] == ["new_sketch", "add_circle"]
    # Round-trips through dict form: load_ops is the batch inverse of to_dict.
    again = sdk.load_ops([o.to_dict() for o in ops])
    assert again == ops
    # An Op passes straight through unchanged.
    assert sdk.load_ops([ops[0]])[0] is ops[0]
    # The op vocabulary is the real registry.
    vocab = sdk.op_vocabulary()
    assert "new_sketch" in vocab and "extrude" in vocab


def test_score_through_bench_dispatch():
    # A trivial identical pred/gold scores a perfect parameter accuracy.
    result = sdk.score("sequence.parameter_accuracy",
                       {"params": {"a": 1.0, "b": 2.0}},
                       {"params": {"a": 1.0, "b": 2.0}})
    assert result["accuracy"] == 1.0
    assert result["matched"] == 2.0
    # metric(name) is the same dispatch object.
    assert sdk.metric("sequence.parameter_accuracy").name == "sequence.parameter_accuracy"
    assert len(sdk.metrics()) > 0
    assert isinstance(sdk.suites(), tuple)


def test_environment_reset_and_step():
    env = sdk.Environment("stub", max_steps=3)
    obs = env.reset()
    assert isinstance(obs, dict)
    for key in ("feature_tree", "sketch_dof", "validity", "digest", "step"):
        assert key in obs

    op = sdk.parse_op({"op": "new_sketch", "plane": "XY"})
    out = env.step(op)
    assert isinstance(out, tuple) and len(out) == 4
    obs2, reward, done, info = out
    assert isinstance(obs2, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert info["ok"] is True
    assert info["applied"] == 1
    assert reward == 1.0  # a clean applied+verified op
    env.close()


def test_session_apply_and_measure():
    session = sdk.Session("stub")
    result = session.apply_ops([sdk.parse_op({"op": "new_sketch", "plane": "XY"})])
    assert result.ok is True
    assert result.applied == 1
    assert isinstance(session.digest(), str)
    assert isinstance(session.summary(), dict)


def test_mcp_server_initialize_handshake():
    server = sdk.mcp_server("stub")
    # The real JSON-RPC dispatch: an initialize request returns capabilities.
    reply = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                           "params": {"protocolVersion": "2025-11-25"}})
    assert reply["id"] == 1
    result = reply["result"]
    caps = result["capabilities"]
    assert isinstance(caps, dict)
    assert "tools" in caps and "resources" in caps and "prompts" in caps
    assert result["serverInfo"]["name"] == "harnesscad"


def test_trajectory_and_task_suite_reexports():
    from harnesscad.agents.agent.tool_trajectory import ToolTrajectory
    assert sdk.Trajectory is ToolTrajectory
    traj = sdk.Trajectory()
    assert len(traj) == 0

    suite = sdk.TaskSuite.dev()
    assert len(suite) > 0
    first_id = suite.ids()[0]
    brief = suite.by_id(first_id)
    assert isinstance(brief, sdk.Brief)
    assert brief.id == first_id
    # load_tasks() is the same underlying corpus tuple.
    assert len(sdk.load_tasks()) == len(suite)


def test_import_harnesscad_is_kernel_free():
    # A fresh interpreter: `import harnesscad` must NOT eagerly import a geometry
    # kernel. Run in a subprocess so pytest's own imports cannot contaminate the
    # check.
    code = (
        "import sys, harnesscad\n"
        "assert 'cadquery' not in sys.modules, 'cadquery imported eagerly'\n"
        "assert 'OCP' not in sys.modules, 'OCP imported eagerly'\n"
        "# Touching a headline name must also stay kernel-free.\n"
        "_ = harnesscad.Op\n"
        "assert 'cadquery' not in sys.modules, 'cadquery imported by name access'\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
