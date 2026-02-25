from pathlib import Path

from utils.pipeline_state import PipelineState


def test_pipeline_state_resume(tmp_path):
    manifest = tmp_path / "manifest.json"
    state = tmp_path / "state.json"

    ps = PipelineState(str(manifest), str(state))
    ps.initialize_run("run1", {"a": 1}, "2026-02-26T00:00:00Z")

    out = tmp_path / "artifact.txt"
    out.write_text("ok", encoding="utf-8")

    ps.set_node("n1", "completed", outputs=[str(out)], metadata={"k": "v"})
    assert ps.node_completed("n1") is True

    ps.set_job("j1", {"status": "done"})
    assert ps.get_job("j1")["status"] == "done"
