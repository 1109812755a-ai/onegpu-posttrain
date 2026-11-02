from fastapi.testclient import TestClient

from onegpu_posttrain.ui.server import create_app


def test_generation_status_snapshot_is_stable_read_only_and_reports_missing_evidence(tmp_path):
    app = create_app(tmp_path, enable_synthetic_generation=True)
    state = {
        "execution_id": "prefgenrun-" + "a" * 20,
        "bundle_id": "plan-preference-test", "mode": "ordinary", "status": "partial",
        "created_at": "2026-10-29T00:00:00Z", "updated_at": "2026-10-29T00:01:00Z",
        "progress": {"completed": 1, "target": 2}, "request_count": 1,
        "request_budget": 2, "source_dataset_id": "seed-dataset",
        "result": {"request_errors": {"timeout": 1}},
    }
    reads = []

    class ReadOnlyJobs:
        def get(self, execution_id):
            reads.append(execution_id)
            return state if execution_id == state["execution_id"] else None

        def shutdown(self):
            return None

    app.state.preference_generation_jobs = ReadOnlyJobs()
    with TestClient(app) as client:
        token = app.state.synthetic_generation_token
        response1 = client.get(
            f"/api/synthetic/status/preference/{state['execution_id']}",
            headers={"X-Local-Synthetic-Token": token},
        )
        response2 = client.get(
            f"/api/synthetic/status/preference/{state['execution_id']}",
            headers={"X-Local-Synthetic-Token": token},
        )
        assert response1.status_code == response2.status_code == 200
        snapshot = response1.json()
        assert snapshot == response2.json()
        assert snapshot["status"] == "partial"
        assert snapshot["inputs"]["source_dataset_id"] == "seed-dataset"
        assert snapshot["provider_requests"]["errors"] == {"timeout": 1}
        assert snapshot["missing_evidence"] == ["registered_dataset"]
        assert snapshot["read_only"] is True
        assert reads == [state["execution_id"], state["execution_id"]]

        denied = client.get(f"/api/synthetic/status/preference/{state['execution_id']}")
        assert denied.status_code == 403
        unknown = client.get("/api/synthetic/status/grpo/unknown",
                             headers={"X-Local-Synthetic-Token": token})
        assert unknown.status_code == 404
