"""Inspector explanations use the selected record's retrieval view."""

import pytest

from tests.test_control_plane import _facade_service


@pytest.mark.parametrize("captured", [False, True])
def test_inspector_explains_selected_record_in_its_own_view(tmp_path, captured):
    service, _ = _facade_service(tmp_path)
    view = "captured_evidence" if captured else "memory"
    channel = "captured_evidence" if captured else "facts"
    calls = []

    def request(operation, payload):
        calls.append((operation, payload))
        if operation == "inspect":
            return {
                "fact": {
                    "id": "selected",
                    "content": "release convention",
                    "metadata": {
                        "source": "captured_passthrough" if captured else "explicit_remember",
                        "scope": "project",
                        "scope_id": "project-1",
                    },
                }
            }
        assert operation == "retrieve"
        assert payload["view"] == view
        return {channel: [{"id": "selected", "scores": {"semantic_rank": 1}}]}

    service._service_request = request
    result = service.get_memory_detail("selected", query="release convention")
    assert result["retrieval"]["status"] == "observed"
    assert result["retrieval"]["view"] == view
    assert result["retrieval"]["semantic_rank"] == 1
    assert calls[0][0] == "inspect"
    assert calls[1][1]["context"]["user_profile_id"] == "alice"


def test_control_plane_forwards_explicit_evidence_view(tmp_path):
    service, client = _facade_service(tmp_path)
    service.retrieve_memory("release convention", view="captured_evidence")
    assert client.calls[-1][1]["view"] == "captured_evidence"
