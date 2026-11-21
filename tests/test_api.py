import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def test_api_roundtrip(dq, monkeypatch):
    import docquery.api.main as api

    monkeypatch.setattr(api, "get_dq", lambda: dq)
    client = TestClient(api.app)
    assert client.get("/health").json()["chunks"] == len(dq.chunks)
    r = client.post("/ask", json={"question": "How many credits are needed for the Bachelor of Engineering degree?"})
    body = r.json()
    assert r.status_code == 200 and "160" in body["answer"] and body["citations"]
