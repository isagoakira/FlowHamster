"""
Security tests for FlowHamster backend.

Covers:
1. Path traversal prevention in /api/workflows/{workflow_id}
2. Raw code execution sandboxing in /api/execute
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app."""
    from backend.main import app
    return TestClient(app)


@pytest.fixture
def workflows_dir():
    """Create a temporary workflows directory with a test workflow."""
    from backend.routers import workflows as wf_module
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = wf_module.WORKFLOWS_DIR
        test_dir = Path(tmpdir)
        wf_module.WORKFLOWS_DIR = test_dir

        # Create a test workflow
        wf_dir = test_dir / "test_workflow"
        wf_dir.mkdir(parents=True)
        meta = {
            "id": "test_workflow",
            "name": "Test Workflow",
            "description": "",
            "created_at": 0.0,
            "updated_at": 0.0,
        }
        (wf_dir / "metadata.json").write_text(json.dumps(meta))
        (wf_dir / "document.json").write_text(json.dumps({"version": "1.0"}))

        yield test_dir

        wf_module.WORKFLOWS_DIR = original_dir


# ── Path Traversal Tests ──────────────────────────────────────────────────────

class TestWorkflowPathTraversal:
    """Test that workflow endpoints reject path traversal attempts."""

    def test_is_safe_workflow_id_rejects_dotdot(self):
        """_is_safe_workflow_id should reject '..' sequences."""
        from backend.routers.workflows import _is_safe_workflow_id
        assert _is_safe_workflow_id("..") is False
        assert _is_safe_workflow_id("../etc/passwd") is False
        assert _is_safe_workflow_id("foo/../bar") is False

    def test_is_safe_workflow_id_rejects_absolute_path(self):
        """_is_safe_workflow_id should reject absolute-looking paths."""
        from backend.routers.workflows import _is_safe_workflow_id
        assert _is_safe_workflow_id("/etc/passwd") is False
        assert _is_safe_workflow_id("C:/Windows") is False

    def test_is_safe_workflow_id_rejects_backslash(self):
        """_is_safe_workflow_id should reject backslashes."""
        from backend.routers.workflows import _is_safe_workflow_id
        assert _is_safe_workflow_id("..\\etc\\passwd") is False
        assert _is_safe_workflow_id("foo\\bar") is False

    def test_is_safe_workflow_id_accepts_valid(self):
        """_is_safe_workflow_id should accept normal IDs."""
        from backend.routers.workflows import _is_safe_workflow_id
        assert _is_safe_workflow_id("test_workflow") is True
        assert _is_safe_workflow_id("my-workflow.v2") is True
        assert _is_safe_workflow_id("workflow_123") is True

    def test_get_workflow_rejects_backslash(self, client, workflows_dir):
        """GET /workflows/..\\etc\\passwd should return 400."""
        response = client.get("/api/workflows/..\\etc\\passwd")
        assert response.status_code == 400

    def test_get_workflow_accepts_valid_id(self, client, workflows_dir):
        """GET /workflows/test_workflow should return 200."""
        response = client.get("/api/workflows/test_workflow")
        assert response.status_code == 200
        assert response.json()["id"] == "test_workflow"

    def test_put_workflow_rejects_suspicious_id(self, client, workflows_dir):
        """PUT /workflows/.hidden should return 400 (rejects leading dot)."""
        response = client.put(
            "/api/workflows/.hidden",
            json={"name": "evil"},
        )
        assert response.status_code == 400

    def test_delete_workflow_rejects_suspicious_id(self, client, workflows_dir):
        """DELETE /workflows/.hidden should return 400."""
        response = client.delete("/api/workflows/.hidden")
        assert response.status_code == 400

    def test_delete_workflow_boundary_check(self, client, workflows_dir):
        """Deleting a valid workflow should succeed and respect boundary."""
        response = client.delete("/api/workflows/test_workflow")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_document_get_rejects_suspicious_id(self, client, workflows_dir):
        """GET /workflows/.hidden/document should return 400."""
        response = client.get("/api/workflows/.hidden/document")
        assert response.status_code == 400

    def test_document_put_rejects_suspicious_id(self, client, workflows_dir):
        """PUT /workflows/.hidden/document should return 400."""
        response = client.put(
            "/api/workflows/.hidden/document",
            json={"version": "1.0"},
        )
        assert response.status_code == 400

    def test_list_workflows_not_affected(self, client, workflows_dir):
        """GET /workflows should still list workflows normally."""
        response = client.get("/api/workflows")
        assert response.status_code == 200
        workflows = response.json()
        assert any(w["id"] == "test_workflow" for w in workflows)


# ── Raw Execution Security Tests ──────────────────────────────────────────────

class TestExecuteSecurity:
    """Test that /api/execute enforces security boundaries."""

    def test_raw_execution_disabled_by_default(self, client):
        """Raw code execution without dev mode should return 403."""
        with patch("backend.routers.execute.ALLOW_RAW_EXECUTION", False):
            response = client.post(
                "/api/execute",
                json={"code": "print('hello')"},
            )
            assert response.status_code == 403
            assert "disabled" in response.json()["detail"].lower()

    def test_raw_execution_requires_code_or_graph(self, client):
        """Request with neither code nor graph should return 400."""
        response = client.post(
            "/api/execute",
            json={},
        )
        assert response.status_code == 400

    def test_raw_execution_blocks_disallowed_imports(self, client):
        """Raw code with disallowed imports should be blocked."""
        with patch("backend.routers.execute.ALLOW_RAW_EXECUTION", True):
            response = client.post(
                "/api/execute",
                json={"code": "import os\nos.system('echo pwned')"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "disallowed" in data["error"].lower() or "import" in data["error"].lower()

    def test_raw_execution_allows_whitelisted_imports(self, client):
        """Raw code with whitelisted imports should be allowed in dev mode."""
        with patch("backend.routers.execute.ALLOW_RAW_EXECUTION", True):
            response = client.post(
                "/api/execute",
                json={"code": "import math\nprint(math.pi)"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "3.14" in data["output"]

    def test_compile_execution_path_rejects_no_graph(self, client):
        """Compile path without graph should fall back to raw and be rejected."""
        with patch("backend.routers.execute.ALLOW_RAW_EXECUTION", False):
            response = client.post(
                "/api/execute",
                json={"code": "print('hello')"},
            )
            assert response.status_code == 403

    def test_compile_execution_with_mocked_generate(self, client):
        """Compile path with graph should generate code and execute."""
        with patch(
            "backend.routers.execute._generate_code_from_graph",
            return_value="print('generated_ok')",
        ):
            response = client.post(
                "/api/execute",
                json={
                    "graph": {
                        "nodes": [],
                        "edges": [],
                    },
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "generated_ok" in data["output"]

    def test_sandbox_timeout(self, client):
        """Infinite loop should be killed by timeout."""
        with patch("backend.routers.execute.ALLOW_RAW_EXECUTION", True):
            with patch("backend.routers.execute.EXEC_TIMEOUT", 1):
                response = client.post(
                    "/api/execute",
                    json={"code": "while True: pass"},
                )
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is False
                assert "timed out" in data["error"].lower()

    def test_sandbox_working_directory_isolation(self, client):
        """Code should run inside an isolated temp directory."""
        with patch("backend.routers.execute.ALLOW_RAW_EXECUTION", True):
            response = client.post(
                "/api/execute",
                json={"code": "import pathlib\nprint([p.name for p in pathlib.Path('.').iterdir()])"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            # Should only see the script.py inside the sandbox
            assert "script.py" in data["output"]


# ── Environment / Config Tests ────────────────────────────────────────────────

class TestSecurityConfig:
    """Test security configuration values."""

    def test_allowed_imports_is_not_empty(self):
        from backend.config.security import ALLOWED_IMPORTS
        assert len(ALLOWED_IMPORTS) > 0
        assert "torch" in ALLOWED_IMPORTS
        assert "math" in ALLOWED_IMPORTS

    def test_allowed_env_vars_is_not_empty(self):
        from backend.config.security import ALLOWED_ENV_VARS
        assert len(ALLOWED_ENV_VARS) > 0
        assert "PATH" in ALLOWED_ENV_VARS

    def test_default_raw_execution_disabled(self):
        from backend.config.security import ALLOW_RAW_EXECUTION
        assert ALLOW_RAW_EXECUTION is False
