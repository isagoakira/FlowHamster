import json
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException, FastAPI
from fastapi.testclient import TestClient

from backend.routers import workflows


class WorkflowRouterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_workflows_dir = workflows.WORKFLOWS_DIR
        workflows.WORKFLOWS_DIR = (Path(self.tmp.name) / "workflows").resolve()

        app = FastAPI()
        app.include_router(workflows.router, prefix="/api")
        self.client = TestClient(app)

    def tearDown(self):
        workflows.WORKFLOWS_DIR = self.original_workflows_dir
        self.tmp.cleanup()

    def test_workflow_crud_uses_safe_resolved_directory(self):
        create_response = self.client.post(
            "/api/workflows",
            json={"name": "safe workflow", "description": "demo"},
        )

        self.assertEqual(create_response.status_code, 200)
        created = create_response.json()
        self.assertEqual(created["id"], "safe_workflow")

        get_response = self.client.get(f"/api/workflows/{created['id']}")
        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(get_response.json()["name"], "safe workflow")

        update_response = self.client.put(
            f"/api/workflows/{created['id']}/document",
            json={"modelGraph": {"nodes": [], "edges": []}},
        )
        self.assertEqual(update_response.status_code, 200)
        self.assertEqual(update_response.json(), {"status": "ok"})

        doc_path = workflows.WORKFLOWS_DIR / created["id"] / "document.json"
        saved_document = json.loads(doc_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_document["modelGraph"], {"nodes": [], "edges": []})

        delete_response = self.client.delete(f"/api/workflows/{created['id']}")
        self.assertEqual(delete_response.status_code, 200)
        self.assertFalse((workflows.WORKFLOWS_DIR / created["id"]).exists())

    def test_workflow_id_rejects_parent_directory_traversal(self):
        with self.assertRaises(HTTPException) as ctx:
            workflows._resolve_workflow_dir("../outside")

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "Invalid workflow id")

    def test_delete_rejects_root_or_parent_escape_without_removing_outside_data(self):
        outside_dir = Path(self.tmp.name) / "outside"
        outside_dir.mkdir()
        sentinel = outside_dir / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")

        for workflow_id in ("..", "../outside", str(outside_dir)):
            with self.assertRaises(HTTPException):
                workflows._resolve_workflow_dir(workflow_id)

        self.assertTrue(sentinel.exists())


if __name__ == "__main__":
    unittest.main()
