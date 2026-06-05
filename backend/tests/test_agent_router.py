import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.routers import agent
from backend.services.agent_orchestrator import AgentOrchestrator, SQLiteConversationStore


def make_client(max_history_messages: int = 20):
    tmp = tempfile.TemporaryDirectory()
    store = SQLiteConversationStore(Path(tmp.name) / "agent.sqlite3")
    original = agent.orchestrator
    agent.orchestrator = AgentOrchestrator(store, max_history_messages=max_history_messages)
    app = FastAPI()
    app.include_router(agent.router, prefix="/api")
    return TestClient(app), tmp, original


def simple_graph():
    return {
        "nodes": [
            {"id": "input", "data": {"nodeType": "input", "label": "Input", "params": {"name": "features"}}},
            {
                "id": "fc",
                "data": {
                    "nodeType": "linear",
                    "label": "Linear",
                    "params": {"in_features": 4, "out_features": 2, "bias": True},
                },
            },
            {"id": "output", "data": {"nodeType": "output", "label": "Output", "params": {}}},
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "fc", "sourceHandle": "result", "targetHandle": "x"},
            {"id": "e2", "source": "fc", "target": "output", "sourceHandle": "result", "targetHandle": "x"},
        ],
    }


class AgentRouterTest(unittest.TestCase):
    def test_agent_creates_session_and_keeps_context(self):
        client, tmp, original = make_client()
        try:
            first = client.post("/api/agent/chat", json={"message": "Help me inspect this model"})
            self.assertEqual(first.status_code, 200)
            session_id = first.json()["session_id"]

            second = client.post(
                "/api/agent/chat",
                json={
                    "session_id": session_id,
                    "message": "Validate the graph",
                    "graph_context": {"model_graph": simple_graph()},
                },
            )

            self.assertEqual(second.status_code, 200)
            self.assertEqual(second.json()["session_id"], session_id)
            sessions = client.get("/api/agent/sessions").json()
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0]["title"], "Help me inspect this model")
            detail = client.get(f"/api/agent/sessions/{session_id}").json()
            self.assertGreaterEqual(detail["message_count"], 3)
            self.assertIn("Validate the graph", [msg["content"] for msg in detail["messages"]])
        finally:
            agent.orchestrator = original
            tmp.cleanup()

    def test_agent_tool_chain_validates_and_generates_code(self):
        client, tmp, original = make_client()
        try:
            response = client.post(
                "/api/agent/chat",
                json={
                    "message": "Validate and generate code",
                    "graph_context": {"model_graph": simple_graph()},
                },
            )
            body = response.json()

            self.assertEqual(response.status_code, 200)
            self.assertTrue(body["success"])
            self.assertEqual([call["name"] for call in body["tool_calls"]], ["graph.validate", "code.generate"])
            self.assertTrue(all(obs["success"] for obs in body["observations"]))
            generated = body["observations"][1]["output"]["code"]
            self.assertIn("class FlowHamsterModel", generated)
            self.assertIn("Graph validated and code generation completed", body["reply"])
        finally:
            agent.orchestrator = original
            tmp.cleanup()

    def test_agent_blocks_invalid_graph_mutation(self):
        client, tmp, original = make_client()
        try:
            response = client.post(
                "/api/agent/chat",
                json={
                    "message": "connect missing to output",
                    "graph_context": {"model_graph": simple_graph()},
                },
            )
            body = response.json()

            self.assertEqual(response.status_code, 200)
            self.assertFalse(body["success"])
            self.assertTrue(body["blocked"])
            self.assertEqual(body["tool_calls"][0]["name"], "graph.mutation")
            output = body["observations"][0]["output"]
            self.assertEqual(output["diff"]["operations"], [])
            self.assertIn("Use an existing node id", output["corrections"][0])
        finally:
            agent.orchestrator = original
            tmp.cleanup()

    def test_agent_requires_confirmation_for_delete_node_and_returns_diff_after_token(self):
        client, tmp, original = make_client()
        try:
            first = client.post(
                "/api/agent/chat",
                json={
                    "message": "delete node fc",
                    "graph_context": {"model_graph": simple_graph()},
                },
            )
            first_body = first.json()

            self.assertEqual(first.status_code, 200)
            self.assertTrue(first_body["requires_confirmation"])
            self.assertIsNotNone(first_body["confirmation_token"])
            self.assertEqual(first_body["tool_calls"][0]["risk"], "high")

            second = client.post(
                "/api/agent/chat",
                json={
                    "session_id": first_body["session_id"],
                    "message": "confirmed",
                    "confirm_token": first_body["confirmation_token"],
                    "graph_context": {"model_graph": simple_graph()},
                },
            )
            second_body = second.json()
            diff = second_body["observations"][0]["output"]["diff"]

            self.assertEqual(second.status_code, 200)
            self.assertTrue(second_body["success"])
            self.assertFalse(second_body["requires_confirmation"])
            self.assertEqual(diff["operations"], [{"op": "remove_node", "id": "fc"}])
            self.assertIn("did not apply it directly", second_body["reply"])
        finally:
            agent.orchestrator = original
            tmp.cleanup()

    def test_agent_compresses_long_context_window(self):
        client, tmp, original = make_client(max_history_messages=4)
        try:
            session_id = None
            for index in range(6):
                payload = {"message": f"message {index}", "max_context_messages": 4}
                if session_id:
                    payload["session_id"] = session_id
                body = client.post("/api/agent/chat", json=payload).json()
                session_id = body["session_id"]

            detail = client.get(f"/api/agent/sessions/{session_id}").json()
            self.assertTrue(detail["summary"])
            self.assertLessEqual(len(detail["messages"]), 4)
            self.assertIn("message 0", detail["summary"])
        finally:
            agent.orchestrator = original
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
