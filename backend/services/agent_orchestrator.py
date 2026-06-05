from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from backend.schema.agent import (
    AgentChatRequest,
    AgentChatResponse,
    AgentGraphContext,
    AgentMessage,
    AgentObservation,
    AgentSessionDetail,
    AgentSessionInfo,
    AgentToolCall,
)
from backend.services.unified_code_gen import generate


DEFAULT_AGENT_DB = Path(__file__).resolve().parents[1] / "data" / "agent_conversations.sqlite3"


class SQLiteConversationStore:
    def __init__(self, db_path: str | Path | None = None):
        configured_path = db_path or os.getenv("FLOWHAMSTER_AGENT_DB") or DEFAULT_AGENT_DB
        self.db_path = Path(configured_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    pending_token TEXT,
                    pending_action TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    tool_name TEXT,
                    FOREIGN KEY(session_id) REFERENCES agent_sessions(id) ON DELETE CASCADE
                )
                """
            )

    def create_session(self, title: str) -> AgentSessionInfo:
        now = time.time()
        session_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO agent_sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
        return AgentSessionInfo(id=session_id, title=title, created_at=now, updated_at=now, message_count=0)

    def get_session(self, session_id: str) -> AgentSessionInfo | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT s.*, COUNT(m.id) AS message_count
                FROM agent_sessions s
                LEFT JOIN agent_messages m ON m.session_id = s.id
                WHERE s.id = ?
                GROUP BY s.id
                """,
                (session_id,),
            ).fetchone()
        return self._session_from_row(row) if row else None

    def list_sessions(self) -> list[AgentSessionInfo]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.*, COUNT(m.id) AS message_count
                FROM agent_sessions s
                LEFT JOIN agent_messages m ON m.session_id = s.id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                """
            ).fetchall()
        return [self._session_from_row(row) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM agent_sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    def add_message(self, session_id: str, role: str, content: str, tool_name: str | None = None) -> AgentMessage:
        now = time.time()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO agent_messages (session_id, role, content, created_at, tool_name) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, now, tool_name),
            )
            conn.execute("UPDATE agent_sessions SET updated_at = ? WHERE id = ?", (now, session_id))
            message_id = int(cursor.lastrowid)
        return AgentMessage(
            id=message_id,
            session_id=session_id,
            role=role,  # type: ignore[arg-type]
            content=content,
            created_at=now,
            tool_name=tool_name,
        )

    def get_messages(self, session_id: str) -> list[AgentMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, created_at, tool_name
                FROM agent_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            AgentMessage(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
                tool_name=row["tool_name"],
            )
            for row in rows
        ]

    def set_summary_and_keep_messages(self, session_id: str, summary: str, keep_message_ids: set[int]) -> None:
        now = time.time()
        with self._connect() as conn:
            if keep_message_ids:
                placeholders = ",".join("?" for _ in keep_message_ids)
                conn.execute(
                    f"DELETE FROM agent_messages WHERE session_id = ? AND id NOT IN ({placeholders})",
                    (session_id, *keep_message_ids),
                )
            else:
                conn.execute("DELETE FROM agent_messages WHERE session_id = ?", (session_id,))
            conn.execute(
                "UPDATE agent_sessions SET summary = ?, updated_at = ? WHERE id = ?",
                (summary, now, session_id),
            )

    def set_pending_action(self, session_id: str, token: str, action: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE agent_sessions SET pending_token = ?, pending_action = ? WHERE id = ?",
                (token, json.dumps(action), session_id),
            )

    def pop_pending_action(self, session_id: str, token: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT pending_token, pending_action FROM agent_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not row or row["pending_token"] != token or not row["pending_action"]:
                return None
            action = json.loads(row["pending_action"])
            conn.execute(
                "UPDATE agent_sessions SET pending_token = NULL, pending_action = NULL WHERE id = ?",
                (session_id,),
            )
        return action

    def get_detail(self, session_id: str) -> AgentSessionDetail | None:
        session = self.get_session(session_id)
        if session is None:
            return None
        return AgentSessionDetail(**session.model_dump(), messages=self.get_messages(session_id))

    def _session_from_row(self, row: sqlite3.Row) -> AgentSessionInfo:
        return AgentSessionInfo(
            id=row["id"],
            title=row["title"],
            summary=row["summary"] or "",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=int(row["message_count"]),
        )


class AgentOrchestrator:
    def __init__(
        self,
        store: SQLiteConversationStore | None = None,
        *,
        max_history_messages: int = 20,
        max_summary_chars: int = 1200,
    ):
        self.store = store or SQLiteConversationStore()
        self.max_history_messages = max_history_messages
        self.max_summary_chars = max_summary_chars

    def chat(self, req: AgentChatRequest) -> AgentChatResponse:
        session = self._get_or_create_session(req.session_id, req.message)
        self.store.add_message(session.id, "user", req.message)

        tool_calls: list[AgentToolCall] = []
        observations: list[AgentObservation] = []
        warnings: list[str] = []
        blocked = False
        requires_confirmation = False
        confirmation_token: str | None = None

        if req.confirm_token:
            pending = self.store.pop_pending_action(session.id, req.confirm_token)
            if pending is None:
                reply = "Confirmation token is invalid or expired. Please request the action again."
                warnings.append("invalid_confirmation_token")
            else:
                calls, obs, reply, blocked = self._execute_pending_action(pending, req.graph_context)
                tool_calls.extend(calls)
                observations.extend(obs)
        else:
            plan = self._plan(req.message, req.graph_context)
            if plan.get("requires_confirmation"):
                token = secrets.token_urlsafe(18)
                self.store.set_pending_action(session.id, token, plan)
                call = AgentToolCall(
                    id=self._tool_id("pending"),
                    name=plan["tool_name"],
                    args=plan.get("args", {}),
                    risk="high",
                    requires_confirmation=True,
                    confirmation_token=token,
                )
                tool_calls.append(call)
                requires_confirmation = True
                confirmation_token = token
                reply = "This is a high-risk action. Send the confirmation token back to receive a safe diff."
            else:
                calls, obs, reply, blocked = self._execute_plan(plan, req.graph_context)
                tool_calls.extend(calls)
                observations.extend(obs)

        self.store.add_message(session.id, "assistant", reply)
        for observation in observations:
            self.store.add_message(
                session.id,
                "tool",
                json.dumps(observation.model_dump(), ensure_ascii=False),
                tool_name=observation.tool_name,
            )

        summary = self._compress_if_needed(session.id, req.max_context_messages or self.max_history_messages)
        detail = self.store.get_session(session.id)
        if detail:
            summary = detail.summary

        return AgentChatResponse(
            success=not blocked,
            session_id=session.id,
            reply=reply,
            mode=req.mode,
            tool_calls=tool_calls,
            observations=observations,
            requires_confirmation=requires_confirmation,
            confirmation_token=confirmation_token,
            blocked=blocked,
            warnings=warnings,
            summary=summary,
        )

    def list_sessions(self) -> list[AgentSessionInfo]:
        return self.store.list_sessions()

    def get_session(self, session_id: str) -> AgentSessionDetail | None:
        return self.store.get_detail(session_id)

    def delete_session(self, session_id: str) -> bool:
        return self.store.delete_session(session_id)

    def _get_or_create_session(self, session_id: str | None, message: str) -> AgentSessionInfo:
        if session_id:
            session = self.store.get_session(session_id)
            if session:
                return session
        return self.store.create_session(self._make_title(message))

    def _make_title(self, message: str) -> str:
        compact = " ".join(message.strip().split())
        if not compact:
            return "Untitled agent session"
        return compact[:48]

    def _plan(self, message: str, graph_context: AgentGraphContext | None) -> dict[str, Any]:
        text = message.lower()
        delete_match = re.search(r"(?:delete|remove)\s+node\s+([A-Za-z0-9_.:-]+)", message, re.IGNORECASE)
        if delete_match:
            return {
                "tool_name": "graph.mutation",
                "requires_confirmation": True,
                "args": {"operations": [{"op": "remove_node", "id": delete_match.group(1)}]},
            }

        if any(keyword in text for keyword in ("execute", "run code", "train now")):
            return {
                "tool_name": "execute.request",
                "requires_confirmation": True,
                "args": {"reason": "code execution can run user-provided workloads"},
            }

        connect_match = re.search(
            r"(?:connect|add edge)\s+([A-Za-z0-9_.:-]+)\s+(?:to|->)\s+([A-Za-z0-9_.:-]+)",
            message,
            re.IGNORECASE,
        )
        if connect_match:
            return {
                "tool_name": "graph.mutation",
                "args": {
                    "operations": [
                        {
                            "op": "add_edge",
                            "id": f"agent_edge_{connect_match.group(1)}_{connect_match.group(2)}",
                            "source": connect_match.group(1),
                            "target": connect_match.group(2),
                        }
                    ]
                },
            }

        needs_validate = "validate" in text or "check graph" in text
        needs_generate = "generate" in text and "code" in text
        if needs_validate or needs_generate:
            return {
                "tool_name": "tool_chain",
                "args": {"validate_graph": needs_validate or needs_generate, "generate_code": needs_generate},
            }

        if graph_context and graph_context.model_graph:
            return {"tool_name": "graph.summary", "args": {}}

        return {"tool_name": "chat", "args": {}}

    def _execute_pending_action(
        self,
        plan: dict[str, Any],
        graph_context: AgentGraphContext | None,
    ) -> tuple[list[AgentToolCall], list[AgentObservation], str, bool]:
        if plan["tool_name"] == "execute.request":
            call = AgentToolCall(id=self._tool_id("execute"), name="execute.request", args=plan["args"], risk="high")
            observation = AgentObservation(
                tool_call_id=call.id,
                tool_name=call.name,
                success=True,
                output={"status": "confirmed", "next_endpoint": "/api/execute"},
            )
            reply = "Execution is confirmed. Use /api/execute with the graph payload so existing execution safeguards apply."
            return [call], [observation], reply, False
        return self._execute_plan(plan, graph_context)

    def _execute_plan(
        self,
        plan: dict[str, Any],
        graph_context: AgentGraphContext | None,
    ) -> tuple[list[AgentToolCall], list[AgentObservation], str, bool]:
        tool_name = plan["tool_name"]
        if tool_name == "graph.mutation":
            return self._run_graph_mutation(plan.get("args", {}), graph_context)
        if tool_name == "tool_chain":
            return self._run_tool_chain(plan.get("args", {}), graph_context)
        if tool_name == "graph.summary":
            graph = (graph_context or AgentGraphContext()).model_graph or {}
            compact = compact_graph_context(graph)
            reply = f"Graph has {len(compact['nodes'])} nodes and {len(compact['edges'])} edges."
            return [], [], reply, False
        return [], [], "I can help validate the graph, generate code, or prepare safe graph mutation diffs.", False

    def _run_tool_chain(
        self,
        args: dict[str, Any],
        graph_context: AgentGraphContext | None,
    ) -> tuple[list[AgentToolCall], list[AgentObservation], str, bool]:
        graph = (graph_context or AgentGraphContext()).model_graph or {"nodes": [], "edges": []}
        calls: list[AgentToolCall] = []
        observations: list[AgentObservation] = []

        if args.get("validate_graph"):
            call = AgentToolCall(id=self._tool_id("validate"), name="graph.validate", args={}, risk="low")
            result = validate_graph(graph)
            calls.append(call)
            observations.append(
                AgentObservation(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    success=not result["errors"],
                    output=result,
                    error="; ".join(result["errors"]) if result["errors"] else None,
                )
            )
            if result["errors"]:
                reply = "Graph validation failed. Fix the reported structure issues before generating code."
                return calls, observations, reply, True

        if args.get("generate_code"):
            call = AgentToolCall(id=self._tool_id("generate"), name="code.generate", args={}, risk="low")
            calls.append(call)
            try:
                options: dict[str, Any] = {}
                if graph_context and graph_context.training_config:
                    options["training_config"] = graph_context.training_config
                code = generate(graph, options=options)
                observations.append(
                    AgentObservation(
                        tool_call_id=call.id,
                        tool_name=call.name,
                        success=True,
                        output={"code": code, "length": len(code)},
                    )
                )
            except Exception as exc:
                observations.append(
                    AgentObservation(tool_call_id=call.id, tool_name=call.name, success=False, error=str(exc))
                )
                return calls, observations, "Code generation failed; see the tool observation error.", True

        if args.get("generate_code"):
            return calls, observations, "Graph validated and code generation completed.", False
        return calls, observations, "Graph validation completed.", False

    def _run_graph_mutation(
        self,
        args: dict[str, Any],
        graph_context: AgentGraphContext | None,
    ) -> tuple[list[AgentToolCall], list[AgentObservation], str, bool]:
        graph = (graph_context or AgentGraphContext()).model_graph or {"nodes": [], "edges": []}
        operations = args.get("operations", [])
        call = AgentToolCall(id=self._tool_id("mutation"), name="graph.mutation", args=args, risk="medium")
        validation = validate_mutation_diff(graph, operations)
        observation = AgentObservation(
            tool_call_id=call.id,
            tool_name=call.name,
            success=not validation["errors"],
            output=validation,
            error="; ".join(validation["errors"]) if validation["errors"] else None,
        )
        if validation["errors"]:
            reply = "Graph mutation was blocked because it would make the graph invalid."
            return [call], [observation], reply, True
        reply = "Prepared a safe graph mutation diff. The agent did not apply it directly."
        return [call], [observation], reply, False

    def _compress_if_needed(self, session_id: str, max_messages: int) -> str:
        messages = self.store.get_messages(session_id)
        if len(messages) <= max_messages:
            session = self.store.get_session(session_id)
            return session.summary if session else ""

        keep_count = max(4, max_messages // 2)
        keep = messages[-keep_count:]
        older = messages[:-keep_count]
        session = self.store.get_session(session_id)
        previous = session.summary if session else ""
        lines = [previous] if previous else []
        for message in older:
            if message.role == "tool":
                continue
            content = " ".join(message.content.split())
            lines.append(f"{message.role}: {content[:180]}")
        summary = "\n".join(line for line in lines if line).strip()
        if len(summary) > self.max_summary_chars:
            summary = summary[-self.max_summary_chars :]
        self.store.set_summary_and_keep_messages(
            session_id,
            summary,
            {message.id for message in keep if message.id is not None},
        )
        return summary

    def _tool_id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:10]}"


def compact_graph_context(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    edges = graph.get("edges", []) if isinstance(graph, dict) else []
    compact_nodes = []
    for node in nodes if isinstance(nodes, list) else []:
        if not isinstance(node, dict):
            continue
        data = node.get("data", {}) if isinstance(node.get("data", {}), dict) else {}
        compact_nodes.append(
            {
                "id": node.get("id"),
                "type": data.get("nodeType") or node.get("type"),
                "label": data.get("label"),
            }
        )
    compact_edges = []
    for edge in edges if isinstance(edges, list) else []:
        if not isinstance(edge, dict):
            continue
        compact_edges.append(
            {
                "id": edge.get("id"),
                "source": edge.get("source"),
                "target": edge.get("target"),
            }
        )
    return {"nodes": compact_nodes, "edges": compact_edges}


def validate_graph(graph: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(graph, dict):
        return {"errors": ["graph must be an object"], "warnings": [], "compact": {"nodes": [], "edges": []}}

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not isinstance(nodes, list):
        errors.append("graph.nodes must be a list")
        nodes = []
    if not isinstance(edges, list):
        errors.append("graph.edges must be a list")
        edges = []

    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"node[{index}] must be an object")
            continue
        node_id = node.get("id")
        if not node_id:
            errors.append(f"node[{index}] is missing id")
        elif node_id in node_ids:
            errors.append(f"duplicate node id '{node_id}'")
        else:
            node_ids.add(str(node_id))
        data = node.get("data", {})
        if not isinstance(data, dict) or not data.get("nodeType"):
            warnings.append(f"node '{node_id}' is missing data.nodeType")

    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"edge[{index}] must be an object")
            continue
        source = edge.get("source")
        target = edge.get("target")
        if source not in node_ids:
            errors.append(f"edge[{index}] source '{source}' does not exist")
        if target not in node_ids:
            errors.append(f"edge[{index}] target '{target}' does not exist")
        if source == target:
            errors.append(f"edge[{index}] cannot connect a node to itself")

    return {"errors": errors, "warnings": warnings, "compact": compact_graph_context(graph)}


def validate_mutation_diff(graph: dict[str, Any], operations: list[dict[str, Any]]) -> dict[str, Any]:
    base = validate_graph(graph)
    errors = list(base["errors"])
    warnings = list(base["warnings"])
    nodes = graph.get("nodes", []) if isinstance(graph, dict) and isinstance(graph.get("nodes", []), list) else []
    edges = graph.get("edges", []) if isinstance(graph, dict) and isinstance(graph.get("edges", []), list) else []
    node_ids = {str(node.get("id")) for node in nodes if isinstance(node, dict) and node.get("id")}
    edge_ids = {str(edge.get("id")) for edge in edges if isinstance(edge, dict) and edge.get("id")}

    if not isinstance(operations, list) or not operations:
        errors.append("mutation diff must include at least one operation")
        operations = []

    safe_operations: list[dict[str, Any]] = []
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            errors.append(f"operation[{index}] must be an object")
            continue
        op = operation.get("op")
        if op == "remove_node":
            node_id = operation.get("id")
            if node_id not in node_ids:
                errors.append(f"cannot remove missing node '{node_id}'")
            else:
                safe_operations.append({"op": "remove_node", "id": node_id})
        elif op == "add_edge":
            source = operation.get("source")
            target = operation.get("target")
            edge_id = operation.get("id") or f"edge_{source}_{target}"
            op_errors: list[str] = []
            if source not in node_ids:
                op_errors.append(f"cannot add edge from missing source '{source}'")
            if target not in node_ids:
                op_errors.append(f"cannot add edge to missing target '{target}'")
            if source == target:
                op_errors.append("cannot add self-loop edge")
            if edge_id in edge_ids:
                warnings.append(f"edge id '{edge_id}' already exists")
            errors.extend(op_errors)
            if not op_errors:
                safe_operations.append({"op": "add_edge", "id": edge_id, "source": source, "target": target})
        elif op == "add_node":
            node = operation.get("node")
            if not isinstance(node, dict) or not node.get("id"):
                errors.append("add_node operation requires node.id")
            elif node["id"] in node_ids:
                errors.append(f"cannot add duplicate node '{node['id']}'")
            elif not isinstance(node.get("data", {}), dict) or not node["data"].get("nodeType"):
                errors.append("add_node operation requires node.data.nodeType")
            else:
                safe_operations.append({"op": "add_node", "node": node})
        else:
            errors.append(f"unsupported mutation operation '{op}'")

    return {
        "errors": errors,
        "warnings": warnings,
        "diff": {"operations": [] if errors else safe_operations},
        "corrections": build_correction_suggestions(errors),
    }


def build_correction_suggestions(errors: list[str]) -> list[str]:
    suggestions: list[str] = []
    for error in errors:
        if "missing source" in error or "missing target" in error or "does not exist" in error:
            suggestions.append("Use an existing node id from the compact graph context before adding the edge.")
        elif "self-loop" in error:
            suggestions.append("Choose distinct source and target nodes.")
        elif "duplicate node" in error:
            suggestions.append("Pick a new node id before proposing add_node.")
    if errors and not suggestions:
        suggestions.append("Review the graph schema and submit a corrected diff.")
    return suggestions
