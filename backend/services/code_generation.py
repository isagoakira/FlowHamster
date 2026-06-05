"""Facade for user-visible FlowHamster code generation entry points."""
from __future__ import annotations

from dataclasses import dataclass

from backend.services.dataflow_compiler import compile_dataflow
from backend.services.unified_code_gen import GeneratedCodeSections, generate_sections


@dataclass
class GeneratedCode:
    code: str
    warnings: list[str]
    sections: GeneratedCodeSections


def build_generation_options(
    *,
    graph: dict,
    options: dict | None = None,
    training_config: dict | None = None,
    data_graph: dict | None = None,
    bindings: list[dict] | None = None,
) -> tuple[dict, list[str]]:
    """Build unified generator options shared by HTTP, websocket, export, and execute."""
    merged_options = dict(options or {})
    warnings: list[str] = []

    if training_config is not None:
        merged_options["training_config"] = training_config

    if data_graph or bindings:
        compiled = compile_dataflow(
            model_graph=graph,
            data_graph=data_graph,
            bindings=bindings,
            training_config=training_config,
        )
        if compiled.has_workflow_runtime:
            merged_options["workflow_scaffold"] = compiled.python_scaffold
            merged_options["workflow_runtime"] = compiled.has_workflow_runtime
        warnings.extend(compiled.warnings)

    return merged_options, warnings


def generate_python(
    *,
    graph: dict,
    options: dict | None = None,
    training_config: dict | None = None,
    data_graph: dict | None = None,
    bindings: list[dict] | None = None,
) -> GeneratedCode:
    """Generate Python through the single unified backend generator."""
    merged_options, warnings = build_generation_options(
        graph=graph,
        options=options,
        training_config=training_config,
        data_graph=data_graph,
        bindings=bindings,
    )
    sections = generate_sections(graph, options=merged_options)
    return GeneratedCode(
        code=sections.render(include_main=True),
        warnings=warnings,
        sections=sections,
    )
