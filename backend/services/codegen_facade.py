"""Public code generation facade for user-visible backend entry points."""
from __future__ import annotations

from dataclasses import dataclass

from backend.services.dataflow_compiler import CompiledDataflow, compile_dataflow
from backend.services.unified_code_gen import UnifiedCodeGenerator


@dataclass(frozen=True)
class GeneratedCode:
    code: str
    warnings: list[str]


def build_options(
    options: dict | None = None,
    training_config: dict | None = None,
) -> dict:
    merged = dict(options or {})
    if training_config is not None:
        merged["training_config"] = training_config
    return merged


def compile_workflow_scaffold(
    model_graph: dict | None = None,
    data_graph: dict | None = None,
    bindings: list[dict] | None = None,
    training_config: dict | None = None,
) -> CompiledDataflow | None:
    if not data_graph and not bindings:
        return None

    compiled = compile_dataflow(
        model_graph=model_graph,
        data_graph=data_graph,
        bindings=bindings,
        training_config=training_config,
    )
    if not compiled.has_workflow_runtime:
        return None
    return compiled


def generate_model_code(graph: dict, options: dict | None = None) -> str:
    return UnifiedCodeGenerator(graph, options).generate_model()


def generate_full_code(
    graph: dict,
    options: dict | None = None,
    training_config: dict | None = None,
    data_graph: dict | None = None,
    bindings: list[dict] | None = None,
) -> GeneratedCode:
    merged_options = build_options(options, training_config)
    warnings: list[str] = []
    compiled = compile_workflow_scaffold(
        model_graph=graph,
        data_graph=data_graph,
        bindings=bindings,
        training_config=training_config,
    )
    if compiled:
        merged_options["workflow_scaffold"] = compiled.python_scaffold
        merged_options["workflow_runtime"] = compiled.has_workflow_runtime
        warnings.extend(compiled.warnings)

    code = UnifiedCodeGenerator(graph, merged_options).generate_full()
    return GeneratedCode(code=code, warnings=warnings)


def generate_notebook_code(
    graph: dict,
    training_config: dict | None = None,
    data_graph: dict | None = None,
    bindings: list[dict] | None = None,
) -> GeneratedCode:
    return generate_full_code(
        graph=graph,
        training_config=training_config,
        data_graph=data_graph,
        bindings=bindings,
    )


def generate(graph: dict, options: dict | None = None) -> str:
    """Backward-compatible string API for callers that only need full code."""
    return generate_full_code(graph, options=options).code
