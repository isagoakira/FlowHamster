"""Shared code generation entry points.

All backend surfaces that need generated model code should import from here
instead of calling legacy generators directly.
"""
from __future__ import annotations

from .unified_code_gen import UnifiedCodeGenerator, build_ast, generate


def generate_full(flow_json: dict, options: dict | None = None) -> str:
    return generate(flow_json, options=options)


def generate_model(flow_json: dict, options: dict | None = None) -> str:
    return UnifiedCodeGenerator(flow_json, options).generate_model()
