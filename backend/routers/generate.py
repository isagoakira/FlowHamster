"""
POST /api/generate — 接收前端图结构，返回生成的 Python 代码
使用新的 AST 代码生成器（ast_core.py v2）
支持 data_graph 和 bindings 参数
"""
from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.unified_code_gen import generate
from backend.services.dataflow_compiler import compile_dataflow

router = APIRouter()


class GenerateRequest(BaseModel):
    graph: dict
    options: dict | None = None
    training_config: dict | None = None
    data_graph: dict | None = None
    bindings: list[dict] | None = None


class GenerateResponse(BaseModel):
    success: bool
    code: str
    warnings: list[str] = []


def _build_workflow_scaffold(data_graph: dict | None, bindings: list[dict] | None, training_config: dict | None) -> str:
    """构建数据流相关的 Python scaffold"""
    if not data_graph and not bindings:
        return ""

    compiled = compile_dataflow(
        model_graph=None,
        data_graph=data_graph,
        bindings=bindings,
        training_config=training_config,
    )

    if not compiled.has_workflow_runtime:
        return ""

    return f"\n\n{compiled.python_scaffold}\n"


@router.post("/generate", response_model=GenerateResponse)
async def generate_code(req: GenerateRequest):
    try:
        options = dict(req.options or {})
        if req.training_config is not None:
            options["training_config"] = req.training_config
        code = generate(req.graph, options=options)

        # 如果有 data_graph 或 bindings，追加数据流 scaffold
        workflow_scaffold = _build_workflow_scaffold(
            req.data_graph,
            req.bindings,
            req.training_config
        )
        if workflow_scaffold:
            code += workflow_scaffold

        return GenerateResponse(success=True, code=code, warnings=[])
    except Exception as e:
        return GenerateResponse(success=False, code=f"# Error: {e}", warnings=[str(e)])
