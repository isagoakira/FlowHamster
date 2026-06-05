"""
POST /api/generate — 接收前端图结构，返回生成的 Python 代码

使用新的 AST 代码生成器（ast_core.py v2）
支持 data_graph 和 bindings 参数

Request body:
    graph: dict       — 前端 flow_json（nodes + edges）
    options: dict      — 可选，代码生成选项
    training_config: dict — 可选，训练配置
    data_graph: dict   — 可选，数据流图
    bindings: list[dict] — 可选，输入输出绑定

Response:
    success: bool    — 是否成功
    code: str         — 生成的 Python 代码
    warnings: list[str] — 警告信息
"""
from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.codegen_facade import generate_full_code

router = APIRouter()


class GenerateRequest(BaseModel):
    """
    代码生成请求

    Attributes:
        graph: 前端 flow_json，包含 nodes 和 edges
        options: 代码生成的可选配置
        training_config: 训练配置（optimizer, scheduler, epochs 等）
        data_graph: 数据流图定义（可选）
        bindings: 模型输入输出与数据源的绑定（可选）
    """
    graph: dict
    options: dict | None = None
    training_config: dict | None = None
    data_graph: dict | None = None
    bindings: list[dict] | None = None


class GenerateResponse(BaseModel):
    """
    代码生成响应

    Attributes:
        success: 是否成功生成代码
        code: 生成的 Python 代码字符串
        warnings: 任何警告或提示信息
    """
    success: bool
    code: str
    warnings: list[str] = []


@router.post("/generate", response_model=GenerateResponse)
async def generate_code(req: GenerateRequest):
    """
    生成 PyTorch 模型代码。

    接收前端 flow JSON，使用 AST 代码生成器输出完整的 PyTorch 模型定义。
    如果提供了 data_graph 或 bindings，还会追加数据流相关的 scaffold 代码。

    Returns:
        GenerateResponse: 包含生成的代码和任何警告
    """
    try:
        generated = generate_full_code(
            req.graph,
            options=req.options,
            training_config=req.training_config,
            data_graph=req.data_graph,
            bindings=req.bindings,
        )
        return GenerateResponse(success=True, code=generated.code, warnings=generated.warnings)
    except Exception as e:
        return GenerateResponse(success=False, code=f"# Error: {e}", warnings=[str(e)])
