"""
FlowHamster 模板 API 路由
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from backend.templates.registry import (
    load_all_templates,
    load_template,
    get_template_code,
    PRESET_TEMPLATES,
)

router = APIRouter()


@router.get("/templates")
async def get_templates() -> List[Dict[str, Any]]:
    """
    获取所有可用模板

    Returns:
        List of template metadata (without full graph)
    """
    templates = load_all_templates()

    # 如果模板没有元数据，使用预设
    result = []
    for t in templates:
        template_id = t["id"]
        preset = PRESET_TEMPLATES.get(template_id, {})
        result.append({
            "id": template_id,
            "name": preset.get("name", t.get("name", template_id)),
            "description": preset.get("description", t.get("description", "")),
            "category": preset.get("category", t.get("category", "generic")),
            "emoji": preset.get("emoji", t.get("emoji", "📦")),
            "version": t.get("version", "1.0.0"),
        })

    return result


@router.get("/templates/{template_id}")
async def get_template(template_id: str) -> Dict[str, Any]:
    """
    获取指定模板的完整信息

    Args:
        template_id: 模板 ID

    Returns:
        Full template with graph definition
    """
    template = load_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found")

    # 获取 Python 源代码
    code = get_template_code(template_id)
    if code:
        template["backend_code"] = {"python": code}

    return template


@router.get("/templates/{template_id}/code")
async def get_template_code_route(template_id: str) -> Dict[str, str]:
    """
    仅获取模板的 Python 代码

    Args:
        template_id: 模板 ID

    Returns:
        Python source code
    """
    code = get_template_code(template_id)
    if not code:
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found")

    return {"template_id": template_id, "code": code}


@router.get("/templates/{template_id}/graph")
async def get_template_graph(template_id: str) -> Dict[str, Any]:
    """
    仅获取模板的图结构

    Args:
        template_id: 模板 ID

    Returns:
        Graph definition (nodes and edges)
    """
    template = load_template(template_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template {template_id} not found")

    return {"id": template_id, "graph": template.get("graph", {"nodes": [], "edges": []})}
