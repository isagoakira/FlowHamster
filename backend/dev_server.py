"""
Backend development launcher with scoped reload watching.

This avoids watching the whole repository tree (especially node_modules)
when starting the FastAPI backend from the project root.
"""
from __future__ import annotations

from pathlib import Path

import uvicorn


def main() -> None:
    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parent

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(backend_dir)],
        app_dir=str(project_root),
    )


if __name__ == "__main__":
    main()
