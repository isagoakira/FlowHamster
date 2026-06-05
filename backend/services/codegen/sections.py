"""Small primitives for composing generated code sections."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CodeSection:
    name: str
    lines: list[str] = field(default_factory=list)
    header: str | None = None

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def extend(self, lines: list[str] | tuple[str, ...]) -> None:
        self.lines.extend(lines)

    def render(self) -> str:
        rendered = list(self.lines)
        if self.header:
            rendered.insert(0, self.header)
        return "\n".join(rendered).rstrip()


class CodeWriter:
    def __init__(self) -> None:
        self.sections: list[CodeSection] = []

    def section(self, name: str, lines: list[str] | None = None, header: str | None = None) -> CodeSection:
        section = CodeSection(name=name, lines=list(lines or []), header=header)
        self.sections.append(section)
        return section

    def render(self) -> str:
        parts = []
        for section in self.sections:
            rendered = section.render()
            if rendered:
                parts.append(rendered)
        return "\n\n".join(parts).rstrip() + "\n"
