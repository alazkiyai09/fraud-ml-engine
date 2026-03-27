"""Minimal shared model contract."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelDescriptor:
    name: str
    family: str
    supports_explanations: bool = True
    notes: list[str] = field(default_factory=list)
