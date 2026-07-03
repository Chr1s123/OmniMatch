from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import OmniMatchSettings
from app.providers.registry import ProviderRegistry


@dataclass
class ToolContext:
    settings: OmniMatchSettings
    providers: ProviderRegistry
    observations: list[dict[str, Any]] = field(default_factory=list)
