from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


TaskStatus = Literal["running", "completed", "failed"]


class ShoppingQuery(BaseModel):
    query: str = Field(..., min_length=1)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query cannot be empty")
        return stripped


class Product(BaseModel):
    id: str
    platform: str
    title: str
    price: float = Field(..., ge=0)
    currency: str = "CNY"
    shipping: float = Field(default=0, ge=0)
    tax: float = Field(default=0, ge=0)
    rating: float = Field(default=0, ge=0, le=5)
    reason: str
    url: str

    @property
    def total_price(self) -> float:
        return round(self.price + self.shipping + self.tax, 2)


class ShoppingSummary(BaseModel):
    message: str
    products: list[Product]
    warnings: list[str] = Field(default_factory=list)


class AgentEvent(BaseModel):
    type: str
    thread_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: str
    tool: str | None = None
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)


class TaskState(BaseModel):
    thread_id: str
    status: TaskStatus = "running"
    query: str | None = None
    result: ShoppingSummary | None = None
    error: str | None = None
    events: list[AgentEvent] = Field(default_factory=list)
