"""Explicit configuration and observations for colocated turn-boundary rollouts."""
from typing import Callable, Any
from pydantic import BaseModel, ConfigDict, Field


def ignore_event(*args, **kwargs):
    pass


def accept_group(group):
    return bool(group)


class TurnAsyncOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    # None means unbounded; zero is an explicit engine-only diagnostic budget.
    max_groups: int | None = Field(default=None, ge=0)
    pool_groups: int = Field(default=0, ge=0)
    budget_seconds: float = Field(default=0, ge=0)
    started_at: float | None = None
    drain_seconds: float = Field(default=300, ge=0)
    max_staleness: int = Field(default=2, ge=0)
    trace_soft_bytes: int = Field(default=96 * 1024**3, gt=0)
    trace_metrics: bool = False
    require_routing: bool = False
    group_filter: Callable = accept_group
    # Task policy may return a replacement-admission delay, e.g. quota backoff.
    result_backoff: Callable | None = None
    event: Callable = ignore_event
    disposition: Callable = ignore_event
    # Explicit testing extension; production defaults never hold or reorder.
    batch_wait: Callable | None = None
    batch_order: Callable | None = None


class TurnTrialOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    run_id: str = ""
    require_routing: bool = False
    event: Callable = ignore_event
    disposition: Callable = ignore_event
    # Task adapters annotate identity / normalize terminal artifacts only.
    identity_adapter: Callable | None = None
    result_adapter: Callable | None = None
    before_trial: Callable | None = None


class TurnTrainingOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    event: Callable = ignore_event
    disposition: Callable = ignore_event
    prepared_metrics: Callable | None = None
    before_train: Callable | None = None
    after_train: Callable | None = None
