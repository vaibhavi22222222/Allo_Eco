"""Typed boundary for Vibhu's optimizer. No API/database/LLM dependencies."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class WeeklyValue(Contract):
    plant: str
    week_start: date
    value: float = Field(ge=0)

    @model_validator(mode="after")
    def monday(self):
        if self.week_start.weekday() != 0:
            raise ValueError("week_start must be a Monday")
        return self


class CapacityEvent(Contract):
    event_id: str = Field(min_length=1)
    plant: str = Field(min_length=1)
    from_date: date
    to_date: date
    capacity_loss_pct: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def dates(self):
        if self.to_date < self.from_date:
            raise ValueError("to_date precedes from_date")
        return self


class PlanningConfig(Contract):
    planning_start: date
    horizon_days: int = Field(default=28, ge=1, le=366)
    currency: str = Field(default="INR", min_length=1)
    production_weekdays: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    energy_price_per_kwh: dict[str, float]
    plant_weekly_capacity: dict[str, int] = Field(default_factory=dict)
    weekly_carbon_overrides: list[WeeklyValue] = Field(default_factory=list)
    weekly_energy_price_overrides: list[WeeklyValue] = Field(default_factory=list)
    percentage_convention: Literal["fraction_0_1"] = "fraction_0_1"
    duty_basis: Literal["production_plus_energy"] = "production_plus_energy"
    overlap_policy: Literal["maximum_active_loss"] = "maximum_active_loss"
    cost_scale: float | None = Field(default=None, gt=0)
    carbon_scale: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def semantics(self):
        if not self.production_weekdays or len(set(self.production_weekdays)) != len(
            self.production_weekdays
        ):
            raise ValueError("production_weekdays must be a nonempty unique list")
        if any(day not in range(7) for day in self.production_weekdays):
            raise ValueError("production_weekdays must be 0..6")
        import math

        if any(
            not math.isfinite(v) or v < 0 for v in self.energy_price_per_kwh.values()
        ):
            raise ValueError("energy prices must be finite and nonnegative")
        if any(v <= 0 for v in self.plant_weekly_capacity.values()):
            raise ValueError("shared plant capacities must be positive")
        for overrides in (
            self.weekly_carbon_overrides,
            self.weekly_energy_price_overrides,
        ):
            keys = [(v.plant, v.week_start) for v in overrides]
            if len(keys) != len(set(keys)):
                raise ValueError("duplicate plant/week override")
        return self


class ObjectiveWeights(Contract):
    cost: float = Field(default=0.7, ge=0)
    carbon: float = Field(default=0.3, ge=0)
    lateness: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def not_zero(self):
        if self.cost + self.carbon + self.lateness <= 0:
            raise ValueError("at least one weight must be positive")
        return self

    def normalized(self) -> dict[str, float]:
        total = self.cost + self.carbon + self.lateness
        return {k: v / total for k, v in self.model_dump().items()}


class SolvePolicy(Contract):
    time_limit_seconds: float = Field(default=15, gt=0, le=600)
    total_time_limit_seconds: float = Field(default=60, gt=0, le=1800)
    allow_diagnostic_alternative: bool = True
    stability_tiebreak: bool = True
    integrality_tolerance: float = Field(default=1e-5, gt=0, le=1e-3)
    objective_tolerance: float = Field(default=1e-7, gt=0, le=1e-3)
    carbon_budget_kg: float | None = Field(default=None, ge=0)
    forced_assignments: dict[str, str] = Field(default_factory=dict)


class SolveResult(Contract):
    schema_version: str = "vibhu-optimizer-1"
    dataset_version_id: str
    data_hash: str
    config_version: str
    scenario_hash: str
    planning_window: dict[str, str]
    solver: dict[str, Any]
    answer_type: Literal["allocation", "infeasible", "scenario", "brief"] = "allocation"
    alternative: bool = False
    original_request_met: bool = False
    allocation: list[dict[str, Any]] = Field(default_factory=list)
    production_schedule: list[dict[str, Any]] = Field(default_factory=list)
    unassigned_orders: list[dict[str, Any]] = Field(default_factory=list)
    kpis: dict[str, Any] | None = None
    weights_used: dict[str, float]
    normalization_scales: dict[str, float]
    validator: dict[str, Any] = Field(default_factory=lambda: {"applicable": False})
    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    problem_context: dict[str, Any] = Field(default_factory=dict)
