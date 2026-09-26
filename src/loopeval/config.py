from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "disabled",
        "typesafe",
        "openrouter_decisions",
        "openrouter",
        "openai",
        "openai_compatible",
        "mock",
    ]
    model: str = ""
    api_key_env: str | None = None
    base_url: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=8)
    input_cost_per_million: float | None = Field(default=None, ge=0)
    output_cost_per_million: float | None = Field(default=None, ge=0)
    structured_output: bool = True
    headers: dict[str, str] = Field(default_factory=dict)
    mock_responses: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_provider(self) -> ProviderConfig:
        if self.type not in {"disabled", "mock"}:
            if not self.model:
                raise ValueError(f"{self.type} requires model")
            if not self.api_key_env:
                raise ValueError(f"{self.type} requires api_key_env")
        if self.type == "openai_compatible" and not self.base_url:
            raise ValueError("openai_compatible requires base_url")
        return self

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


class ProvidersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: ProviderConfig
    fallback: ProviderConfig = Field(default_factory=lambda: ProviderConfig(type="disabled"))

    @model_validator(mode="after")
    def validate_roles(self) -> ProvidersConfig:
        if self.decision.type not in {"typesafe", "openrouter_decisions", "mock"}:
            raise ValueError(f"{self.decision.type} cannot be used as a decision provider")
        if self.fallback.type not in {
            "disabled",
            "openrouter",
            "openai",
            "openai_compatible",
            "mock",
        }:
            raise ValueError(f"{self.fallback.type} cannot be used as a fallback provider")
        return self


class EscalationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    on_uncertain: bool = True
    on_decision_error: bool = True
    on_no_applicable_checks: bool = True
    short_circuit_on_deterministic_failure: bool = True
    coverage_check: bool = True
    coverage_pass_threshold: float = Field(default=0.2, ge=0, le=1)
    coverage_failure_threshold: float = Field(default=0.65, ge=0, le=1)
    random_audit_rate: float = Field(default=0.02, ge=0, le=1)
    fail_closed_without_fallback: bool = False
    max_fallbacks_per_run: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> EscalationPolicy:
        if self.coverage_pass_threshold >= self.coverage_failure_threshold:
            raise ValueError("coverage_pass_threshold must be below coverage_failure_threshold")
        return self


class BudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_sample_seconds: float = Field(default=60.0, gt=0)
    run_cost_usd: float | None = Field(default=None, gt=0)
    concurrency: int = Field(default=8, ge=1, le=128)
    max_state_chars: int = Field(default=30_000, ge=1_000)


class PromotionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_examples: int = Field(default=20, ge=1)
    minimum_precision: float = Field(default=0.9, ge=0, le=1)
    minimum_recall: float = Field(default=0.5, ge=0, le=1)
    minimum_coverage: float = Field(default=0.8, ge=0, le=1)
    require_human_approval: bool = True


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: str = ".loopeval"
    database: str = "loopeval.db"
    cache: bool = True


class LoopEvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    project: str = "loopeval-project"
    checks: list[str] = Field(default_factory=lambda: ["checks"])
    providers: ProvidersConfig
    escalation: EscalationPolicy = Field(default_factory=EscalationPolicy)
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    promotion: PromotionPolicy = Field(default_factory=PromotionPolicy)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_config(path: str | Path = "loopeval.yaml") -> LoopEvalConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text())
    config = LoopEvalConfig.model_validate(data)
    base = config_path.resolve().parent
    config.checks = [
        str((base / p).resolve()) if not Path(p).is_absolute() else p for p in config.checks
    ]
    storage_dir = Path(config.storage.directory)
    if not storage_dir.is_absolute():
        config.storage.directory = str((base / storage_dir).resolve())
    return config
