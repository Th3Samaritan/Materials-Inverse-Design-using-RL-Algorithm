from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Equipment(StrictModel):
    max_temperature_c: Annotated[FiniteFloat, Field(ge=750, le=1100)] = 1100
    max_hold_hours: Annotated[FiniteFloat, Field(ge=1, le=12)] = 12
    allowed_quenches: list[Literal[0, 1, 2, 3]] = Field(default_factory=lambda: [0, 1, 2, 3], min_length=1, max_length=4)
    section_diameter_mm: Literal[25] = 25


class DesignRequest(StrictModel):
    composition: dict[str, FiniteFloat]
    targets: dict[str, FiniteFloat]
    constraints: Equipment = Field(default_factory=Equipment)
    n_candidates: int = Field(default=300, ge=10, le=1000, strict=True)
    top_k: int = Field(default=4, ge=1, le=10, strict=True)
    seed: int = Field(default=42, ge=0, le=2147483647, strict=True)
    method: Literal['policy', 'random', 'hybrid'] = 'hybrid'
    tolerance_fraction: Annotated[FiniteFloat, Field(gt=0, le=.5)] = .15
    time_weight: Annotated[FiniteFloat, Field(ge=0, le=2)] = 0
    save: bool = False


class Process(StrictModel):
    T_aus: Annotated[FiniteFloat, Field(ge=750, le=1100)] = 870
    t_aus: Annotated[FiniteFloat, Field(ge=.5, le=4)] = 1
    quench: Literal[0, 1, 2, 3] = 1
    T_temper: Annotated[FiniteFloat, Field(ge=150, le=700)] = 450
    t_temper: Annotated[FiniteFloat, Field(ge=.5, le=8)] = 2


class ForwardRequest(StrictModel):
    composition: dict[str, FiniteFloat]
    process: Process
    constraints: Equipment = Field(default_factory=Equipment)
