from typing import Dict, List, Set, Tuple, Optional, Union
import numpy as np
from pydantic import BaseModel, Field, validator
from enum import Enum

class PlayerType(Enum):
    SUPPLIER = "supplier"
    MANUFACTURER = "manufacturer"
    RETAILER = "retailer"

class Player(BaseModel):
    id: str
    type: PlayerType
    capacity: float = Field(gt=0)
    production_cost: float = Field(gt=0)
    holding_cost: float = Field(ge=0)
    setup_cost: float = Field(ge=0)
    market_power: float = Field(ge=0, le=1)
    
    class Config:
        frozen = True

class Coalition(BaseModel):
    members: Set[Player]
    value: float = Field(default=0.0)
    stability_index: float = Field(default=1.0)
    contributions: Dict[Player, float] = Field(default_factory=dict)
    partition: Optional['Partition'] = None

    class Config:
        arbitrary_types_allowed = True

class Partition(BaseModel):
    coalitions: Set[Coalition]
    
    @validator('coalitions')
    def validate_disjoint_coalitions(cls, v):
        all_players = set()
        for coalition in v:
            if any(p in all_players for p in coalition.members):
                raise ValueError("Coalitions must be disjoint")
            all_players.update(coalition.members)
        return v