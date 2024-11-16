from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from pydantic import BaseModel, Field, validator
from enum import Enum
from itertools import combinations

class PlayerType(Enum):
    SUPPLIER = "supplier"
    MANUFACTURER = "manufacturer"
    RETAILER = "retailer"

class Player(BaseModel):
    """
    Pydantic basemodel for the Player, inspired by section 2 of "Game Theory In Supply Chain Analysis chapter 2 section 2" - G.P Cachon and S. Netessine
    """
    id: str
    type: PlayerType
    capacity: float = Field(gt=0)
    production_cost: float = Field(gt=0)
    holding_cost: float = Field(ge=0)
    setup_cost: float = Field(ge=0)
    market_power: float = Field(ge=0, le=1)

    def __hash__(self):
        return hash(self.id)
        
    def __eq__(self, other):
        if not isinstance(other, Player):
            return False
        return self.id == other.id

class Coalition(BaseModel):
    """
    Pydantic basemodel for the Player, inspired by section 4 of "Game Theory In Supply Chain Analysis chapter 2 section 4" - G.P Cachon and S. Netessine
    """
    members: Set[Player]
    value: float = Field(default=0.0)
    stability_index: float = Field(default=1.0)
    contributions: Dict[str, float] = Field(default_factory=dict)
    
    def __hash__(self):
        return hash(frozenset(p.id for p in self.members))
        
    def __eq__(self, other):
        if not isinstance(other, Coalition):
            return False
        return {p.id for p in self.members} == {p.id for p in other.members}

class Partition(BaseModel):
    """
    Pydantic basemodel for Partition form games, inspired by "Partition-form Cooperative Games in Two-Echelon Supply Chains" - G. Wadhwa, T.S. Walunj, V. Kavitha
    """
    coalitions: Set[Coalition]
    
    @validator('coalitions')
    def validate_disjoint_coalitions(cls, v):
        all_players: Set[str] = set()
        for coalition in v:
            member_ids = {p.id for p in coalition.members}
            if any(pid in all_players for pid in member_ids):
                raise ValueError("Coalitions must be disjoint")
            all_players.update(member_ids)
        return v

    def __hash__(self):
        return hash(frozenset(c.members for c in self.coalitions))