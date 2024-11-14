from datetime import datetime, timedelta, date
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np
import pandas as pd

# Enums
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SEVERE = "severe"
    CRITICAL = "critical"

    @property
    def numeric_value(self) -> int:
        risk_values = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "severe": 4,
            "critical": 5
        }
        return risk_values[self.value]

class RiskCategory(str, Enum):
    POLITICAL = "political"
    TRADE = "trade"
    SECURITY = "security"
    REGULATORY = "regulatory"
    ECONOMIC = "economic"

class ConflictType(str, Enum):
    ARMED_CONFLICT = "armed_conflict"
    TRADE_SANCTIONS = "trade_sanctions"
    TERRITORIAL_DISPUTE = "territorial_dispute"
    CIVIL_UNREST = "civil_unrest"
    ECONOMIC_WARFARE = "economic_warfare"

# Pydantic Models
class ImpactFactors(BaseModel):
    lead_time: float = Field(ge=1.0, default=1.0)
    cost: float = Field(ge=1.0, default=1.0)
    reliability: float = Field(gt=0.0, le=1.0, default=1.0)

class SanctionImpact(BaseModel):
    restricted_items: List[str]
    alternative_sources: List[str]
    cost_increase_factor: float = Field(ge=1.0)
    lead_time_increase: int = Field(ge=0)  # in days
    reliability_decrease: float = Field(ge=0.0, le=1.0)

class GeopoliticalEvent(BaseModel):
    name: str
    region: str
    risk_level: RiskLevel
    categories: List[RiskCategory]
    start_date: datetime
    end_date: Optional[datetime] = None
    affected_routes: List[str] = Field(default_factory=list)
    impact_factors: ImpactFactors = Field(default_factory=ImpactFactors)

    @model_validator(mode='after')
    def validate_dates(self) -> 'GeopoliticalEvent':
        if self.end_date and self.start_date > self.end_date:
            raise ValueError("end_date must be after start_date")
        return self

class RiskZone(BaseModel):
    base_risk_level: RiskLevel
    affected_countries: List[str]
    impact_radius_km: float = Field(ge=0)

class ConflictZone(BaseModel):
    name: str
    conflict_type: ConflictType
    affected_regions: List[str]
    start_date: datetime
    severity: RiskLevel
    displaced_population: int = Field(ge=0, default=0)
    sanctions: List[SanctionImpact] = Field(default_factory=list)
    strategic_commodities: List[str] = Field(default_factory=list)

    @property
    def impact_radius(self) -> float:
        base_radius = {
            RiskLevel.LOW: 100,
            RiskLevel.MEDIUM: 300,
            RiskLevel.HIGH: 500,
            RiskLevel.SEVERE: 800,
            RiskLevel.CRITICAL: 1200
        }
        
        type_multiplier = {
            ConflictType.ARMED_CONFLICT: 1.5,
            ConflictType.TRADE_SANCTIONS: 1.0,
            ConflictType.TERRITORIAL_DISPUTE: 1.2,
            ConflictType.CIVIL_UNREST: 0.8,
            ConflictType.ECONOMIC_WARFARE: 1.0
        }
        
        return base_radius[self.severity] * type_multiplier[self.conflict_type]

class RouteRiskAssessment(BaseModel):
    risk_score: float = Field(ge=1.0)
    delay_factor: float = Field(ge=1.0)
    cost_factor: float = Field(ge=1.0)
    impacted_segments: List[Tuple[str, str]]
    risk_level: RiskLevel

class CommodityRiskAnalysis(BaseModel):
    risk_level: RiskLevel
    affected_by_conflicts: List[Dict]
    alternative_sources: List[str]

class ConflictImpactReport(BaseModel):
    conflict_name: str
    type: str
    severity: str
    affected_regions: str
    displaced_population: int
    strategic_commodities: int
    lead_time_impact: str
    cost_impact: str
    reliability_impact: str
    impact_radius_km: float

class GeopoliticalRiskModule:
    def __init__(self):
        self.risk_zones: Dict[str, RiskZone] = {}
        self.active_events: List[GeopoliticalEvent] = []
        self.trade_restrictions: Dict[Tuple[str, str], Dict] = {}
        self.alternative_routes: Dict[Tuple[str, str], List[List[str]]] = {}
        
        self.risk_weights = {
            RiskLevel.LOW: 1.1,
            RiskLevel.MEDIUM: 1.25,
            RiskLevel.HIGH: 1.5,
            RiskLevel.SEVERE: 2.0,
            RiskLevel.CRITICAL: 3.0
        }
        
        self._initialize_base_risk_levels()

    def _initialize_base_risk_levels(self):
        self.risk_zones = {
            "Red Sea": RiskZone(
                base_risk_level=RiskLevel.CRITICAL,
                affected_countries=["Yemen", "Saudi Arabia", "Egypt"],
                impact_radius_km=500
            ),
            "Eastern Europe": RiskZone(
                base_risk_level=RiskLevel.HIGH,
                affected_countries=["Ukraine", "Russia", "Belarus"],
                impact_radius_km=300
            ),
            "South China Sea": RiskZone(
                base_risk_level=RiskLevel.MEDIUM,
                affected_countries=["China", "Vietnam", "Philippines"],
                impact_radius_km=400
            )
        }

    def suggest_alternative_routes(
        self,
        origin: str,
        destination: str,
        date: date,
        max_alternatives: int = 3
    ) -> List[Dict]:
        """Suggest alternative routes based on risk levels"""
        if (origin, destination) not in self.alternative_routes:
            return []
            
        alternatives = []
        for route in self.alternative_routes[(origin, destination)]:
            risk_assessment = self.calculate_route_risk(
                origin=origin,
                destination=destination,
                route_waypoints=route,
                date=date
            )
            alternatives.append({
                "route": route,
                "risk_assessment": risk_assessment
            })
        
        # Sort by risk score and return top alternatives
        alternatives.sort(key=lambda x: x["risk_assessment"].risk_score)
        return alternatives[:max_alternatives]

    def _convert_score_to_risk_level(self, risk_score: float) -> RiskLevel:
        """Convert numerical risk score to RiskLevel enum"""
        if risk_score < 1.2:
            return RiskLevel.LOW
        elif risk_score < 1.5:
            return RiskLevel.MEDIUM
        elif risk_score < 2.0:
            return RiskLevel.HIGH
        elif risk_score < 3.0:
            return RiskLevel.SEVERE
        else:
            return RiskLevel.CRITICAL

    def add_geopolitical_event(self, event: GeopoliticalEvent) -> None:
        self.active_events.append(event)
        print(f"Added event: {event.name} in {event.region} with risk level {event.risk_level.value}")
        
        if event.region not in self.risk_zones:
            self.risk_zones[event.region] = RiskZone(
                base_risk_level=event.risk_level,
                affected_countries=[],
                impact_radius_km=200
            )
        elif event.risk_level.value > self.risk_zones[event.region].base_risk_level.value:
            self.risk_zones[event.region].base_risk_level = event.risk_level

    def calculate_route_risk(
        self,
        origin: str,
        destination: str,
        route_waypoints: List[str],
        date: datetime
    ) -> RouteRiskAssessment:
        total_risk_score = 1.0
        impacted_segments = []
        delay_factor = 1.0
        cost_factor = 1.0
        
        for i in range(len(route_waypoints) - 1):
            start = route_waypoints[i]
            end = route_waypoints[i+1]
            segment_risks = self._assess_segment_risks(start, end, date)
            
            total_risk_score *= segment_risks["risk_score"]
            delay_factor *= segment_risks["delay_factor"]
            cost_factor *= segment_risks["cost_factor"]
            
            if segment_risks["risk_score"] > 1.2:
                impacted_segments.append((start, end))

        return RouteRiskAssessment(
            risk_score=total_risk_score,
            delay_factor=delay_factor,
            cost_factor=cost_factor,
            impacted_segments=impacted_segments,
            risk_level=self._convert_score_to_risk_level(total_risk_score)
        )

class EnhancedGeopoliticalRiskModule(GeopoliticalRiskModule):
    def __init__(self):
        super().__init__()
        self.conflict_zones: List[ConflictZone] = []
        self.strategic_commodities: Dict[str, List[str]] = {}
        self.sanctions_registry: Dict[str, List[SanctionImpact]] = {}
        self.displacement_impact: Dict[str, float] = {}

    def _calculate_conflict_impact_factors(self, conflict: ConflictZone) -> Dict[str, float]:
        """Calculate impact factors based on conflict characteristics"""
        type_impacts = {
            ConflictType.ARMED_CONFLICT: {"lead_time": 2.0, "cost": 2.5, "reliability": 0.5},
            ConflictType.TRADE_SANCTIONS: {"lead_time": 1.5, "cost": 1.8, "reliability": 0.7},
            ConflictType.TERRITORIAL_DISPUTE: {"lead_time": 1.3, "cost": 1.4, "reliability": 0.8},
            ConflictType.CIVIL_UNREST: {"lead_time": 1.4, "cost": 1.3, "reliability": 0.75},
            ConflictType.ECONOMIC_WARFARE: {"lead_time": 1.2, "cost": 1.6, "reliability": 0.85}
        }
        
        impact = type_impacts[conflict.conflict_type].copy()
        
        # Adjust based on severity
        severity_multiplier = self.risk_weights[conflict.severity]
        for factor in impact:
            if factor == "reliability":
                impact[factor] = max(0.1, impact[factor] / severity_multiplier)
            else:
                impact[factor] *= severity_multiplier
                
        return impact
    
    
    def _assess_segment_risks(
        self,
        start: str,
        end: str,
        date: date
    ) -> Dict[str, float]:
        """Assess risks for a specific route segment"""
        risk_score = 1.0
        delay_factor = 1.0
        cost_factor = 1.0
        
        # Check active events affecting this segment
        for event in self.active_events:
            if (date >= event.start_date and 
                (event.end_date is None or date <= event.end_date)):
                
                # Check if segment is in affected routes or regions
                route_key = f"{start}-{end}"
                if (route_key in event.affected_routes or 
                    start in event.affected_routes or 
                    end in event.affected_routes):
                    
                    risk_multiplier = self.risk_weights[event.risk_level]
                    risk_score *= risk_multiplier
                    delay_factor *= event.impact_factors.lead_time
                    cost_factor *= event.impact_factors.cost

        # Check if any conflict zones affect this segment
        for conflict in self.conflict_zones:
            if start in conflict.affected_regions or end in conflict.affected_regions:
                impact_factors = self._calculate_conflict_impact_factors(conflict)
                risk_score *= self.risk_weights[conflict.severity]
                delay_factor *= impact_factors['lead_time']
                cost_factor *= impact_factors['cost']
        
        # Check trade restrictions
        if (start, end) in self.trade_restrictions:
            restriction = self.trade_restrictions[(start, end)]
            risk_score *= restriction.get("risk_multiplier", 1.5)
            delay_factor *= restriction.get("delay_multiplier", 1.3)
            cost_factor *= restriction.get("cost_multiplier", 1.4)
        
        return {
            "risk_score": risk_score,
            "delay_factor": delay_factor,
            "cost_factor": cost_factor
        }

    def add_conflict_zone(self, conflict: ConflictZone) -> None:
        """Add a new conflict zone and calculate its supply chain impacts"""
        self.conflict_zones.append(conflict)
        
        event = GeopoliticalEvent(
            name=conflict.name,
            region=conflict.affected_regions[0],
            risk_level=conflict.severity,
            categories=[RiskCategory.SECURITY, RiskCategory.POLITICAL],
            start_date=conflict.start_date,
            impact_factors=ImpactFactors(**self._calculate_conflict_impact_factors(conflict))
        )
        self.add_geopolitical_event(event)
        
        # Update strategic commodities registry
        for commodity in conflict.strategic_commodities:
            if commodity not in self.strategic_commodities:
                self.strategic_commodities[commodity] = []
            self.strategic_commodities[commodity].append(conflict.name)
            
        self._update_displacement_impact(conflict)

    def _update_displacement_impact(self, conflict: ConflictZone) -> None:
        """Calculate impact of population displacement on supply chain"""
        for region in conflict.affected_regions:
            if region not in self.displacement_impact:
                self.displacement_impact[region] = 0
            
            # Calculate impact based on displaced population
            impact_factor = min(1.0, conflict.displaced_population / 1000000)  # Scale by millions
            self.displacement_impact[region] += impact_factor

    def analyze_commodity_risk(self, commodity: str) -> CommodityRiskAnalysis:
        """Analyze risk for specific strategic commodities"""
        if commodity not in self.strategic_commodities:
            return CommodityRiskAnalysis(
                risk_level=RiskLevel.LOW,
                affected_by_conflicts=[],
                alternative_sources=[]
            )
            
        affecting_conflicts = self.strategic_commodities[commodity]
        max_risk = RiskLevel.LOW
        impact_analysis = []
        all_alternative_sources = set()
        
        for conflict_name in affecting_conflicts:
            conflict = next(c for c in self.conflict_zones if c.name == conflict_name)
            # Update max risk based on conflict severity
            if conflict.severity.numeric_value > max_risk.numeric_value:
                max_risk = conflict.severity
                
            # Collect all alternative sources from sanctions
            for sanction in conflict.sanctions:
                if commodity in sanction.restricted_items:
                    all_alternative_sources.update(sanction.alternative_sources)
            
            impact_analysis.append({
                "conflict": conflict_name,
                "severity": conflict.severity,
                "affected_regions": conflict.affected_regions,
                "sanctions": [s.restricted_items for s in conflict.sanctions if commodity in s.restricted_items]
            })
                
        return CommodityRiskAnalysis(
            risk_level=max_risk,  # Now properly reflects highest severity
            affected_by_conflicts=impact_analysis,
            alternative_sources=list(all_alternative_sources)
        )

    def _find_alternative_sources(self, commodity: str) -> List[str]:
        """Find alternative sources for strategic commodities"""
        all_sources = set()
        affected_sources = set()
        
        for conflict in self.conflict_zones:
            if commodity in conflict.strategic_commodities:
                affected_sources.update(conflict.affected_regions)
                
            for sanction in conflict.sanctions:
                if commodity in sanction.restricted_items:
                    all_sources.update(sanction.alternative_sources)
                    
        return list(all_sources - affected_sources)

    def generate_conflict_impact_report(self) -> List[ConflictImpactReport]:
        """Generate comprehensive conflict impact report"""
        report_data = []
        
        for conflict in self.conflict_zones:
            impact_factors = self._calculate_conflict_impact_factors(conflict)
            
            report_data.append(ConflictImpactReport(
                conflict_name=conflict.name,
                type=conflict.conflict_type.value,
                severity=conflict.severity.value,
                affected_regions=", ".join(conflict.affected_regions),
                displaced_population=conflict.displaced_population,
                strategic_commodities=len(conflict.strategic_commodities),
                lead_time_impact=f"{(impact_factors['lead_time'] - 1) * 100:.1f}%",
                cost_impact=f"{(impact_factors['cost'] - 1) * 100:.1f}%",
                reliability_impact=f"{(1 - impact_factors['reliability']) * 100:.1f}%",
                impact_radius_km=conflict.impact_radius
            ))
            
        return report_data

def create_current_conflict_scenario() -> EnhancedGeopoliticalRiskModule:
    risk_module = EnhancedGeopoliticalRiskModule()
    
    ukraine_conflict = ConflictZone(
        name="Ukraine-Russia Conflict",
        conflict_type=ConflictType.ARMED_CONFLICT,
        affected_regions=["Ukraine", "Russia", "Belarus"],
        start_date=datetime(2022, 2, 24),
        severity=RiskLevel.CRITICAL,
        displaced_population=8000000,
        strategic_commodities=["grain", "metals", "energy", "fertilizers"],
        sanctions=[
            SanctionImpact(
                restricted_items=["technology", "industrial equipment", "luxury goods"],
                alternative_sources=["EU", "USA", "Japan"],
                cost_increase_factor=1.8,
                lead_time_increase=45,
                reliability_decrease=0.4
            )
        ]
    )
    
    taiwan_tension = ConflictZone(
        name="Taiwan Strait Tensions",
        conflict_type=ConflictType.TERRITORIAL_DISPUTE,
        affected_regions=["Taiwan", "South China Sea"],
        start_date=datetime(2024, 1, 1),
        severity=RiskLevel.HIGH,
        strategic_commodities=["semiconductors", "electronics", "computer chips"],
        sanctions=[
            SanctionImpact(
                restricted_items=["advanced semiconductors", "chip manufacturing equipment"],
                alternative_sources=["South Korea", "Japan", "USA"],
                cost_increase_factor=1.5,
                lead_time_increase=30,
                reliability_decrease=0.3
            )
        ]
    )
    
    red_sea_crisis = ConflictZone(
        name="Red Sea Shipping Crisis",
        conflict_type=ConflictType.ARMED_CONFLICT,
        affected_regions=["Red Sea", "Gulf of Aden"],
        start_date=datetime(2024, 1, 1),
        severity=RiskLevel.SEVERE,
        strategic_commodities=["oil", "consumer goods"],
        sanctions=[]
    )
    
    risk_module.add_conflict_zone(ukraine_conflict)
    risk_module.add_conflict_zone(taiwan_tension)
    risk_module.add_conflict_zone(red_sea_crisis)
    
    return risk_module