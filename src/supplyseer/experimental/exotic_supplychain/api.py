from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from datetime import datetime, date
from pydantic import BaseModel, Field
import uvicorn
import os
os.chdir("D:/my_py_packages/supplyseer")
from exotic_supplychain.geopolitical import *

# API Request/Response Models
class RouteAnalysisRequest(BaseModel):
    origin: str = Field(..., description="Starting point of the route")
    destination: str = Field(..., description="End point of the route")
    waypoints: List[str] = Field(default_factory=list, description="Intermediate stops")
    date: datetime = Field(default_factory=datetime.now, description="Date of shipment")
    
    class Config:
        json_schema_extra = {
            "example": {
                "origin": "Shanghai",
                "destination": "Rotterdam",
                "waypoints": ["Singapore", "Suez Canal", "Mediterranean"],
                "date": "2024-02-01"
            }
        }

class RouteAnalysisResponse(BaseModel):
    risk_score: float
    risk_level: str
    delay_factor: float
    cost_factor: float
    impacted_segments: List[Tuple[str, str]]
    estimated_delay_days: float
    additional_cost_percentage: float
    recommendations: List[str]

class CommodityAnalysisResponse(BaseModel):
    commodity: str
    risk_level: str
    affected_by_conflicts: List[dict]
    alternative_sources: List[str]
    impacts: Optional[List[dict]] = None
    sanctions_details: Optional[List[dict]] = None
    impact_summary: str
    recommendations: List[str]

class ConflictStatusResponse(BaseModel):
    name: str
    type: str
    severity: str
    affected_regions: List[str]
    impact_summary: dict
    start_date: datetime
    active: bool

# FastAPI App
app = FastAPI(
    title="Supply Chain Risk Analysis API",
    description="API for analyzing geopolitical risks in supply chains",
    version="1.0.0"
)

# Initialize risk module
risk_module = create_current_conflict_scenario()

@app.get("/")
async def root():
    return {
        "status": "operational",
        "version": "1.0.0",
        "current_active_conflicts": len(risk_module.conflict_zones)
    }

@app.post("/analyze/route", response_model=RouteAnalysisResponse)
async def analyze_route(request: RouteAnalysisRequest):
    try:
        # Combine waypoints into full route
        full_route = [request.origin] + request.waypoints + [request.destination]
        
        # Get risk assessment
        risk_assessment = risk_module.calculate_route_risk(
            origin=request.origin,
            destination=request.destination,
            route_waypoints=full_route,
            date=request.date
        )
        
        # Generate recommendations based on risk level
        recommendations = []
        if risk_assessment.risk_level.value >= RiskLevel.HIGH.value:
            alt_routes = risk_module.suggest_alternative_routes(
                request.origin, 
                request.destination,
                request.date
            )
            if alt_routes:
                recommendations.append(f"Consider alternative route: {' -> '.join(alt_routes[0]['route'])}")
        
        if risk_assessment.delay_factor > 1.5:
            recommendations.append("Consider increasing buffer time for this route")
        
        return RouteAnalysisResponse(
            risk_score=risk_assessment.risk_score,
            risk_level=risk_assessment.risk_level.value,
            delay_factor=risk_assessment.delay_factor,
            cost_factor=risk_assessment.cost_factor,
            impacted_segments=risk_assessment.impacted_segments,
            estimated_delay_days=(risk_assessment.delay_factor - 1) * 7,  # Rough estimate
            additional_cost_percentage=(risk_assessment.cost_factor - 1) * 100,
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/commodity/{commodity_name}", response_model=CommodityAnalysisResponse)
async def analyze_commodity(
    commodity_name: str,
    detailed: bool = Query(False, description="Include detailed conflict analysis")
):
    try:
        analysis = risk_module.analyze_commodity_risk(commodity_name)
        
        impacts = []
        sanctions_info = []
        all_alternative_sources = set(analysis.alternative_sources)  # Start with existing sources
        
        for conflict in analysis.affected_by_conflicts:
            conflict_obj = next(
                c for c in risk_module.conflict_zones 
                if c.name == conflict["conflict"]
            )
            
            impact_factors = risk_module._calculate_conflict_impact_factors(conflict_obj)
            impacts.append({
                "conflict_name": conflict_obj.name,
                "type": conflict_obj.conflict_type.value,
                "severity": conflict_obj.severity.value,
                "lead_time_increase": f"{(impact_factors['lead_time'] - 1) * 100:.1f}%",
                "cost_increase": f"{(impact_factors['cost'] - 1) * 100:.1f}%",
                "reliability_decrease": f"{(1 - impact_factors['reliability']) * 100:.1f}%"
            })
            
            # Update sanctions info and alternative sources
            for sanction in conflict_obj.sanctions:
                if commodity_name in sanction.restricted_items:
                    sanctions_info.append({
                        "restricted_items": sanction.restricted_items,
                        "cost_increase": f"{(sanction.cost_increase_factor - 1) * 100:.1f}%",
                        "lead_time_increase": f"{sanction.lead_time_increase} days"
                    })
                    all_alternative_sources.update(sanction.alternative_sources)

        # Generate recommendations based on actual severity
        recommendations = []
        has_high_impact = any(
            float(impact["cost_increase"].rstrip('%')) > 50 or
            float(impact["lead_time_increase"].rstrip('%')) > 50
            for impact in impacts
        )
        
        if analysis.risk_level.value >= RiskLevel.HIGH.value or has_high_impact:
            if all_alternative_sources:
                recommendations.append(
                    f"Consider sourcing from: {', '.join(all_alternative_sources)}"
                )
            recommendations.append("Increase safety stock levels")
            recommendations.append("Develop contingency sourcing plans")
            if has_high_impact:
                recommendations.append("Consider long-term contracts to stabilize costs")
                recommendations.append("Evaluate price hedging strategies")

        return CommodityAnalysisResponse(
            commodity=commodity_name,
            risk_level=analysis.risk_level.value,  # Now correctly propagated
            affected_by_conflicts=analysis.affected_by_conflicts,
            alternative_sources=list(all_alternative_sources),
            impacts=impacts if detailed else None,
            sanctions_details=sanctions_info if detailed else None,
            impact_summary=(
                f"This commodity is affected by {len(analysis.affected_by_conflicts)} "
                f"active conflicts with maximum severity level {analysis.risk_level.value}. "
                f"{'Trade restrictions apply. ' if sanctions_info else ''}"
                f"{'Alternative sources are available: ' + ', '.join(all_alternative_sources) if all_alternative_sources else 'No alternative sources identified.'}"
                + (f" Maximum cost impact is {max(impacts, key=lambda x: float(x['cost_increase'].rstrip('%')))['cost_increase']}." if impacts else "")
            ),
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conflicts/active", response_model=List[ConflictStatusResponse])
async def get_active_conflicts():
    """Get status of all active conflicts"""
    try:
        return [
            ConflictStatusResponse(
                name=conflict.name,
                type=conflict.conflict_type.value,
                severity=conflict.severity.value,
                affected_regions=conflict.affected_regions,
                impact_summary={
                    "displaced_population": conflict.displaced_population,
                    "strategic_commodities": len(conflict.strategic_commodities),
                    "impact_radius_km": conflict.impact_radius
                },
                start_date=conflict.start_date,
                active=True
            )
            for conflict in risk_module.conflict_zones
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)