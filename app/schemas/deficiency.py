"""
Pydantic schemas for Project Proposal and Deficiency Detection
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class DeficiencyType(str, Enum):
    """Types of deficiencies that can be detected."""
    MISSING = "missing"  # Field missing in one document
    MISMATCH = "mismatch"  # Values don't match
    INCONSISTENT = "inconsistent"  # Values are inconsistent (e.g., different units)
    NOT_FOUND = "not_found"  # Information not found in EIA
    VERIFIED = "verified"  # Information matches


class DeficiencySeverity(str, Enum):
    """Severity levels for deficiencies."""
    CRITICAL = "critical"  # Must be fixed
    HIGH = "high"  # Should be fixed
    MEDIUM = "medium"  # Recommended to fix
    LOW = "low"  # Minor issue
    INFO = "info"  # Informational only


# Project Proposal Schemas
class Address(BaseModel):
    address: str
    village_city: str
    state: str
    district: str
    pin_code: str
    landmark: Optional[str] = None


class ContactDetails(BaseModel):
    email: str
    mobile: str


class ProjectDetails(BaseModel):
    project_name: str
    project_proposal_for: str
    expansion_under_7ii_a: str
    project_id: str
    project_description: str


class OrganizationDetails(BaseModel):
    legal_status: str
    organization_name: str
    registered_address: Address
    contact_details: ContactDetails


class ApplicantDetails(BaseModel):
    name: str
    designation: str
    correspondence_address: Address
    contact_details: ContactDetails


class LocationDetails(BaseModel):
    topsheet_no: str
    state: str
    district: str
    sub_district: str
    village: str


class ProjectLocation(BaseModel):
    kml_uploaded: str
    international_border_state: str
    aerial_distance_from_border_km: float
    project_shape: str
    location_details: LocationDetails
    remarks: str


class LandBreakup(BaseModel):
    existing: float
    additional: float
    total_after_expansion: float


class LandRequirement(BaseModel):
    non_forest_land_ha: LandBreakup
    forest_land_ha: LandBreakup
    total_land_ha: float


class ProjectCost(BaseModel):
    existing_project_cost: float
    proposed_expansion_cost: float
    total_project_cost: float


class EmploymentDetails(BaseModel):
    permanent_employment: Dict[str, Any]
    temporary_employment_man_days: Optional[float] = None
    total_man_days: Optional[float] = None
    period_days: Optional[Dict[str, Any]] = None
    man_days: Optional[Dict[str, Any]] = None


class EmploymentGeneration(BaseModel):
    construction_phase: EmploymentDetails
    operational_phase: EmploymentDetails


class OtherInformation(BaseModel):
    rehabilitation_and_resettlement: str
    shifting_of_utilities_required: str
    alternative_site_examined: str
    government_order_or_policy_applicable: str
    litigation_pending: str
    violation_of_laws_or_rules: str


class ProjectProposal(BaseModel):
    """Complete project proposal schema."""
    project_details: ProjectDetails
    organization_details: OrganizationDetails
    applicant_details: ApplicantDetails
    project_location: ProjectLocation
    land_requirement: LandRequirement
    project_cost_lakhs: ProjectCost
    employment_generation: EmploymentGeneration
    other_information: OtherInformation


# Deficiency Detection Schemas
class DeficiencyItem(BaseModel):
    """Individual deficiency item."""
    field_name: str = Field(..., description="Name of the field being checked")
    field_path: str = Field(..., description="JSON path to the field")
    deficiency_type: DeficiencyType
    severity: DeficiencySeverity
    proposal_value: Optional[Any] = Field(None, description="Value in project proposal")
    eia_value: Optional[Any] = Field(None, description="Value found in EIA report")
    description: str = Field(..., description="Description of the deficiency")
    recommendation: Optional[str] = Field(None, description="Recommendation to fix")
    eia_reference: Optional[str] = Field(None, description="Reference/chunk from EIA")
    confidence_score: Optional[float] = Field(None, description="Confidence score (0-1)")


class CategoryDeficiencies(BaseModel):
    """Deficiencies grouped by category."""
    category_name: str
    total_checks: int
    deficiencies_found: int
    items: List[DeficiencyItem]


class DeficiencyReport(BaseModel):
    """Complete deficiency detection report."""
    project_id: str
    project_name: str
    timestamp: str
    total_fields_checked: int
    total_deficiencies: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    verified_count: int
    categories: List[CategoryDeficiencies]
    overall_compliance_score: float = Field(..., description="Overall compliance score (0-100)")
    summary: str


class ComparisonRequest(BaseModel):
    """Request model for deficiency detection."""
    project_proposal: ProjectProposal
    include_low_severity: bool = Field(default=False, description="Include low severity issues")
    top_k_rag_results: int = Field(default=5, description="Number of RAG results to retrieve per query")
