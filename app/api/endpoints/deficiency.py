"""
API Router for Deficiency Detection
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Body
from fastapi.responses import JSONResponse
import json
from typing import Optional, Dict, Any

from app.services.deficiency_detection import DeficiencyDetectionService

router = APIRouter(prefix="/deficiency", tags=["Deficiency Detection"])


@router.post("/detect")
async def detect_deficiencies(
    project_proposal: Dict[str, Any] = Body(...),
    include_low_severity: bool = Body(default=False),
    top_k_rag_results: int = Body(default=5)
):
    """
    Detect deficiencies by comparing project proposal against EIA report in RAG knowledge base.
    
    Args:
        project_proposal: Project proposal JSON object
        include_low_severity: Include low severity deficiencies
        top_k_rag_results: Number of RAG results to retrieve per query
        
    Returns:
        Deficiency report with detailed analysis
    """
    try:
        # Initialize service
        service = DeficiencyDetectionService()
        
        # Connect to Milvus
        service.connect()
        
        # Detect deficiencies
        report = await service.detect_deficiencies(
            proposal=project_proposal,
            top_k=top_k_rag_results,
            include_low=include_low_severity
        )
        
        # Disconnect
        service.disconnect()
        
        return report
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting deficiencies: {str(e)}")


@router.post("/detect-from-file")
async def detect_deficiencies_from_file(
    file: UploadFile = File(...),
    include_low_severity: bool = False,
    top_k_rag_results: int = 5
):
    """
    Detect deficiencies by uploading a project proposal JSON file.
    
    Args:
        file: JSON file containing project proposal
        include_low_severity: Include low severity deficiencies
        top_k_rag_results: Number of RAG results to retrieve per query
        
    Returns:
        Deficiency report with detailed analysis
    """
    try:
        # Read and parse file
        contents = await file.read()
        proposal_data = json.loads(contents)
        
        # Initialize service
        service = DeficiencyDetectionService()
        
        # Connect to Milvus
        service.connect()
        
        # Detect deficiencies
        report = await service.detect_deficiencies(
            proposal=proposal_data,
            top_k=top_k_rag_results,
            include_low=include_low_severity
        )
        
        # Disconnect
        service.disconnect()
        
        return report
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/health")
async def health_check():
    """Check if deficiency detection service is healthy."""
    try:
        service = DeficiencyDetectionService()
        service.connect()
        
        # Get collection stats
        stats = service.milvus_client.get_collection_stats()
        
        service.disconnect()
        
        return {
            "status": "healthy",
            "milvus_connected": True,
            "knowledge_base_chunks": stats.get("num_entities", 0),
            "collection_name": stats.get("name")
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )
