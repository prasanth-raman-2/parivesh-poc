"""
Deficiency Detection Service - Compares Project Proposal against EIA Report using RAG
"""

from typing import List, Dict, Any, Tuple
from datetime import datetime

from litellm import completion, embedding
from app.milvus import MilvusClient
from app.core.settings import settings


class DeficiencyDetectionService:
    def __init__(self):
        self.milvus_client = MilvusClient(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            collection_name=settings.MILVUS_COLLECTION_NAME,
            dim=settings.EMBEDDING_DIMENSION
        )
        self.embedding_model = "text-embedding-3-large"
        self.llm_model = "gpt-4o-mini"
        
    def connect(self):
        """Connect to Milvus and load collection."""
        self.milvus_client.connect()
        self.milvus_client.load_collection()
    
    def disconnect(self):
        """Disconnect from Milvus."""
        self.milvus_client.disconnect()
    
    def query_rag(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query RAG knowledge base for relevant information."""
        # Generate embedding
        response = embedding(model=self.embedding_model, input=query)
        query_embedding = response['data'][0]['embedding']
        
        # Search Milvus
        results = self.milvus_client.search(
            query_embeddings=[query_embedding],
            top_k=top_k,
            output_fields=["text", "metadata"]
        )
        
        return results[0] if results else []
    
    def extract_value_from_rag(
        self,
        field_name: str,
        expected_value: Any,
        context_chunks: List[Dict[str, Any]]
    ) -> Tuple[Any, str, float]:
        """
        Use LLM to extract and verify value from RAG context.
        
        Returns:
            Tuple of (extracted_value, reference_text, confidence_score)
        """
        if not context_chunks:
            return None, "No relevant information found in EIA report", 0.0
        
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(f"[Chunk {i}]:\n{chunk['text'][:500]}\n")
        
        context_text = "\n".join(context_parts)
        
        # LLM prompt for extraction
        prompt = f"""You are analyzing an EIA (Environmental Impact Assessment) report to verify information.

Field to verify: {field_name}
Expected value from proposal: {expected_value}

EIA Report Context:
{context_text}

Task: Extract the corresponding value for "{field_name}" from the EIA report context above.

Respond in JSON format:
{{
    "found": true/false,
    "extracted_value": "the value you found or null",
    "confidence": 0.0-1.0,
    "reference": "brief quote from context showing where you found it",
    "matches_expected": true/false/null,
    "explanation": "brief explanation"
}}"""
        
        try:
            response = completion(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            extracted_value = result.get("extracted_value")
            confidence = result.get("confidence", 0.5)
            reference = result.get("reference", "")
            
            return extracted_value, reference, confidence
        
        except Exception as e:
            return None, f"Error extracting value: {str(e)}", 0.0
    
    def safe_get(self, data: Dict, *keys, default=None):
        """Safely get nested dictionary value."""
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, {})
            else:
                return default
        return data if data != {} else default
    
    def detect_deficiencies(
        self,
        proposal: Dict[str, Any],
        top_k: int = 5,
        include_low: bool = False
    ) -> Dict[str, Any]:
        """
        Main method to detect deficiencies by comparing proposal with EIA report.
        """
        deficiencies = []
        
        # Extract values safely from proposal
        project_name = self.safe_get(proposal, 'project_details', 'project_name')
        project_id = self.safe_get(proposal, 'project_details', 'project_id')
        village = self.safe_get(proposal, 'project_location', 'location_details', 'village')
        district = self.safe_get(proposal, 'project_location', 'location_details', 'district')
        state = self.safe_get(proposal, 'project_location', 'location_details', 'state')
        org_name = self.safe_get(proposal, 'organization_details', 'organization_name')
        total_land = self.safe_get(proposal, 'land_requirement', 'total_land_ha')
        
        # Check 1: Project Name
        if project_name:
            query = f"project name expansion PCBL carbon black {project_name}"
            chunks = self.query_rag(query, top_k)
            eia_value, reference, confidence = self.extract_value_from_rag(
                "Project Name", project_name, chunks
            )
            
            status = "verified" if (eia_value and confidence > 0.6) else "not_found"
            severity = "info" if status == "verified" else "high"
            
            deficiencies.append({
                "field": "Project Name",
                "status": status,
                "severity": severity,
                "proposal_value": project_name,
                "eia_value": eia_value,
                "confidence": round(confidence, 2) if confidence else 0,
                "reference": reference[:200] if reference else ""
            })
        
        # Check 2: Production Capacity
        if project_name and ("550" in str(project_name) or "675" in str(project_name)):
            query = "production capacity TPD carbon black expansion 550 675"
            chunks = self.query_rag(query, top_k)
            eia_value, reference, confidence = self.extract_value_from_rag(
                "Production Capacity", "550 to 675 TPD", chunks
            )
            
            status = "verified" if (eia_value and confidence > 0.6) else "not_found"
            severity = "info" if status == "verified" else "critical"
            
            deficiencies.append({
                "field": "Production Capacity",
                "status": status,
                "severity": severity,
                "proposal_value": "550 to 675 TPD",
                "eia_value": eia_value,
                "confidence": round(confidence, 2) if confidence else 0,
                "reference": reference[:200] if reference else ""
            })
        
        # Check 3: Village
        if village:
            query = f"village location {village}"
            chunks = self.query_rag(query, top_k)
            eia_value, reference, confidence = self.extract_value_from_rag(
                "Village", village, chunks
            )
            
            status = "verified" if (eia_value and confidence > 0.6) else "not_found"
            severity = "info" if status == "verified" else "high"
            
            deficiencies.append({
                "field": "Village",
                "status": status,
                "severity": severity,
                "proposal_value": village,
                "eia_value": eia_value,
                "confidence": round(confidence, 2) if confidence else 0,
                "reference": reference[:200] if reference else ""
            })
        
        # Check 4: District
        if district:
            query = f"district location {district}"
            chunks = self.query_rag(query, top_k)
            eia_value, reference, confidence = self.extract_value_from_rag(
                "District", district, chunks
            )
            
            status = "verified" if (eia_value and confidence > 0.6) else "not_found"
            severity = "info" if status == "verified" else "critical"
            
            deficiencies.append({
                "field": "District",
                "status": status,
                "severity": severity,
                "proposal_value": district,
                "eia_value": eia_value,
                "confidence": round(confidence, 2) if confidence else 0,
                "reference": reference[:200] if reference else ""
            })
        
        # Check 5: State
        if state:
            query = f"state location {state}"
            chunks = self.query_rag(query, top_k)
            eia_value, reference, confidence = self.extract_value_from_rag(
                "State", state, chunks
            )
            
            status = "verified" if (eia_value and confidence > 0.6) else "not_found"
            severity = "info" if status == "verified" else "high"
            
            deficiencies.append({
                "field": "State",
                "status": status,
                "severity": severity,
                "proposal_value": state,
                "eia_value": eia_value,
                "confidence": round(confidence, 2) if confidence else 0,
                "reference": reference[:200] if reference else ""
            })
        
        # Check 6: Organization Name
        if org_name:
            query = f"organization company name {org_name} PCBL"
            chunks = self.query_rag(query, top_k)
            eia_value, reference, confidence = self.extract_value_from_rag(
                "Organization Name", org_name, chunks
            )
            
            status = "verified" if (eia_value and confidence > 0.6) else "not_found"
            severity = "info" if status == "verified" else "high"
            
            deficiencies.append({
                "field": "Organization Name",
                "status": status,
                "severity": severity,
                "proposal_value": org_name,
                "eia_value": eia_value,
                "confidence": round(confidence, 2) if confidence else 0,
                "reference": reference[:200] if reference else ""
            })
        
        # Check 7: Total Land Area
        if total_land:
            query = f"land area hectare ha {total_land}"
            chunks = self.query_rag(query, top_k)
            eia_value, reference, confidence = self.extract_value_from_rag(
                "Total Land Area", f"{total_land} Ha", chunks
            )
            
            status = "verified" if (eia_value and confidence > 0.6) else "not_found"
            severity = "info" if status == "verified" else "medium"
            
            deficiencies.append({
                "field": "Total Land Area",
                "status": status,
                "severity": severity,
                "proposal_value": f"{total_land} Ha",
                "eia_value": eia_value,
                "confidence": round(confidence, 2) if confidence else 0,
                "reference": reference[:200] if reference else ""
            })
        
        # Calculate statistics
        verified_count = sum(1 for d in deficiencies if d['status'] == 'verified')
        not_found_count = sum(1 for d in deficiencies if d['status'] == 'not_found')
        critical_count = sum(1 for d in deficiencies if d['severity'] == 'critical')
        high_count = sum(1 for d in deficiencies if d['severity'] == 'high')
        medium_count = sum(1 for d in deficiencies if d['severity'] == 'medium')
        
        total_checked = len(deficiencies)
        compliance_score = (verified_count / total_checked * 100) if total_checked > 0 else 0
        
        return {
            "project_id": project_id or "N/A",
            "project_name": project_name or "N/A",
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_fields_checked": total_checked,
                "verified_count": verified_count,
                "deficiencies_found": not_found_count,
                "critical_count": critical_count,
                "high_count": high_count,
                "medium_count": medium_count,
                "compliance_score": round(compliance_score, 2)
            },
            "deficiencies": deficiencies
        }

