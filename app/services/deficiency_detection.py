"""
Deficiency Detection Service - Compares Project Proposal against EIA Report using RAG
"""

from typing import List, Dict, Any, Tuple
from datetime import datetime
import json
import asyncio

from attr import field
from litellm import completion, embedding, acompletion, aembedding
from app.milvus import MilvusClient
from app.core.settings import settings
from app.models.model_catalogue import LLMModels, EmbeddingModels


class DeficiencyDetectionService:
    def __init__(self):
        self.milvus_client = MilvusClient(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            collection_name=settings.MILVUS_COLLECTION_NAME,
            dim=settings.EMBEDDING_DIMENSION
        )
        self.embedding_model = EmbeddingModels.COHERE_EMBED_ENGLISH_V3.value
        self.llm_model = LLMModels.CLAUDE_3_SONNET.value
        
    def connect(self):
        """Connect to Milvus and load collection."""
        self.milvus_client.connect()
        self.milvus_client.load_collection()
    
    def disconnect(self):
        """Disconnect from Milvus."""
        self.milvus_client.disconnect()
    
    async def query_rag_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query RAG knowledge base for relevant information (async version)."""
        # Generate embedding
        response = await aembedding(model=self.embedding_model, input=query)
        query_embedding = response['data'][0]['embedding']
        
        # Search Milvus
        results = self.milvus_client.search(
            query_embeddings=[query_embedding],
            top_k=top_k,
            output_fields=["text", "metadata"]
        )
        
        return results[0] if results else []
    
    def query_rag(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query RAG knowledge base for relevant information (sync wrapper)."""
        return asyncio.run(self.query_rag_async(query, top_k))
    
    async def extract_value_from_rag_async(
        self,
        field_name: str,
        expected_value: Any,
        context_chunks: List[Dict[str, Any]]
    ) -> Tuple[Any, str, float, bool]:
        """
        Use LLM to extract and verify value from RAG context (async version).
        
        Returns:
            Tuple of (extracted_value, reference_text, confidence_score, matches)
        """
        if not context_chunks:
            return None, "No relevant information found in EIA report", 0.0, False
        
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
    "matches_expected": true/false/null,
    "explanation": "brief explanation"
}}"""
        
        try:
            response = await acompletion(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            extracted_value = result.get("extracted_value")
            confidence = result.get("confidence", 0.5)
            reference = context_text
            matches = result.get("matches_expected", False)
            
            return extracted_value, reference, confidence, matches
        
        except Exception as e:
            return None, f"Error extracting value: {str(e)}", 0.0, False
    
    def flatten_json(self, data: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flatten nested JSON structure.
        
        Returns:
            Dictionary with flattened keys and values
        """
        items = []
        
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.extend(self.flatten_json(value, new_key, sep=sep).items())
            elif isinstance(value, list):
                # For lists, convert to string representation
                items.append((new_key, str(value)))
            else:
                items.append((new_key, value))
        
        return dict(items)
    
    async def detect_deficiencies(
        self,
        proposal: Dict[str, Any],
        top_k: int = 5,
        include_low: bool = False,
        max_concurrent: int = 5
    ) -> Dict[str, Any]:
        """
        Main method to detect deficiencies by validating ALL fields in proposal against EIA report.
        Uses parallel processing for faster validation.
        
        Returns comprehensive report with verification status for each field.
        """
        print(f"\n{'='*80}")
        print("Starting deficiency detection for all fields (parallel processing)...")
        print(f"{'='*80}\n")
        
        # Flatten the proposal to get all fields
        flattened = self.flatten_json(proposal)
        
        total_fields = len(flattened)
        
        # Filter out empty values
        fields_to_check = [
            (field_path, field_value)
            for field_path, field_value in flattened.items()
            if field_value is not None and field_value != "" and field_value != "N/A"
        ]
        
        print(f"Processing {len(fields_to_check)} fields with parallel processing (max {max_concurrent} concurrent)\n")
        
        # Process fields in parallel with rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_field(idx: int, field_path: str, field_value: Any):
            async with semaphore:
                print(f"[{idx}/{len(fields_to_check)}] Checking: {field_path} = {field_value}")
                
                # Create search query
                field_name = field_path.split('.')[-1].replace('_', ' ').title()
                query = f" {field_name} {field_value}"
                
                # Query RAG
                chunks = await self.query_rag_async(query, 20)
                
                # Extract and verify
                eia_value, reference, confidence, matches = await self.extract_value_from_rag_async(
                    field_name, field_value, chunks
                )
                
                # Determine verification status
                is_verified = (eia_value is not None and confidence > 0.5) or matches
                
                field_result = {
                    "field_path": field_path,
                    "field_name": field_name,
                    "proposal_value": field_value,
                    "eia_value": eia_value,
                    "verified": is_verified,
                    "confidence": confidence,
                    "rag_reference": reference,
                    "matches": matches
                }
                
                if is_verified:
                    print(f"  ✓ VERIFIED (confidence: {confidence:.2f})")
                else:
                    print(f"  ✗ NOT VERIFIED (confidence: {confidence:.2f})")
                
                return field_result
        
        # Process all fields in parallel
        tasks = [
            process_field(idx, field_path, field_value)
            for idx, (field_path, field_value) in enumerate(fields_to_check, 1)
        ]
        
        field_results = await asyncio.gather(*tasks)
        
        # Separate verified and not verified
        verified_fields = [r for r in field_results if r["verified"]]
        not_verified_fields = [r for r in field_results if not r["verified"]]
        
        # Calculate statistics
        verified_count = len(verified_fields)
        not_verified_count = len(not_verified_fields)
        verification_rate = (verified_count / total_fields * 100) if total_fields > 0 else 0
        
        # Extract project info for report header
        project_id = proposal.get('project_details', {}).get('project_id', 'N/A')
        project_name = proposal.get('project_details', {}).get('project_name', 'N/A')
        
        # Generate summary
        summary = (
            f"Validated {total_fields} fields from project proposal against EIA report. "
            f"{verified_count} fields verified ({verification_rate:.1f}%), "
            f"{not_verified_count} fields could not be verified ({100-verification_rate:.1f}%). "
        )
        
        # Create comprehensive report
        report = {
            "project_id": project_id,
            "project_name": project_name,
            "timestamp": datetime.utcnow().isoformat(),
            "validation_summary": {
                "total_fields_checked": total_fields,
                "verified_count": verified_count,
                "not_verified_count": not_verified_count,
                "verification_rate_percent": round(verification_rate, 2)
            },
            "summary": summary,
            "verified_fields": verified_fields,
            "not_verified_fields": not_verified_fields,
            "all_field_results": field_results
        }
        
        print(f"\n{'='*80}")
        print("Deficiency Detection Complete!")
        print(f"{'='*80}")
        print(f"Total Fields: {total_fields}")
        print(f"Verified: {verified_count} ({verification_rate:.1f}%)")
        print(f"Not Verified: {not_verified_count} ({100-verification_rate:.1f}%)")
        print(f"{'='*80}\n")
        
        return report


