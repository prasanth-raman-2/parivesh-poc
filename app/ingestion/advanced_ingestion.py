"""
Advanced Document Ingestion with Rich Metadata and Keyword Extraction
for Agentic RAG System
"""

import re
from typing import List, Dict, Any, Tuple, Set
from pathlib import Path
from collections import Counter
import json
import asyncio
import hashlib
import os

from litellm import completion, acompletion
from app.ingestion.document_ingestion import DocumentIngestion
from app.milvus import MilvusClient
from app.core.settings import settings
from app.models.model_catalogue import LLMModels, EmbeddingModels, ModelConfig


class AdvancedDocumentIngestion:
    def __init__(self):
        self.doc_ingest = DocumentIngestion(use_milvus=True)
        self.embedding_model = EmbeddingModels.TEXT_EMBEDDING_3_LARGE.value
        self.llm_model = LLMModels.GPT_5_2.value
        
        # Get dynamic configuration based on embedding model
        embedding_enum = EmbeddingModels.TEXT_EMBEDDING_3_LARGE
        collection_name = ModelConfig.get_collection_name(embedding_enum)
        dimension = ModelConfig.get_embedding_dimension(embedding_enum)
        
        self.milvus_client = MilvusClient(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            collection_name=collection_name,
            dim=dimension
        )
        
        # Cache directory
        self.cache_dir = Path("cache/ingestion")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, file_path: str, chunk_size: int, overlap: int) -> str:
        """Generate cache key based on file path and chunking parameters."""
        # Read file to compute hash
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        cache_key = f"{file_hash}_{chunk_size}_{overlap}"
        return cache_key
    
    def _get_cache_paths(self, cache_key: str) -> Dict[str, Path]:
        """Get cache file paths for metadata and embeddings."""
        return {
            'metadata': self.cache_dir / f"{cache_key}_metadata.json",
            'embeddings': self.cache_dir / f"{cache_key}_embeddings.json"
        }
    
    def _load_from_cache(self, cache_key: str) -> tuple[List[Dict[str, Any]], List[List[float]]]:
        """Load cached metadata and embeddings if available."""
        cache_paths = self._get_cache_paths(cache_key)
        
        if cache_paths['metadata'].exists() and cache_paths['embeddings'].exists():
            print(f"✓ Loading from cache: {cache_key}")
            
            with open(cache_paths['metadata'], 'r') as f:
                chunks_with_metadata = json.load(f)
            
            with open(cache_paths['embeddings'], 'r') as f:
                embeddings = json.load(f)
            
            print(f"✓ Loaded {len(chunks_with_metadata)} chunks and {len(embeddings)} embeddings from cache")
            return chunks_with_metadata, embeddings
        
        return None, None
    
    def _save_to_cache(self, cache_key: str, chunks_with_metadata: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Save metadata and embeddings to cache."""
        cache_paths = self._get_cache_paths(cache_key)
        
        print(f"\n💾 Saving to cache: {cache_key}")
        
        with open(cache_paths['metadata'], 'w') as f:
            json.dump(chunks_with_metadata, f)
        
        with open(cache_paths['embeddings'], 'w') as f:
            json.dump(embeddings, f)
        
        print(f"✓ Cached {len(chunks_with_metadata)} chunks and {len(embeddings)} embeddings")
        print(f"  Metadata: {cache_paths['metadata']}")
        print(f"  Embeddings: {cache_paths['embeddings']}")
    
    async def extract_metadata_async(self, text: str, top_n: int = 15) -> Dict[str, Any]:
        """
        Extract all metadata using AI in a single call (async version).
        Uses LLM to intelligently identify page number, section info, keywords, entities, and chunk type.
        
        Returns:
            Dictionary with page_number, section_number, section_title, keywords, entities, and chunk_type.
        """
        # Truncate text if too long (max 4000 chars for efficient processing)
        text_sample = text[:4000] if len(text) > 4000 else text
        
        prompt = f"""Analyze the following text and extract comprehensive metadata.

Extract:
1. Page number if present (look for patterns like "Page | 23", "Page 23", "page 23")
2. Section number and title if present (e.g., "1.0 Introduction", "2.3.4 Environmental Impact")
3. Chunk type - classify the content as one of: table, section_content, executive_summary, toc, baseline_data, impact_assessment, compliance, general
4. Top {top_n} most important keywords and key phrases (technical terms, domain concepts, specifications)
5. Named entities categorized as:
   - locations: cities, villages, districts, states, countries
   - organizations: company names, institutions, regulatory bodies
   - numbers: numerical values with their units (e.g., "240 TPD", "150 MW", "25.5 Ha")
   - dates: any temporal references (dates, months, years)

Return a JSON object with this exact structure:
{{
  "page_number": 0,
  "section_number": null,
  "section_title": null,
  "chunk_type": "general",
  "keywords": ["keyword1", "keyword2", ...],
  "entities": {{
    "locations": ["location1", "location2", ...],
    "organizations": ["org1", "org2", ...],
    "numbers": ["number1", "number2", ...],
    "dates": ["date1", "date2", ...]
  }}
}}

If page_number is not found, return 0.
If section is not found, return null for section_number and section_title.

Text:
{text_sample}

Respond with only the JSON object, no explanation:"""
        
        try:
            response = await acompletion(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            result = json.loads(response.choices[0].message.content)
            
            # Extract page number
            page_number = result.get('page_number', 0)
            if not isinstance(page_number, int):
                try:
                    page_number = int(page_number)
                except:
                    page_number = 0
            
            # Extract section info
            section_number = result.get('section_number')
            section_title = result.get('section_title')
            
            # Extract chunk type
            chunk_type = result.get('chunk_type', 'general')
            valid_types = ['table', 'section_content', 'executive_summary', 'toc', 'baseline_data', 'impact_assessment', 'compliance', 'general']
            if chunk_type not in valid_types:
                chunk_type = 'general'
            
            # Extract keywords
            keywords = result.get('keywords', [])
            if not isinstance(keywords, list):
                keywords = []
            keywords = [str(kw).strip() for kw in keywords if kw][:top_n]
            
            # Extract entities
            entities_raw = result.get('entities', {})
            entities = {
                'locations': [str(e).strip() for e in entities_raw.get('locations', []) if e],
                'organizations': [str(e).strip() for e in entities_raw.get('organizations', []) if e],
                'numbers': [str(e).strip() for e in entities_raw.get('numbers', []) if e],
                'dates': [str(e).strip() for e in entities_raw.get('dates', []) if e]
            }
            
            return {
                'page_number': page_number,
                'section_number': section_number,
                'section_title': section_title,
                'chunk_type': chunk_type,
                'keywords': keywords,
                'entities': entities
            }
            
        except Exception as e:
            print(f"Warning: AI extraction failed: {e}. Using fallback.")
            # Simple fallback with regex
            page_number = 0
            page_patterns = [r'Page\s*\|\s*(\d+)', r'Page\s+(\d+)', r'page\s+(\d+)']
            for pattern in page_patterns:
                match = re.search(pattern, text_sample, re.IGNORECASE)
                if match:
                    page_number = int(match.group(1))
                    break
            
            section_number = None
            section_title = None
            section_pattern = r'^\s*(\d+(?:\.\d+)*)\s+(.+?)(?:\s*\.{2,}|\n|$)'
            section_matches = re.findall(section_pattern, text_sample, re.MULTILINE)
            if section_matches:
                section_number, section_title = section_matches[0]
                section_title = section_title.strip()
            
            # Simple chunk type detection
            text_lower = text_sample.lower()
            if 'table' in text_lower and '|' in text_sample:
                chunk_type = 'table'
            elif 'executive summary' in text_lower:
                chunk_type = 'executive_summary'
            elif 'table of contents' in text_lower:
                chunk_type = 'toc'
            else:
                chunk_type = 'general'
            
            capitalized = re.findall(r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b', text_sample)
            acronyms = re.findall(r'\b[A-Z]{2,}\b', text_sample)
            fallback_keywords = list(set(capitalized + acronyms))[:top_n]
            
            return {
                'page_number': page_number,
                'section_number': section_number,
                'section_title': section_title,
                'chunk_type': chunk_type,
                'keywords': fallback_keywords,
                'entities': {
                    'locations': [],
                    'organizations': [],
                    'numbers': [],
                    'dates': []
                }
            }
    
    async def intelligent_chunk(self, text: str, chunk_size: int = 1500, overlap: int = 150, max_concurrent_ai: int = 3) -> List[Dict[str, Any]]:
        """
        Intelligently chunk document with parallel metadata extraction.
        Returns chunks with rich metadata extracted by AI.
        """
        # First, create basic chunks without metadata
        chunk_texts = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            current_chunk.append(line)
            current_size += len(line)
            
            # If chunk is large enough, split it
            if current_size >= chunk_size:
                chunk_text = '\n'.join(current_chunk)
                chunk_texts.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_lines = current_chunk[-5:] if len(current_chunk) > 5 else []
                current_chunk = overlap_lines
                current_size = sum(len(l) for l in overlap_lines)
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunk_texts.append(chunk_text)
        
        # Use semaphore to limit concurrent AI requests to avoid rate limits
        semaphore = asyncio.Semaphore(max_concurrent_ai)
        
        async def bounded_metadata(text: str, index: int):
            async with semaphore:
                return await self._create_chunk_metadata_async(text, index)
        
        # Process all chunks with metadata in parallel with rate limiting
        tasks = [bounded_metadata(text, i) for i, text in enumerate(chunk_texts)]
        chunks = await asyncio.gather(*tasks)
        return chunks
    
    async def _create_chunk_metadata_async(self, text: str, chunk_index: int) -> Dict[str, Any]:
        """Create rich metadata for a chunk using AI extraction (async)."""
        
        # Extract all metadata using AI
        extraction_result = await self.extract_metadata_async(text)
        
        metadata = {
            'chunk_index': chunk_index,
            'page_number': extraction_result['page_number'],
            'section_number': extraction_result['section_number'],
            'section_title': extraction_result['section_title'],
            'chunk_type': extraction_result['chunk_type'],
            'keywords': extraction_result['keywords'],
            'entities': extraction_result['entities'],
            'char_count': len(text),
            'word_count': len(text.split()),
            'has_tables': '|' in text and 'table' in text.lower(),
            'has_numbers': bool(re.search(r'\d+', text)),
            'source_document': 'EIA_Report_PCBL_TN_Limited.md'
        }
        
        return {
            'text': text,
            'metadata': metadata
        }
    
    async def ingest_document(self, file_path: str, clear_existing: bool = True, use_cache: bool = True) -> Dict[str, Any]:
        """
        Main ingestion pipeline for EIA document.
        
        Args:
            file_path: Path to the markdown document
            clear_existing: If True, clear existing Milvus data first
            use_cache: If True, use cached metadata and embeddings if available
            
        Returns:
            Ingestion statistics
        """
        print("\n" + "="*80)
        print("ADVANCED DOCUMENT INGESTION FOR AGENTIC RAG")
        print("="*80)
        
        # Get chunk configuration
        chunk_size, overlap = ModelConfig.get_chunk_config(EmbeddingModels.TEXT_EMBEDDING_3_LARGE)
        
        # Generate cache key
        cache_key = self._get_cache_key(file_path, chunk_size, overlap)
        
        # Try to load from cache if enabled
        chunks_with_metadata = None
        embeddings = None
        
        if use_cache:
            print(f"\n[Cache] Checking for cached data...")
            chunks_with_metadata, embeddings = self._load_from_cache(cache_key)
        
        # Connect to Milvus
        print("\n[1/6] Connecting to Milvus...")
        self.doc_ingest.connect_milvus()
        
        # Clear existing data if requested
        if clear_existing:
            print("\n[2/6] Clearing existing Milvus data...")
            from pymilvus import utility
            collection_name = ModelConfig.get_collection_name(EmbeddingModels.TEXT_EMBEDDING_3_LARGE)
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                print("✓ Cleared existing collection")
                # Recreate collection
                self.milvus_client.create_collection(drop_existing=False)
        else:
            print("\n[2/6] Keeping existing data...")
        
        # If not cached, process the document
        if chunks_with_metadata is None or embeddings is None:
            # Read document
            print("\n[3/6] Reading document...")
            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
            
            print(f"✓ Loaded document: {len(document_text)} characters, {len(document_text.split())} words")
            
            # Intelligent chunking
            print("\n[4/6] Creating intelligent chunks with metadata...")
            print(f"Using chunk_size={chunk_size}, overlap={overlap} for {EmbeddingModels.TEXT_EMBEDDING_3_LARGE.value}")
            chunks_with_metadata = await self.intelligent_chunk(document_text, chunk_size=chunk_size, overlap=overlap)
            print(f"✓ Created {len(chunks_with_metadata)} chunks")
            
            # Generate embeddings
            print("\n[5/6] Generating embeddings...")
            texts = [chunk['text'] for chunk in chunks_with_metadata]
            embeddings = await self.doc_ingest.get_embeddings_async(texts)
            print(f"✓ Generated {len(embeddings)} embeddings")
            
            # Save to cache
            if use_cache:
                self._save_to_cache(cache_key, chunks_with_metadata, embeddings)
        else:
            print("\n[3-5/6] Skipped (using cache)")
        
        # Store to Milvus
        print("\n[6/6] Storing to Milvus...")
        metadata_list = [chunk['metadata'] for chunk in chunks_with_metadata]
        ids = self.doc_ingest.store_embeddings_to_milvus(texts, embeddings, metadata_list)
        
        # Print statistics
        print("\n" + "="*80)
        print("INGESTION COMPLETE")
        print("="*80)
        
        # Analyze metadata
        chunk_types = [m['chunk_type'] for m in metadata_list]
        chunk_type_counts = Counter(chunk_types)
        
        sections = [m.get('section_number') for m in metadata_list if m.get('section_number')]
        unique_sections = len(set(sections))
        
        pages = [m['page_number'] for m in metadata_list if m['page_number'] > 0]
        page_range = f"{min(pages)} - {max(pages)}" if pages else "Unknown"
        
        all_keywords = []
        for m in metadata_list:
            all_keywords.extend(m.get('keywords', []))
        top_keywords = Counter(all_keywords).most_common(15)
        
        stats = {
            'total_chunks': len(chunks_with_metadata),
            'total_embeddings': len(embeddings),
            'milvus_ids_sample': ids[:5],
            'chunk_types': dict(chunk_type_counts),
            'unique_sections': unique_sections,
            'page_range': page_range,
            'top_keywords': top_keywords,
            'source_document': file_path
        }
        
        print(f"\nTotal Chunks: {stats['total_chunks']}")
        print(f"Embeddings Generated: {stats['total_embeddings']}")
        print(f"Unique Sections: {stats['unique_sections']}")
        print(f"Page Range: {stats['page_range']}")
        print(f"\nChunk Type Distribution:")
        for chunk_type, count in chunk_type_counts.most_common():
            print(f"  {chunk_type:20s}: {count:4d}")
        
        print(f"\nTop 15 Keywords:")
        for keyword, count in top_keywords:
            print(f"  {keyword:20s}: {count:4d}")
        
        print(f"\nSample Milvus IDs: {ids[:5]}")
        
        # Disconnect
        self.doc_ingest.disconnect_milvus()
        
        return stats

