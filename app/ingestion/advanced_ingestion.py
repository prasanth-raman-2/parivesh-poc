"""
Advanced Document Ingestion with Rich Metadata and Keyword Extraction
for Agentic RAG System
"""

import re
from typing import List, Dict, Any, Tuple, Set
from pathlib import Path
from collections import Counter
import json

from app.ingestion.document_ingestion import DocumentIngestion
from app.milvus import MilvusClient
from app.core.settings import settings

# NLP libraries for better keyword extraction
try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import nltk
    
    # Download required NLTK data if not available
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    NLTK_AVAILABLE = True
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except ImportError:
    NLTK_AVAILABLE = False
    # Fallback stopwords if NLTK not available
    ENGLISH_STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'it', 'its', 'they', 'them', 'their', 'there',
        'where', 'when', 'which', 'who', 'what', 'how', 'all', 'each',
        'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
        'than', 'too', 'very', 'just', 'also', 'only', 'own', 'same', 'so'
    }


class AdvancedDocumentIngestion:
    def __init__(self):
        self.doc_ingest = DocumentIngestion(use_milvus=True)
        self.milvus_client = MilvusClient(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            collection_name=settings.MILVUS_COLLECTION_NAME,
            dim=settings.EMBEDDING_DIMENSION
        )
        
        # Use NLTK stopwords or fallback
        self.stop_words = ENGLISH_STOPWORDS
        print(f"✓ Using {'NLTK' if NLTK_AVAILABLE else 'fallback'} stopwords ({len(self.stop_words)} words)")
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using NLTK if available, otherwise simple regex."""
        if NLTK_AVAILABLE:
            return word_tokenize(text)
        else:
            return re.findall(r'\b[a-zA-Z]+\b', text)
    
    def extract_page_number(self, text_chunk: str) -> int:
        """Extract page number from text chunk."""
        # Look for patterns like "Page | 23" or "Page 23"
        page_patterns = [
            r'Page\s*\|\s*(\d+)',
            r'Page\s+(\d+)',
            r'page\s+(\d+)',
        ]
        
        for pattern in page_patterns:
            match = re.search(pattern, text_chunk, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return 0  # Unknown page
    
    def extract_section_info(self, text_chunk: str) -> Dict[str, Any]:
        """Extract section number and title from text."""
        # Pattern for section numbers like "1.0", "2.3.4", "3.14.5"
        section_pattern = r'^\s*(\d+(?:\.\d+)*)\s+(.+?)(?:\s*\.{2,}|\n|$)'
        
        section_matches = re.findall(section_pattern, text_chunk, re.MULTILINE)
        
        if section_matches:
            section_number, section_title = section_matches[0]
            return {
                'section_number': section_number,
                'section_title': section_title.strip(),
                'has_section': True
            }
        
        return {
            'section_number': None,
            'section_title': None,
            'has_section': False
        }
    
    def extract_keywords(self, text: str, top_n: int = 15) -> List[str]:
        """
        Extract important keywords dynamically from text chunk.
        Uses NLTK for better tokenization and stopword filtering.
        Combines multiple strategies: capitalized terms, technical terms,
        multi-word phrases, and frequent important words.
        """
        keywords = set()
        
        # 1. Extract capitalized multi-word terms (proper nouns, technical terms)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for term in capitalized_terms:
            if len(term) > 3 and term.lower() not in self.stop_words:
                keywords.add(term)
        
        # 2. Extract acronyms and abbreviations
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        keywords.update(acronyms)
        
        # 3. Extract technical terms with numbers and units
        technical_patterns = [
            r'\b\d+(?:\.\d+)?\s*(?:TPD|MW|KW|Ha|hectare|m³|km|m|km²|mg|kg|MT|KL|ppm|°C|%)\b',
            r'\b(?:carbon black|environmental|pollution|emission|baseline|mitigation|compliance|clearance)\b',
            r'\b(?:PCBL|SIPCOT|Tamil Nadu|Tiruvallur|Thervoy Kandigai|Gummidipoondi)\b'
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.strip():
                    keywords.add(match.strip())
        
        # 4. Tokenize text using NLTK or fallback
        words = self.tokenize_text(text)
        
        # Extract multi-word technical phrases (2-3 words)
        for i in range(len(words) - 2):
            word1_lower = words[i].lower()
            word2_lower = words[i+1].lower()
            word3_lower = words[i+2].lower()
            
            # Two-word phrases
            if (len(words[i]) > 3 and len(words[i+1]) > 3 and
                word1_lower not in self.stop_words and 
                word2_lower not in self.stop_words):
                two_word = f"{words[i]} {words[i+1]}"
                if len(two_word) > 8:
                    keywords.add(two_word.lower())
            
            # Three-word phrases
            if (len(words[i]) > 3 and len(words[i+1]) > 3 and len(words[i+2]) > 3 and
                word1_lower not in self.stop_words and
                word2_lower not in self.stop_words and
                word3_lower not in self.stop_words):
                three_word = f"{words[i]} {words[i+1]} {words[i+2]}"
                if len(three_word) > 12:
                    keywords.add(three_word.lower())
        
        # 5. Extract important single words (filter by frequency and length)
        words_lower = [w.lower() for w in words if len(w) > 4 and w.isalpha()]
        word_freq = Counter(words_lower)
        
        # Keep words that appear more than once and aren't stop words
        for word, count in word_freq.items():
            if count >= 2 and word not in self.stop_words and len(word) > 5:
                keywords.add(word)
        
        # 6. Extract specific domain terms
        domain_patterns = [
            r'\b(?:impact assessment|environmental clearance|project proponent|baseline data)\b',
            r'\b(?:air quality|water quality|noise level|soil quality)\b',
            r'\b(?:manufacturing|production|capacity|expansion|operation)\b',
            r'\b(?:monitoring|sampling|analysis|measurement)\b',
            r'\b(?:village|district|taluk|state|location|coordinates)\b'
        ]
        
        for pattern in domain_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.update([m.lower() for m in matches])
        
        # Convert to list and prioritize by relevance
        keyword_list = list(keywords)
        
        # Score keywords by multiple factors
        keyword_scores = []
        text_lower = text.lower()
        
        for kw in keyword_list:
            score = 0
            kw_lower = kw.lower()
            
            # Frequency in text
            freq = text_lower.count(kw_lower)
            score += freq * 2
            
            # Length bonus (longer terms often more specific)
            if len(kw) > 10:
                score += 3
            elif len(kw) > 6:
                score += 2
            
            # Capitalized terms bonus (likely proper nouns/important)
            if kw[0].isupper():
                score += 2
            
            # Acronym bonus
            if kw.isupper() and len(kw) > 1:
                score += 3
            
            # Contains numbers (technical specs)
            if re.search(r'\d', kw):
                score += 2
            
            keyword_scores.append((kw, score))
        
        # Sort by score and return top N
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [kw for kw, _ in keyword_scores[:top_n]]
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities like locations, organizations, numbers."""
        entities = {
            'locations': [],
            'organizations': [],
            'numbers': [],
            'dates': []
        }
        
        # Extract location patterns (villages, districts, states)
        location_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Village|District|Taluk|State)',
            r'\bTamil Nadu\b',
            r'\bThervoy Kandigai\b',
            r'\bTiruvallur\b',
            r'\bGummidipoondi\b'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            entities['locations'].extend(matches)
        
        # Extract organization names
        org_patterns = [
            r'\b(PCBL\s*\(TN\)\s*Limited)\b',
            r'\b([A-Z][A-Za-z]+\s+Labs\s+India\s+Pvt\s+Ltd)\b',
            r'\bSIPCOT\b'
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, text)
            entities['organizations'].extend(matches)
        
        # Extract numbers with units (TPD, MW, Ha, etc.)
        number_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*(?:TPD|MW|Ha|KL|MT|m³|km²)',
            r'\b(\d+(?:\.\d+)?)\s*hectare',
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['numbers'].extend(matches)
        
        # Extract dates
        date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['dates'].extend(matches)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def identify_chunk_type(self, text: str) -> str:
        """Identify the type of content in the chunk."""
        text_lower = text.lower()
        
        if 'table' in text_lower and '|' in text:
            return 'table'
        elif re.search(r'^\s*\d+\.\d+', text, re.MULTILINE):
            return 'section_content'
        elif 'executive summary' in text_lower:
            return 'executive_summary'
        elif 'table of contents' in text_lower:
            return 'toc'
        elif any(term in text_lower for term in ['baseline', 'monitoring', 'measurement']):
            return 'baseline_data'
        elif any(term in text_lower for term in ['impact', 'mitigation', 'assessment']):
            return 'impact_assessment'
        elif any(term in text_lower for term in ['clearance', 'compliance', 'statutory']):
            return 'compliance'
        else:
            return 'general'
    
    def intelligent_chunk(self, text: str, chunk_size: int = 4000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Intelligently chunk document while preserving section boundaries.
        Returns chunks with rich metadata.
        """
        chunks = []
        
        # Split by major sections first
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        current_page = 0
        current_section = {'section_number': None, 'section_title': None}
        chunk_index = 0
        
        for i, line in enumerate(lines):
            # Check for page markers
            page_num = self.extract_page_number(line)
            if page_num > 0:
                current_page = page_num
            
            # Check for section headers
            section_info = self.extract_section_info(line)
            if section_info['has_section']:
                # If we have a substantial chunk, save it before starting new section
                if current_size > chunk_size // 2:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append(self._create_chunk_metadata(
                        chunk_text, chunk_index, current_page, current_section
                    ))
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    overlap_lines = current_chunk[-5:] if len(current_chunk) > 5 else current_chunk
                    current_chunk = overlap_lines
                    current_size = sum(len(l) for l in overlap_lines)
                
                current_section = {
                    'section_number': section_info['section_number'],
                    'section_title': section_info['section_title']
                }
            
            # Add line to current chunk
            current_chunk.append(line)
            current_size += len(line)
            
            # If chunk is large enough, split it
            if current_size >= chunk_size:
                chunk_text = '\n'.join(current_chunk)
                chunks.append(self._create_chunk_metadata(
                    chunk_text, chunk_index, current_page, current_section
                ))
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_lines = current_chunk[-5:] if len(current_chunk) > 5 else []
                current_chunk = overlap_lines
                current_size = sum(len(l) for l in overlap_lines)
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(self._create_chunk_metadata(
                chunk_text, chunk_index, current_page, current_section
            ))
        
        return chunks
    
    def _create_chunk_metadata(
        self,
        text: str,
        chunk_index: int,
        page_number: int,
        section_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create rich metadata for a chunk."""
        
        keywords = self.extract_keywords(text)
        entities = self.extract_entities(text)
        chunk_type = self.identify_chunk_type(text)
        
        metadata = {
            'chunk_index': chunk_index,
            'page_number': page_number,
            'section_number': section_info.get('section_number'),
            'section_title': section_info.get('section_title'),
            'keywords': keywords,
            'entities': entities,
            'chunk_type': chunk_type,
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
    
    def ingest_document(self, file_path: str, clear_existing: bool = True) -> Dict[str, Any]:
        """
        Main ingestion pipeline for EIA document.
        
        Args:
            file_path: Path to the markdown document
            clear_existing: If True, clear existing Milvus data first
            
        Returns:
            Ingestion statistics
        """
        print("\n" + "="*80)
        print("ADVANCED DOCUMENT INGESTION FOR AGENTIC RAG")
        print("="*80)
        
        # Connect to Milvus
        print("\n[1/6] Connecting to Milvus...")
        self.doc_ingest.connect_milvus()
        
        # Clear existing data if requested
        if clear_existing:
            print("\n[2/6] Clearing existing Milvus data...")
            from pymilvus import utility
            if utility.has_collection(settings.MILVUS_COLLECTION_NAME):
                utility.drop_collection(settings.MILVUS_COLLECTION_NAME)
                print("✓ Cleared existing collection")
                # Recreate collection
                self.milvus_client.create_collection(drop_existing=False)
        else:
            print("\n[2/6] Keeping existing data...")
        
        # Read document
        print("\n[3/6] Reading document...")
        with open(file_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
        
        print(f"✓ Loaded document: {len(document_text)} characters, {len(document_text.split())} words")
        
        # Intelligent chunking
        print("\n[4/6] Creating intelligent chunks with metadata...")
        chunks_with_metadata = self.intelligent_chunk(document_text, chunk_size=4000, overlap=200)
        print(f"✓ Created {len(chunks_with_metadata)} chunks")
        
        # Generate embeddings
        print("\n[5/6] Generating embeddings...")
        texts = [chunk['text'] for chunk in chunks_with_metadata]
        embeddings = self.doc_ingest.get_embeddings(texts)
        print(f"✓ Generated {len(embeddings)} embeddings")
        
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


def main():
    """Main entry point."""
    import sys
    
    # Default document path
    doc_path = "/home/prasa/projects/negd/parivesh-poc/app/data/document.md"
    
    if len(sys.argv) > 1:
        doc_path = sys.argv[1]
    
    clear_existing = True
    if len(sys.argv) > 2 and sys.argv[2].lower() == 'keep':
        clear_existing = False
    
    # Run ingestion
    ingester = AdvancedDocumentIngestion()
    stats = ingester.ingest_document(doc_path, clear_existing=clear_existing)
    
    # Save stats to file
    stats_file = "ingestion_stats.json"
    # Convert Counter objects to dicts for JSON serialization
    stats['top_keywords'] = [{'keyword': k, 'count': v} for k, v in stats['top_keywords']]
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
