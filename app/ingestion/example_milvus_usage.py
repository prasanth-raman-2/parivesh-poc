"""
Example script demonstrating how to use DocumentIngestion with Milvus.

Before running this script, make sure:
1. Milvus server is running (docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest)
2. pymilvus is installed: pip install pymilvus
3. Environment variables are set in .env file or update settings.py defaults
"""

from app.ingestion.document_ingestion import DocumentIngestion
from app.core.settings import settings


def main():
    # Sample document text
    sample_text = """
    Artificial Intelligence (AI) is revolutionizing various industries by automating complex tasks 
    and providing insights from data. Machine learning, a subset of AI, enables systems to learn 
    from experience without being explicitly programmed. Deep learning, which uses neural networks 
    with multiple layers, has achieved remarkable success in image recognition, natural language 
    processing, and game playing.
    
    Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, 
    and manipulate human language. Modern NLP systems use transformer architectures like BERT and GPT 
    to achieve state-of-the-art results in tasks such as translation, summarization, and question answering.
    
    Computer vision enables machines to interpret and understand visual information from the world. 
    Applications include facial recognition, autonomous vehicles, medical image analysis, and 
    quality control in manufacturing.
    """
    
    # Initialize document ingestion with Milvus enabled
    doc_ingestion = DocumentIngestion(use_milvus=True)
    
    try:
        # Connect to Milvus
        print("=" * 60)
        print("Step 1: Connecting to Milvus...")
        print("=" * 60)
        doc_ingestion.connect_milvus()
        
        # Ingest and store document
        print("\n" + "=" * 60)
        print("Step 2: Ingesting document and storing to Milvus...")
        print("=" * 60)
        result = doc_ingestion.ingest_and_store(
            text=sample_text,
            metadata={
                "source": "example_document",
                "topic": "AI and Machine Learning",
                "date": "2024-01-01"
            }
        )
        
        print("\nIngestion Results:")
        print(f"  - Number of chunks: {result['num_chunks']}")
        print(f"  - Number of embeddings: {result['num_embeddings']}")
        print(f"  - Milvus enabled: {result['milvus_enabled']}")
        print(f"  - Stored IDs: {result['milvus_ids'][:3]}..." if result['milvus_ids'] else "  - No IDs (Milvus disabled)")
        
        # Search for similar content
        print("\n" + "=" * 60)
        print("Step 3: Searching for similar content...")
        print("=" * 60)
        query = "What is natural language processing?"
        print(f"Query: '{query}'")
        
        search_results = doc_ingestion.search_similar(query, top_k=3)
        
        print(f"\nFound {len(search_results)} similar results:")
        for i, result in enumerate(search_results, 1):
            print(f"\n  Result {i}:")
            print(f"    Distance: {result['distance']:.4f}")
            print(f"    Text: {result['text'][:100]}...")
            print(f"    Metadata: {result['metadata']}")
        
        # Get collection statistics
        print("\n" + "=" * 60)
        print("Step 4: Collection Statistics")
        print("=" * 60)
        stats = doc_ingestion.milvus_client.get_collection_stats()
        print(f"  Collection Name: {stats['name']}")
        print(f"  Total Entities: {stats['num_entities']}")
        print(f"  Description: {stats['description']}")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Disconnect from Milvus
        print("\n" + "=" * 60)
        print("Step 5: Disconnecting from Milvus...")
        print("=" * 60)
        doc_ingestion.disconnect_milvus()
        print("Done!")


if __name__ == "__main__":
    main()
