"""
Utility script to view and explore data stored in Milvus.
"""

from app.milvus import MilvusClient
from app.core.settings import settings
from pymilvus import Collection
import json


def view_all_data():
    """View all data stored in the Milvus collection."""
    
    # Initialize client
    client = MilvusClient(
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
        collection_name=settings.MILVUS_COLLECTION_NAME,
        dim=settings.EMBEDDING_DIMENSION
    )
    
    try:
        # Connect
        client.connect()
        print(f"Connected to Milvus at {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        
        # Check if collection exists
        from pymilvus import utility
        if not utility.has_collection(settings.MILVUS_COLLECTION_NAME):
            print(f"\nCollection '{settings.MILVUS_COLLECTION_NAME}' does not exist!")
            return
        
        # Get collection
        collection = Collection(settings.MILVUS_COLLECTION_NAME)
        
        # Get collection stats
        stats = client.get_collection_stats()
        print("\n" + "="*80)
        print("COLLECTION STATISTICS")
        print("="*80)
        print(f"Collection Name: {stats['name']}")
        print(f"Total Entities: {stats['num_entities']}")
        print(f"Description: {stats['description']}")
        
        if stats['num_entities'] == 0:
            print("\nNo data in collection!")
            return
        
        # Load collection
        collection.load()
        print("\n" + "="*80)
        print("COLLECTION DATA")
        print("="*80)
        
        # Query all data (limit to first 100 for display)
        results = collection.query(
            expr="id >= 0",
            output_fields=["id", "text", "metadata"],
            limit=100
        )
        
        print(f"\nDisplaying {len(results)} entities (max 100):\n")
        
        for i, entity in enumerate(results, 1):
            print(f"\n{'─'*80}")
            print(f"Entity {i}:")
            print(f"{'─'*80}")
            print(f"ID: {entity['id']}")
            print(f"Text: {entity['text'][:200]}{'...' if len(entity['text']) > 200 else ''}")
            print(f"Metadata: {json.dumps(entity['metadata'], indent=2)}")
        
        # Release collection
        collection.release()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        client.disconnect()
        print("\n\nDisconnected from Milvus")


def search_by_text(search_text: str, top_k: int = 5):
    """Search for similar texts in Milvus."""
    
    from app.ingestion.document_ingestion import DocumentIngestion
    
    doc_ingest = DocumentIngestion(use_milvus=True)
    
    try:
        doc_ingest.connect_milvus()
        print(f"\nSearching for: '{search_text}'")
        print("="*80)
        
        results = doc_ingest.search_similar(search_text, top_k=top_k)
        
        print(f"\nFound {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            print(f"\n{'─'*80}")
            print(f"Result {i}:")
            print(f"{'─'*80}")
            print(f"Distance: {result['distance']:.4f}")
            print(f"ID: {result['id']}")
            print(f"Text: {result['text']}")
            print(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
        
    finally:
        doc_ingest.disconnect_milvus()


def get_entity_by_id(entity_id: int):
    """Get a specific entity by ID."""
    
    client = MilvusClient(
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
        collection_name=settings.MILVUS_COLLECTION_NAME,
        dim=settings.EMBEDDING_DIMENSION
    )
    
    try:
        client.connect()
        collection = Collection(settings.MILVUS_COLLECTION_NAME)
        collection.load()
        
        results = collection.query(
            expr=f"id == {entity_id}",
            output_fields=["id", "text", "metadata"]
        )
        
        if results:
            entity = results[0]
            print("\n" + "="*80)
            print(f"ENTITY ID: {entity_id}")
            print("="*80)
            print(f"ID: {entity['id']}")
            print(f"Text: {entity['text']}")
            print(f"Metadata: {json.dumps(entity['metadata'], indent=2)}")
        else:
            print(f"\nNo entity found with ID: {entity_id}")
        
        collection.release()
        
    finally:
        client.disconnect()


def delete_all_data():
    """Delete all data from the collection."""
    
    client = MilvusClient(
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
        collection_name=settings.MILVUS_COLLECTION_NAME,
        dim=settings.EMBEDDING_DIMENSION
    )
    
    try:
        client.connect()
        
        from pymilvus import utility
        if utility.has_collection(settings.MILVUS_COLLECTION_NAME):
            confirm = input(f"\nAre you sure you want to delete all data from '{settings.MILVUS_COLLECTION_NAME}'? (yes/no): ")
            if confirm.lower() == 'yes':
                utility.drop_collection(settings.MILVUS_COLLECTION_NAME)
                print(f"\nCollection '{settings.MILVUS_COLLECTION_NAME}' has been deleted!")
            else:
                print("\nDeletion cancelled.")
        else:
            print(f"\nCollection '{settings.MILVUS_COLLECTION_NAME}' does not exist!")
        
    finally:
        client.disconnect()


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("MILVUS DATA VIEWER")
    print("="*80)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "view":
            view_all_data()
        
        elif command == "search":
            if len(sys.argv) > 2:
                search_text = " ".join(sys.argv[2:])
                top_k = 5
                search_by_text(search_text, top_k)
            else:
                print("\nUsage: python -m app.milvus.view_data search <search_text>")
        
        elif command == "get":
            if len(sys.argv) > 2:
                entity_id = int(sys.argv[2])
                get_entity_by_id(entity_id)
            else:
                print("\nUsage: python -m app.milvus.view_data get <entity_id>")
        
        elif command == "delete":
            delete_all_data()
        
        else:
            print(f"\nUnknown command: {command}")
            print("\nAvailable commands:")
            print("  view          - View all data in the collection")
            print("  search <text> - Search for similar texts")
            print("  get <id>      - Get entity by ID")
            print("  delete        - Delete all data from collection")
    
    else:
        # Default: view all data
        view_all_data()
