"""
Test script for Document Ingestion API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8001"  # Adjust based on your server configuration


def test_ingest_document():
    """Test document ingestion endpoint."""
    print("\n=== Testing Document Ingestion ===")
    
    url = f"{BASE_URL}/ingestion/ingest"
    payload = {
        "file_path": "/home/prasa/projects/negd/parivesh-poc/app/data/document.md",
        "clear_existing": True,
        "chunk_size": 4000,
        "overlap": 200
    }
    
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Message: {data['message']}")
        if data.get('stats'):
            stats = data['stats']
            print(f"\nStats:")
            print(f"  Total Chunks: {stats['total_chunks']}")
            print(f"  Total Embeddings: {stats['total_embeddings']}")
            print(f"  Unique Sections: {stats['unique_sections']}")
            print(f"  Page Range: {stats['page_range']}")
            print(f"  Chunk Types: {stats['chunk_types']}")
    else:
        print(f"Error: {response.text}")


def test_list_documents():
    """Test listing documents endpoint."""
    print("\n=== Testing List Documents ===")
    
    url = f"{BASE_URL}/ingestion/documents"
    response = requests.get(url)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Total Documents: {data['total_documents']}")
        for doc in data['documents']:
            print(f"\nDocument: {doc['source_document']}")
            print(f"  Total Chunks: {doc['total_chunks']}")
            print(f"  Page Range: {doc['page_range']}")
            print(f"  Unique Sections: {doc['unique_sections']}")
            print(f"  Chunk Types: {doc['chunk_types']}")
    else:
        print(f"Error: {response.text}")


def test_get_chunks_paginated():
    """Test getting chunks with pagination."""
    print("\n=== Testing Paginated Chunks ===")
    
    url = f"{BASE_URL}/ingestion/chunks"
    params = {
        "page": 1,
        "page_size": 5
    }
    
    response = requests.get(url, params=params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Total: {data['total']}")
        print(f"Page: {data['page']} of {data['total_pages']}")
        print(f"Page Size: {data['page_size']}")
        print(f"\nChunks on this page:")
        
        for chunk in data['chunks']:
            print(f"\n  Chunk ID: {chunk['id']}")
            print(f"  Chunk Index: {chunk['metadata']['chunk_index']}")
            print(f"  Page: {chunk['metadata']['page_number']}")
            print(f"  Section: {chunk['metadata']['section_number']} - {chunk['metadata']['section_title']}")
            print(f"  Type: {chunk['metadata']['chunk_type']}")
            print(f"  Keywords: {', '.join(chunk['metadata']['keywords'][:5])}")
            print(f"  Text preview: {chunk['text'][:100]}...")
    else:
        print(f"Error: {response.text}")


def test_get_chunks_filtered():
    """Test getting chunks with filters."""
    print("\n=== Testing Filtered Chunks ===")
    
    url = f"{BASE_URL}/ingestion/chunks"
    params = {
        "page": 1,
        "page_size": 3,
        "chunk_type": "section_content"
    }
    
    response = requests.get(url, params=params)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Total section_content chunks: {data['total']}")
        print(f"Showing {len(data['chunks'])} chunks")
        
        for chunk in data['chunks']:
            print(f"\n  Chunk ID: {chunk['id']}")
            print(f"  Type: {chunk['metadata']['chunk_type']}")
            print(f"  Section: {chunk['metadata']['section_number']}")
    else:
        print(f"Error: {response.text}")


def test_get_chunk_by_id():
    """Test getting a specific chunk by ID."""
    print("\n=== Testing Get Chunk by ID ===")
    
    # First get a chunk ID from the list
    list_url = f"{BASE_URL}/ingestion/chunks"
    list_response = requests.get(list_url, params={"page": 1, "page_size": 1})
    
    if list_response.status_code == 200:
        chunks = list_response.json()['chunks']
        if chunks:
            chunk_id = chunks[0]['id']
            
            url = f"{BASE_URL}/ingestion/chunks/{chunk_id}"
            response = requests.get(url)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                chunk = response.json()
                print(f"Chunk ID: {chunk['id']}")
                print(f"Metadata: {json.dumps(chunk['metadata'], indent=2)}")
                print(f"Text: {chunk['text'][:200]}...")
            else:
                print(f"Error: {response.text}")
        else:
            print("No chunks available")
    else:
        print("Failed to get chunk list")


if __name__ == "__main__":
    print("Document Ingestion API Test Suite")
    print("=" * 50)
    
    # Run tests
    test_ingest_document()
    test_list_documents()
    test_get_chunks_paginated()
    test_get_chunks_filtered()
    test_get_chunk_by_id()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
