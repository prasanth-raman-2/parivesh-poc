"""
Test script for Deficiency Detection API
"""

import requests
import json
from pathlib import Path


def test_deficiency_detection_from_file():
    """Test deficiency detection by uploading project_proposal.json file."""
    
    # API endpoint
    url = "http://localhost:8001/deficiency/detect-from-file"
    
    # Path to project proposal file
    file_path = Path("/home/prasa/projects/negd/parivesh-poc/app/data/project_proposal.json")
    
    # Parameters
    params = {
        "include_low_severity": True,
        "top_k_rag_results": 3
    }
    
    # Upload file
    with open(file_path, 'rb') as f:
        files = {'file': ('project_proposal.json', f, 'application/json')}
        
        print("Sending request to deficiency detection API...")
        print(f"URL: {url}")
        print(f"Params: {params}")
        print("-" * 80)
        
        response = requests.post(url, files=files, params=params)
    
    if response.status_code == 200:
        print("✓ Success!")
        print("-" * 80)
        
        result = response.json()
        
        # Print summary
        print("\n" + "="*80)
        print("DEFICIENCY DETECTION REPORT - ALL FIELDS VALIDATED")
        print("="*80)
        print(f"\nProject ID: {result['project_id']}")
        print(f"Project Name: {result['project_name']}")
        print(f"Timestamp: {result['timestamp']}")
        
        validation_summary = result['validation_summary']
        print(f"\n{'='*80}")
        print("VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Fields Checked: {validation_summary['total_fields_checked']}")
        print(f"Verified Fields: {validation_summary['verified_count']} ✓")
        print(f"Not Verified Fields: {validation_summary['not_verified_count']} ✗")
        print(f"Verification Rate: {validation_summary['verification_rate_percent']}%")
        
        print(f"\n{result['summary']}")
        
        # Print verified fields
        print(f"\n{'='*80}")
        print(f"VERIFIED FIELDS ({len(result['verified_fields'])})")
        print(f"{'='*80}")
        
        for field in result['verified_fields'][:10]:  # Show first 10
            print(f"\n✓ {field['field_name']}")
            print(f"   Path: {field['field_path']}")
            print(f"   Proposal Value: {field['proposal_value']}")
            print(f"   EIA Value: {field['eia_value']}")
            print(f"   Confidence: {field['confidence']:.2f}")
            if field.get('rag_reference'):
                print(f"   RAG Reference: {field['rag_reference'][:150]}...")
        
        if len(result['verified_fields']) > 10:
            print(f"\n... and {len(result['verified_fields']) - 10} more verified fields")
        
        # Print not verified fields
        print(f"\n{'='*80}")
        print(f"NOT VERIFIED FIELDS ({len(result['not_verified_fields'])})")
        print(f"{'='*80}")
        
        for field in result['not_verified_fields']:
            print(f"\n✗ {field['field_name']}")
            print(f"   Path: {field['field_path']}")
            print(f"   Proposal Value: {field['proposal_value']}")
            print(f"   EIA Value: {field['eia_value']}")
            print(f"   Confidence: {field['confidence']:.2f}")
            if field.get('rag_reference'):
                print(f"   RAG Reference: {field['rag_reference'][:150]}...")
        
        # Save full report
        output_file = "deficiency_report_full.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✓ Full report saved to: {output_file}")
    
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)


def test_health_check():
    """Test health check endpoint."""
    url = "http://localhost:8001/deficiency/health"
    
    print("\nChecking deficiency detection service health...")
    response = requests.get(url)
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Service is healthy")
        print(f"  Milvus Connected: {result['milvus_connected']}")
        print(f"  Knowledge Base Chunks: {result['knowledge_base_chunks']}")
        print(f"  Collection Name: {result['collection_name']}")
    else:
        print(f"✗ Service unhealthy: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    print("="*80)
    print("DEFICIENCY DETECTION API TEST")
    print("="*80)
    
    # Test health check first
    test_health_check()
    
    print("\n" + "="*80)
    
    # Test deficiency detection
    test_deficiency_detection_from_file()
