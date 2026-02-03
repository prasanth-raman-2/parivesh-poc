"""
Test script for Deficiency Detection API
"""

import requests
import json
from pathlib import Path


def test_deficiency_detection_from_file():
    """Test deficiency detection by uploading project_proposal.json file."""
    
    # API endpoint
    url = "http://localhost:8000/deficiency/detect-from-file"
    
    # Path to project proposal file
    file_path = Path("/home/prasa/projects/negd/parivesh-poc/app/data/project_proposal.json")
    
    # Parameters
    params = {
        "include_low_severity": True,
        "top_k_rag_results": 5
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
        print("DEFICIENCY DETECTION REPORT")
        print("="*80)
        print(f"\nProject ID: {result['project_id']}")
        print(f"Project Name: {result['project_name']}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"\nTotal Fields Checked: {result['total_fields_checked']}")
        print(f"Total Deficiencies: {result['total_deficiencies']}")
        print(f"\nSeverity Breakdown:")
        print(f"  Critical: {result['critical_count']}")
        print(f"  High: {result['high_count']}")
        print(f"  Medium: {result['medium_count']}")
        print(f"  Low: {result['low_count']}")
        print(f"  Verified: {result['verified_count']}")
        print(f"\nOverall Compliance Score: {result['overall_compliance_score']:.1f}%")
        print(f"\nSummary: {result['summary']}")
        
        # Print deficiencies by category
        print("\n" + "="*80)
        print("DEFICIENCIES BY CATEGORY")
        print("="*80)
        
        for category in result['categories']:
            print(f"\n{category['category_name']}")
            print(f"  Checks: {category['total_checks']} | Deficiencies: {category['deficiencies_found']}")
            print("-" * 80)
            
            for item in category['items']:
                severity_icon = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🟢',
                    'info': '✓'
                }.get(item['severity'], '•')
                
                print(f"\n  {severity_icon} [{item['severity'].upper()}] {item['field_name']}")
                print(f"     Type: {item['deficiency_type']}")
                print(f"     Proposal: {item['proposal_value']}")
                print(f"     EIA: {item['eia_value']}")
                print(f"     Description: {item['description']}")
                if item.get('confidence_score'):
                    print(f"     Confidence: {item['confidence_score']:.2f}")
                if item.get('recommendation'):
                    print(f"     Recommendation: {item['recommendation']}")
                if item.get('eia_reference'):
                    print(f"     EIA Reference: {item['eia_reference'][:150]}...")
        
        # Save full report
        output_file = "deficiency_report.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n✓ Full report saved to: {output_file}")
    
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)


def test_health_check():
    """Test health check endpoint."""
    url = "http://localhost:8000/deficiency/health"
    
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
