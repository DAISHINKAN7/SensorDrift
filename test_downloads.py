#!/usr/bin/env python3
"""
Test script for download functionality
"""

import requests
import json
import time
import os

def test_downloads():
    base_url = "http://localhost:5001"
    
    print("🧪 Testing Download Functionality")
    print("=" * 50)
    
    # Test CSV export
    print("\n📊 Testing CSV Export...")
    try:
        response = requests.post(f"{base_url}/api/export_csv", 
                               json={"data_source": "real"})
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ CSV Export successful: {len(data['csv_files'])} files generated")
                for filename in data['csv_files']:
                    print(f"   - {filename}")
            else:
                print(f"❌ CSV Export failed: {data.get('error')}")
        else:
            print(f"❌ CSV Export request failed: {response.status_code}")
    except Exception as e:
        print(f"❌ CSV Export error: {e}")
    
    # Test JSON export
    print("\n📄 Testing JSON Export...")
    try:
        response = requests.post(f"{base_url}/api/export_json", 
                               json={"data_source": "real"})
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ JSON Export successful: {data['json_file']}")
            else:
                print(f"❌ JSON Export failed: {data.get('error')}")
        else:
            print(f"❌ JSON Export request failed: {response.status_code}")
    except Exception as e:
        print(f"❌ JSON Export error: {e}")
    
    # Test PDF generation
    print("\n📋 Testing PDF Report Generation...")
    try:
        response = requests.post(f"{base_url}/api/generate_pdf_report", 
                               json={"data_source": "real"})
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ PDF Report generated: {data['filename']}")
            else:
                print(f"❌ PDF Generation failed: {data.get('error')}")
        else:
            print(f"❌ PDF Generation request failed: {response.status_code}")
    except Exception as e:
        print(f"❌ PDF Generation error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Test completed! Start the app with: python app.py")
    print("🌐 Then visit: http://localhost:5001")
    print("📥 Click the download buttons in the Analytics tab to test!")

if __name__ == "__main__":
    print("⚠️  Make sure the Flask app is running first!")
    print("   Run: python app.py")
    print("   Then run this test script in another terminal")
    
    input("\nPress Enter when the app is running...")
    test_downloads()
