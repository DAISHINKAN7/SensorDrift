#!/usr/bin/env python3
"""
Demo script showing what the download functionality produces
"""

import pandas as pd
import json
from datetime import datetime
from report_generator import ReportGenerator
from app import Config
import os

def create_demo_data():
    """Create sample analytics data for demonstration"""
    return {
        'total_samples': 1000,
        'distribution': [
            {'gas': 'Ammonia', 'count': 167, 'percentage': 16.7, 'color': '#3b82f6', 'danger': 'Toxic', 'threshold': 25, 'risk_score': 2.8},
            {'gas': 'Acetaldehyde', 'count': 166, 'percentage': 16.6, 'color': '#8b5cf6', 'danger': 'Irritant', 'threshold': 50, 'risk_score': 2.7},
            {'gas': 'Acetone', 'count': 167, 'percentage': 16.7, 'color': '#ec4899', 'danger': 'Flammable', 'threshold': 750, 'risk_score': 2.8},
            {'gas': 'Ethanol', 'count': 167, 'percentage': 16.7, 'color': '#10b981', 'danger': 'Flammable', 'threshold': 1000, 'risk_score': 2.8},
            {'gas': 'Ethylene', 'count': 166, 'percentage': 16.6, 'color': '#f59e0b', 'danger': 'Asphyxiant', 'threshold': 100, 'risk_score': 2.7},
            {'gas': 'Toluene', 'count': 167, 'percentage': 16.7, 'color': '#ef4444', 'danger': 'Toxic', 'threshold': 50, 'risk_score': 2.8}
        ],
        'sensor_performance': [
            {'sensor': 'f1', 'mean': 45.2, 'std': 12.3, 'performance_score': 72.8, 'status': 'GOOD'},
            {'sensor': 'f2', 'mean': 38.7, 'std': 9.8, 'performance_score': 74.7, 'status': 'GOOD'},
            {'sensor': 'f3', 'mean': 52.1, 'std': 15.2, 'performance_score': 70.8, 'status': 'GOOD'},
            {'sensor': 'f4', 'mean': 41.3, 'std': 11.7, 'performance_score': 71.7, 'status': 'GOOD'},
            {'sensor': 'f5', 'mean': 48.9, 'std': 13.4, 'performance_score': 72.6, 'status': 'GOOD'},
            {'sensor': 'f6', 'mean': 44.6, 'std': 12.1, 'performance_score': 72.9, 'status': 'GOOD'},
            {'sensor': 'f7', 'mean': 39.8, 'std': 10.2, 'performance_score': 74.4, 'status': 'GOOD'},
            {'sensor': 'f8', 'mean': 46.7, 'std': 13.8, 'performance_score': 70.4, 'status': 'GOOD'}
        ],
        'safety_analysis': [
            {'gas': 'Ammonia', 'avg_concentration': 18.5, 'threshold': 25, 'risk_level': 'LOW', 'danger_type': 'Toxic', 'samples': 167},
            {'gas': 'Acetaldehyde', 'avg_concentration': 35.2, 'threshold': 50, 'risk_level': 'LOW', 'danger_type': 'Irritant', 'samples': 166},
            {'gas': 'Acetone', 'avg_concentration': 520.8, 'threshold': 750, 'risk_level': 'LOW', 'danger_type': 'Flammable', 'samples': 167},
            {'gas': 'Ethanol', 'avg_concentration': 680.4, 'threshold': 1000, 'risk_level': 'LOW', 'danger_type': 'Flammable', 'samples': 167},
            {'gas': 'Ethylene', 'avg_concentration': 72.3, 'threshold': 100, 'risk_level': 'LOW', 'danger_type': 'Asphyxiant', 'samples': 166},
            {'gas': 'Toluene', 'avg_concentration': 38.7, 'threshold': 50, 'risk_level': 'LOW', 'danger_type': 'Toxic', 'samples': 167}
        ],
        'kpis': {
            'total_samples': 1000,
            'total_gases': 6,
            'most_common_gas': 'Ammonia',
            'least_common_gas': 'Acetaldehyde',
            'data_balance_score': 99.4,
            'avg_samples_per_gas': 167,
            'sensor_diversity': 12.5,
            'detection_complexity': 85.5,
            'model_accuracy': 99.91,
            'model_path': '/Users/kshitijnavale/Desktop/sensor data/model/model_optimal_50_50_mix.keras'
        },
        'data_quality': {
            'completeness': 100.0,
            'uniqueness': 98.7,
            'consistency': 98.5,
            'validity': 99.2,
            'outlier_percentage': 2.3
        },
        'model_info': {
            'name': 'Optimal Mix Model',
            'accuracy': 99.91,
            'path': '/Users/kshitijnavale/Desktop/sensor data/model/model_optimal_50_50_mix.keras',
            'description': 'Best performing model using 50-50 mix of real and synthetic data'
        }
    }

def demo_csv_export():
    """Demonstrate CSV export functionality"""
    print("📊 CSV Export Demo")
    print("-" * 30)
    
    analytics_data = create_demo_data()
    report_gen = ReportGenerator(Config)
    
    try:
        csv_files = report_gen.export_csv(analytics_data)
        print(f"✅ Generated {len(csv_files)} CSV files:")
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            print(f"   📄 {filename}")
            
            # Show preview of each CSV
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"      Rows: {len(df)}, Columns: {len(df.columns)}")
                print(f"      Columns: {', '.join(df.columns.tolist())}")
            print()
        
        return csv_files
    except Exception as e:
        print(f"❌ CSV Export failed: {e}")
        return []

def demo_json_export():
    """Demonstrate JSON export functionality"""
    print("📄 JSON Export Demo")
    print("-" * 30)
    
    analytics_data = create_demo_data()
    report_gen = ReportGenerator(Config)
    
    try:
        json_file = report_gen.export_json(analytics_data)
        filename = os.path.basename(json_file)
        print(f"✅ Generated JSON file: {filename}")
        
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            print(f"   📊 Contains {len(data)} main sections:")
            for key in data.keys():
                print(f"      - {key}")
            
            file_size = os.path.getsize(json_file)
            print(f"   📏 File size: {file_size:,} bytes")
        
        return json_file
    except Exception as e:
        print(f"❌ JSON Export failed: {e}")
        return None

def demo_pdf_report():
    """Demonstrate PDF report generation"""
    print("📋 PDF Report Demo")
    print("-" * 30)
    
    analytics_data = create_demo_data()
    report_gen = ReportGenerator(Config)
    
    try:
        pdf_file = report_gen.create_pdf_report(analytics_data)
        filename = os.path.basename(pdf_file)
        print(f"✅ Generated PDF report: {filename}")
        
        if os.path.exists(pdf_file):
            file_size = os.path.getsize(pdf_file)
            print(f"   📏 File size: {file_size:,} bytes")
            print(f"   📋 Contains:")
            print(f"      - Executive Summary")
            print(f"      - Key Performance Indicators")
            print(f"      - Gas Distribution Analysis with charts")
            print(f"      - Sensor Performance Analysis with charts")
            print(f"      - Safety Analysis with charts")
            print(f"      - Data Quality Assessment")
            print(f"      - Recommendations")
        
        return pdf_file
    except Exception as e:
        print(f"❌ PDF Report failed: {e}")
        return None

def main():
    """Run all demos"""
    print("🎯 Gas Monitoring Dashboard - Download Features Demo")
    print("=" * 60)
    print()
    
    # Demo CSV export
    csv_files = demo_csv_export()
    print()
    
    # Demo JSON export
    json_file = demo_json_export()
    print()
    
    # Demo PDF report
    pdf_file = demo_pdf_report()
    print()
    
    # Summary
    print("📁 Generated Files Summary")
    print("-" * 30)
    
    all_files = []
    if csv_files:
        all_files.extend(csv_files)
    if json_file:
        all_files.append(json_file)
    if pdf_file:
        all_files.append(pdf_file)
    
    if all_files:
        print(f"✅ Successfully generated {len(all_files)} files:")
        for file_path in all_files:
            filename = os.path.basename(file_path)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"   📄 {filename} ({size:,} bytes)")
            else:
                print(f"   ❌ {filename} (not found)")
    else:
        print("❌ No files were generated")
    
    print()
    print("🚀 To use in the web app:")
    print("   1. Run: python app.py")
    print("   2. Visit: http://localhost:5001")
    print("   3. Go to Analytics tab")
    print("   4. Click download buttons!")

if __name__ == "__main__":
    main()
