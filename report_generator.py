import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import json
import os

class ReportGenerator:
    def __init__(self, config):
        self.config = config
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
        
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        styles = {}
        
        styles['CustomTitle'] = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        styles['CustomHeading'] = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        styles['CustomNormal'] = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6
        )
        
        return styles
    
    def generate_visualization(self, data, chart_type, title, filename):
        """Generate matplotlib visualization and save as image"""
        plt.figure(figsize=(10, 6))
        plt.style.use('default')  # Use default style instead of seaborn
        
        if chart_type == 'distribution':
            gas_names = [item['gas'] for item in data]
            percentages = [item['percentage'] for item in data]
            colors_list = [item['color'] for item in data]
            
            plt.pie(percentages, labels=gas_names, autopct='%1.1f%%', colors=colors_list)
            plt.title(title, fontsize=14, fontweight='bold')
            
        elif chart_type == 'sensor_performance':
            sensors = [item['sensor'] for item in data]
            scores = [item['performance_score'] for item in data]
            
            bars = plt.bar(sensors, scores, color='skyblue', alpha=0.7)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Sensors')
            plt.ylabel('Performance Score')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{score:.1f}', ha='center', va='bottom')
        
        elif chart_type == 'safety_analysis':
            gases = [item['gas'] for item in data]
            concentrations = [item['avg_concentration'] for item in data]
            thresholds = [item['threshold'] for item in data]
            
            x = np.arange(len(gases))
            width = 0.35
            
            plt.bar(x - width/2, concentrations, width, label='Avg Concentration', alpha=0.7)
            plt.bar(x + width/2, thresholds, width, label='Threshold', alpha=0.7)
            
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Gas Types')
            plt.ylabel('Concentration (ppm)')
            plt.xticks(x, gases, rotation=45)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def create_pdf_report(self, analytics_data, predictions_data=None):
        """Generate comprehensive PDF report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gas_monitoring_report_{timestamp}.pdf"
        filepath = os.path.join("/Users/kshitijnavale/Desktop/sensor data", filename)
        
        doc = SimpleDocTemplate(filepath, pagesize=A4, topMargin=0.5*inch)
        story = []
        
        # Title Page
        story.append(Paragraph("Gas Monitoring System", self.custom_styles['CustomTitle']))
        story.append(Paragraph("Comprehensive Analytics Report", self.custom_styles['CustomHeading']))
        story.append(Spacer(1, 0.2*inch))
        
        # Report Info
        report_info = f"""
        <b>Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br/>
        <b>Model:</b> {analytics_data.get('model_info', {}).get('name', 'N/A')}<br/>
        <b>Accuracy:</b> {analytics_data.get('model_info', {}).get('accuracy', 'N/A')}%<br/>
        <b>Total Samples:</b> {analytics_data.get('total_samples', 'N/A')}<br/>
        """
        story.append(Paragraph(report_info, self.custom_styles['CustomNormal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.custom_styles['CustomHeading']))
        
        kpis = analytics_data.get('kpis', {})
        summary_text = f"""
        This report provides a comprehensive analysis of the gas monitoring system performance.
        The system analyzed {kpis.get('total_samples', 0)} samples across {kpis.get('total_gases', 6)} different gas types.
        The model achieved an accuracy of {kpis.get('model_accuracy', 0)}% with a data balance score of {kpis.get('data_balance_score', 0)}%.
        """
        story.append(Paragraph(summary_text, self.custom_styles['CustomNormal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Key Performance Indicators
        story.append(Paragraph("Key Performance Indicators", self.custom_styles['CustomHeading']))
        
        kpi_data = [
            ['Metric', 'Value'],
            ['Total Samples', str(kpis.get('total_samples', 'N/A'))],
            ['Model Accuracy', f"{kpis.get('model_accuracy', 'N/A')}%"],
            ['Data Balance Score', f"{kpis.get('data_balance_score', 'N/A')}%"],
            ['Most Common Gas', kpis.get('most_common_gas', 'N/A')],
            ['Least Common Gas', kpis.get('least_common_gas', 'N/A')],
            ['Sensor Diversity', str(kpis.get('sensor_diversity', 'N/A'))],
            ['Detection Complexity', f"{kpis.get('detection_complexity', 'N/A')}%"]
        ]
        
        kpi_table = Table(kpi_data, colWidths=[3*inch, 2*inch])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(kpi_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Gas Distribution Analysis
        story.append(Paragraph("Gas Distribution Analysis", self.custom_styles['CustomHeading']))
        
        distribution_data = analytics_data.get('distribution', [])
        if distribution_data:
            # Create visualization
            chart_file = self.generate_visualization(
                distribution_data, 'distribution', 
                'Gas Type Distribution', 
                '/tmp/gas_distribution.png'
            )
            
            # Add image to report
            try:
                img = Image(chart_file, width=5*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
            except:
                pass
            
            # Add distribution table
            dist_table_data = [['Gas Type', 'Count', 'Percentage', 'Danger Level', 'Risk Score']]
            for item in distribution_data:
                dist_table_data.append([
                    item['gas'],
                    str(item['count']),
                    f"{item['percentage']}%",
                    item['danger'],
                    str(item.get('risk_score', 'N/A'))
                ])
            
            dist_table = Table(dist_table_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.2*inch, 1*inch])
            dist_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(dist_table)
        
        story.append(PageBreak())
        
        # Sensor Performance Analysis
        story.append(Paragraph("Sensor Performance Analysis", self.custom_styles['CustomHeading']))
        
        sensor_performance = analytics_data.get('sensor_performance', [])
        if sensor_performance:
            # Create visualization
            chart_file = self.generate_visualization(
                sensor_performance, 'sensor_performance',
                'Sensor Performance Scores',
                '/tmp/sensor_performance.png'
            )
            
            try:
                img = Image(chart_file, width=5*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
            except:
                pass
            
            # Add sensor table
            sensor_table_data = [['Sensor', 'Mean', 'Std Dev', 'Performance Score', 'Status']]
            for item in sensor_performance:
                sensor_table_data.append([
                    item['sensor'],
                    str(item['mean']),
                    str(item['std']),
                    f"{item['performance_score']}%",
                    item['status']
                ])
            
            sensor_table = Table(sensor_table_data, colWidths=[1*inch, 1*inch, 1*inch, 1.5*inch, 1.2*inch])
            sensor_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(sensor_table)
        
        story.append(Spacer(1, 0.3*inch))
        
        # Safety Analysis
        story.append(Paragraph("Safety Analysis", self.custom_styles['CustomHeading']))
        
        safety_analysis = analytics_data.get('safety_analysis', [])
        if safety_analysis:
            # Create visualization
            chart_file = self.generate_visualization(
                safety_analysis, 'safety_analysis',
                'Gas Concentrations vs Thresholds',
                '/tmp/safety_analysis.png'
            )
            
            try:
                img = Image(chart_file, width=5*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
            except:
                pass
            
            # Add safety table
            safety_table_data = [['Gas', 'Avg Concentration', 'Threshold', 'Risk Level', 'Samples']]
            for item in safety_analysis:
                safety_table_data.append([
                    item['gas'],
                    f"{item['avg_concentration']} ppm",
                    f"{item['threshold']} ppm",
                    item['risk_level'],
                    str(item['samples'])
                ])
            
            safety_table = Table(safety_table_data, colWidths=[1.2*inch, 1.3*inch, 1.2*inch, 1*inch, 1*inch])
            safety_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(safety_table)
        
        story.append(PageBreak())
        
        # Data Quality Assessment
        story.append(Paragraph("Data Quality Assessment", self.custom_styles['CustomHeading']))
        
        data_quality = analytics_data.get('data_quality', {})
        quality_text = f"""
        <b>Completeness:</b> {data_quality.get('completeness', 'N/A')}%<br/>
        <b>Uniqueness:</b> {data_quality.get('uniqueness', 'N/A')}%<br/>
        <b>Consistency:</b> {data_quality.get('consistency', 'N/A')}%<br/>
        <b>Validity:</b> {data_quality.get('validity', 'N/A')}%<br/>
        <b>Outlier Percentage:</b> {data_quality.get('outlier_percentage', 'N/A')}%<br/>
        """
        story.append(Paragraph(quality_text, self.custom_styles['CustomNormal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        story.append(Paragraph("Recommendations", self.custom_styles['CustomHeading']))
        
        recommendations = self._generate_recommendations(analytics_data)
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", self.custom_styles['CustomNormal']))
        
        # Build PDF
        doc.build(story)
        
        return filepath
    
    def _generate_recommendations(self, analytics_data):
        """Generate recommendations based on analytics data"""
        recommendations = []
        
        # Check data balance
        kpis = analytics_data.get('kpis', {})
        balance_score = kpis.get('data_balance_score', 100)
        
        if balance_score < 50:
            recommendations.append("Consider collecting more data for underrepresented gas types to improve model balance.")
        
        # Check sensor performance
        sensor_performance = analytics_data.get('sensor_performance', [])
        poor_sensors = [s for s in sensor_performance if s.get('performance_score', 100) < 60]
        
        if poor_sensors:
            sensor_names = [s['sensor'] for s in poor_sensors]
            recommendations.append(f"Review and calibrate sensors {', '.join(sensor_names)} as they show suboptimal performance.")
        
        # Check safety analysis
        safety_analysis = analytics_data.get('safety_analysis', [])
        high_risk_gases = [s for s in safety_analysis if s.get('risk_level') == 'HIGH']
        
        if high_risk_gases:
            gas_names = [g['gas'] for g in high_risk_gases]
            recommendations.append(f"Implement enhanced monitoring for {', '.join(gas_names)} due to high risk levels.")
        
        # Data quality recommendations
        data_quality = analytics_data.get('data_quality', {})
        if data_quality.get('outlier_percentage', 0) > 5:
            recommendations.append("High outlier percentage detected. Consider data cleaning and sensor calibration.")
        
        if not recommendations:
            recommendations.append("System performance is optimal. Continue regular monitoring and maintenance.")
        
        return recommendations
    
    def export_csv(self, analytics_data):
        """Export analytics data to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export distribution data
        distribution_df = pd.DataFrame(analytics_data.get('distribution', []))
        dist_filename = f"gas_distribution_{timestamp}.csv"
        dist_filepath = os.path.join("/Users/kshitijnavale/Desktop/sensor data", dist_filename)
        distribution_df.to_csv(dist_filepath, index=False)
        
        # Export sensor performance data
        sensor_df = pd.DataFrame(analytics_data.get('sensor_performance', []))
        sensor_filename = f"sensor_performance_{timestamp}.csv"
        sensor_filepath = os.path.join("/Users/kshitijnavale/Desktop/sensor data", sensor_filename)
        sensor_df.to_csv(sensor_filepath, index=False)
        
        # Export safety analysis data
        safety_df = pd.DataFrame(analytics_data.get('safety_analysis', []))
        safety_filename = f"safety_analysis_{timestamp}.csv"
        safety_filepath = os.path.join("/Users/kshitijnavale/Desktop/sensor data", safety_filename)
        safety_df.to_csv(safety_filepath, index=False)
        
        return [dist_filepath, sensor_filepath, safety_filepath]
    
    def export_json(self, analytics_data):
        """Export complete analytics data to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"complete_analytics_{timestamp}.json"
        filepath = os.path.join("/Users/kshitijnavale/Desktop/sensor data", filename)
        
        # Add metadata
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'report_version': '1.0',
            'analytics_data': analytics_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filepath
