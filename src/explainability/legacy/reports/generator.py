"""HTML report generator for fraud explanation reports."""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import base64
from io import BytesIO

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class ReportGenerator:
    """
    Generate professional HTML reports for fraud detection explanations.

    Reports are designed for non-technical users (analysts, regulators, auditors).
    Include visualizations, clear risk factor descriptions, and regulatory compliance information.
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize report generator.

        Args:
            template_dir: Directory containing HTML templates (default: src/reports/templates)
        """
        if template_dir is None:
            # Default to templates directory relative to this file
            template_dir = os.path.join(
                os.path.dirname(__file__),
                'templates'
            )

        self.template_dir = Path(template_dir)
        self.template_path = self.template_dir / 'report_template.html'

    def generate_html_report(
        self,
        transaction_id: str,
        prediction: float,
        predicted_class: str,
        risk_factors: List[Dict[str, Any]],
        global_importance: Optional[List[Dict[str, Any]]] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        include_visualizations: bool = True
    ) -> str:
        """
        Generate a complete HTML fraud explanation report.

        Args:
            transaction_id: Unique transaction identifier
            prediction: Fraud probability score (0-1)
            predicted_class: 'Fraud' or 'Legitimate'
            risk_factors: List of formatted risk factors (from format_risk_factors)
            global_importance: Optional global feature importance list
            model_metadata: Model information (name, version, training date, etc.)
            additional_info: Optional additional transaction data
            include_visualizations: Whether to include visualization placeholders

        Returns:
            Complete HTML report as string
        """
        # Prepare data
        report_data = self._prepare_report_data(
            transaction_id=transaction_id,
            prediction=prediction,
            predicted_class=predicted_class,
            risk_factors=risk_factors,
            global_importance=global_importance,
            model_metadata=model_metadata or {},
            additional_info=additional_info or {}
        )

        # Generate HTML
        html_content = self._render_html_report(report_data, include_visualizations)

        return html_content

    def _prepare_report_data(
        self,
        transaction_id: str,
        prediction: float,
        predicted_class: str,
        risk_factors: List[Dict[str, Any]],
        global_importance: Optional[List[Dict[str, Any]]],
        model_metadata: Dict[str, Any],
        additional_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare data for HTML rendering."""

        # Risk level
        risk_level = self._get_risk_level(prediction)
        risk_color = self._get_risk_color(risk_level)

        # Current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        return {
            'transaction_id': transaction_id,
            'prediction': prediction,
            'prediction_percent': f"{prediction * 100:.2f}%",
            'predicted_class': predicted_class,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'timestamp': timestamp,
            'risk_factors': risk_factors,
            'global_importance': global_importance or [],
            'model_metadata': {
                'name': model_metadata.get('name', 'Fraud Detection Model'),
                'version': model_metadata.get('version', '1.0'),
                'type': model_metadata.get('type', 'N/A'),
                'training_date': model_metadata.get('training_date', 'N/A'),
                'last_validated': model_metadata.get('last_validated', 'N/A'),
            },
            'additional_info': additional_info,
            'regulatory_compliance': {
                'sr_11_7': 'Model validation and documentation per SR 11-7',
                'eu_ai_act': 'Explainability requirements for high-risk AI systems',
                'documentation': 'This report meets model governance documentation requirements'
            }
        }

    def _render_html_report(
        self,
        data: Dict[str, Any],
        include_visualizations: bool
    ) -> str:
        """Render the HTML report."""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraud Analysis Report - {data['transaction_id']}</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="report-header">
            <h1>Fraud Detection Analysis Report</h1>
            <div class="report-meta">
                <span class="transaction-id">Transaction ID: {data['transaction_id']}</span>
                <span class="timestamp">Generated: {data['timestamp']}</span>
            </div>
        </header>

        <!-- Executive Summary -->
        <section class="summary-section">
            <h2>Executive Summary</h2>
            <div class="prediction-card" style="border-left-color: {data['risk_color']};">
                <div class="prediction-row">
                    <div class="prediction-label">Prediction:</div>
                    <div class="prediction-value">{data['prediction_percent']}</div>
                </div>
                <div class="prediction-row">
                    <div class="prediction-label">Classification:</div>
                    <div class="prediction-class prediction-{data['risk_level'].lower()}">{data['predicted_class']}</div>
                </div>
                <div class="prediction-row">
                    <div class="prediction-label">Risk Level:</div>
                    <div class="risk-level" style="color: {data['risk_color']};">{data['risk_level']}</div>
                </div>
            </div>
        </section>

        <!-- Risk Factors -->
        <section class="risk-factors-section">
            <h2>Key Risk Factors</h2>
            <p class="section-description">
                The following factors most strongly influenced this prediction.
                Factors are listed in order of importance.
            </p>

            <div class="risk-factors-list">
"""

        # Add risk factors
        for i, factor in enumerate(data['risk_factors'], 1):
            impact_class = factor.get('impact_level', 'Unknown').lower().replace(' ', '-')
            html += f"""
                <div class="risk-factor-card impact-{impact_class}">
                    <div class="factor-rank">{i}</div>
                    <div class="factor-content">
                        <h3 class="factor-name">{factor['description']}</h3>
                        <div class="factor-details">
                            <span class="factor-direction">
                                <strong>Direction:</strong> {factor['direction']} fraud risk
                            </span>
                            <span class="factor-score">
                                <strong>Score:</strong> {factor['importance']:.4f}
                            </span>
                            <span class="factor-impact">
                                <strong>Impact:</strong> {factor.get('impact_level', 'N/A')}
                            </span>
                        </div>
                    </div>
                </div>
"""

        html += """
            </div>
        </section>
"""

        # Add global importance if available
        if data['global_importance']:
            html += """
        <!-- Global Feature Importance -->
        <section class="global-importance-section">
            <h2>Global Feature Importance</h2>
            <p class="section-description">
                These features are most important for the model overall,
                across all transactions. This provides context for understanding
                which factors the model typically considers.
            </p>

            <div class="global-importance-list">
"""
            for i, item in enumerate(data['global_importance'][:10], 1):
                if isinstance(item, dict):
                    name = item.get('description', item.get('feature', 'Unknown'))
                    score = item.get('abs_importance', item.get('importance', 0))
                else:
                    name, score = item
                html += f"""
                <div class="global-factor-item">
                    <span class="factor-rank-small">{i}</span>
                    <span class="factor-name-small">{name}</span>
                    <span class="factor-score-small">{score:.4f}</span>
                </div>
"""

            html += """
            </div>
        </section>
"""

        # Add model information
        html += f"""
        <!-- Model Information -->
        <section class="model-info-section">
            <h2>Model Information</h2>
            <div class="info-grid">
                <div class="info-item">
                    <span class="info-label">Model Name:</span>
                    <span class="info-value">{data['model_metadata']['name']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Version:</span>
                    <span class="info-value">{data['model_metadata']['version']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Type:</span>
                    <span class="info-value">{data['model_metadata']['type']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Training Date:</span>
                    <span class="info-value">{data['model_metadata']['training_date']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Last Validated:</span>
                    <span class="info-value">{data['model_metadata']['last_validated']}</span>
                </div>
            </div>
        </section>

        <!-- Regulatory Compliance -->
        <section class="compliance-section">
            <h2>Regulatory Compliance</h2>
            <div class="compliance-notice">
                <p><strong>SR 11-7 Compliance:</strong> {data['regulatory_compliance']['sr_11_7']}</p>
                <p><strong>EU AI Act Compliance:</strong> {data['regulatory_compliance']['eu_ai_act']}</p>
                <p><strong>Documentation:</strong> {data['regulatory_compliance']['documentation']}</p>
            </div>
        </section>

        <!-- Footer -->
        <footer class="report-footer">
            <p>
                This report was generated automatically by the Fraud Model Explainability System.
                <br>
                For questions or concerns, contact the Model Risk Management team.
            </p>
            <p class="disclaimer">
                <strong>Disclaimer:</strong> This report is provided for informational purposes only
                and should be used in conjunction with other fraud detection methods and human expertise.
            </p>
        </footer>
    </div>
</body>
</html>
"""
        return html

    def _get_css_styles(self) -> str:
        """Return CSS styles for the HTML report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }

        .report-header {
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }

        .report-header h1 {
            color: #2c3e50;
            font-size: 28px;
            margin-bottom: 10px;
        }

        .report-meta {
            display: flex;
            justify-content: space-between;
            color: #7f8c8d;
            font-size: 14px;
        }

        section {
            margin-bottom: 40px;
        }

        h2 {
            color: #2c3e50;
            font-size: 22px;
            margin-bottom: 15px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }

        .section-description {
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 20px;
            font-style: italic;
        }

        .prediction-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 8px;
            border-left: 6px solid;
            color: white;
        }

        .prediction-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }

        .prediction-row:last-child {
            border-bottom: none;
        }

        .prediction-label {
            font-size: 16px;
            opacity: 0.9;
        }

        .prediction-value {
            font-size: 32px;
            font-weight: bold;
        }

        .prediction-class {
            font-size: 24px;
            font-weight: bold;
            text-transform: uppercase;
        }

        .prediction-critical { color: #e74c3c; }
        .prediction-high { color: #e67e22; }
        .prediction-medium { color: #f39c12; }
        .prediction-low { color: #27ae60; }
        .prediction-very-low { color: #2980b9; }

        .risk-level {
            font-size: 20px;
            font-weight: bold;
        }

        .risk-factors-list {
            display: grid;
            gap: 15px;
        }

        .risk-factor-card {
            display: flex;
            gap: 15px;
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid;
            background: #f8f9fa;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .risk-factor-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .impact-very-high, .impact-high {
            border-left-color: #e74c3c;
            background: linear-gradient(90deg, #fff5f5 0%, #ffffff 100%);
        }

        .impact-medium {
            border-left-color: #f39c12;
            background: linear-gradient(90deg, #fffbf0 0%, #ffffff 100%);
        }

        .impact-low, .impact-very-low {
            border-left-color: #27ae60;
            background: linear-gradient(90deg, #f0fff4 0%, #ffffff 100%);
        }

        .factor-rank {
            font-size: 32px;
            font-weight: bold;
            color: #bdc3c7;
            min-width: 50px;
            text-align: center;
        }

        .factor-content {
            flex: 1;
        }

        .factor-name {
            font-size: 18px;
            color: #2c3e50;
            margin-bottom: 10px;
        }

        .factor-details {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            font-size: 14px;
            color: #7f8c8d;
        }

        .global-importance-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .global-factor-item {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 4px;
        }

        .factor-rank-small {
            font-weight: bold;
            color: #3498db;
            min-width: 30px;
        }

        .factor-name-small {
            flex: 1;
            font-weight: 500;
        }

        .factor-score-small {
            color: #7f8c8d;
            font-family: monospace;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }

        .info-item {
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
        }

        .info-label {
            display: block;
            font-weight: bold;
            color: #7f8c8d;
            margin-bottom: 5px;
            font-size: 13px;
        }

        .info-value {
            color: #2c3e50;
            font-size: 16px;
        }

        .compliance-notice {
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 20px;
            border-radius: 4px;
        }

        .compliance-notice p {
            margin-bottom: 10px;
        }

        .compliance-notice p:last-child {
            margin-bottom: 0;
        }

        .report-footer {
            border-top: 2px solid #ecf0f1;
            padding-top: 20px;
            margin-top: 40px;
            text-align: center;
            color: #7f8c8d;
            font-size: 14px;
        }

        .disclaimer {
            margin-top: 15px;
            padding: 15px;
            background: #fff3cd;
            border-radius: 4px;
            border-left: 3px solid #ffc107;
        }

        @media print {
            body {
                background: white;
                padding: 0;
            }
            .container {
                box-shadow: none;
                padding: 20px;
            }
            .risk-factor-card:hover {
                transform: none;
                box-shadow: none;
            }
        }

        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }
            .factor-details {
                flex-direction: column;
                gap: 5px;
            }
            .info-grid {
                grid-template-columns: 1fr;
            }
        }
        """

    def _get_risk_level(self, probability: float) -> str:
        """Get risk level from probability."""
        if probability >= 0.8:
            return "Critical"
        elif probability >= 0.6:
            return "High"
        elif probability >= 0.4:
            return "Medium"
        elif probability >= 0.2:
            return "Low"
        else:
            return "Very Low"

    def _get_risk_color(self, risk_level: str) -> str:
        """Get color associated with risk level."""
        colors = {
            "Critical": "#e74c3c",
            "High": "#e67e22",
            "Medium": "#f39c12",
            "Low": "#27ae60",
            "Very Low": "#2980b9"
        }
        return colors.get(risk_level, "#7f8c8d")

    def save_report(self, html_content: str, output_path: str) -> None:
        """
        Save HTML report to file.

        Args:
            html_content: HTML content to save
            output_path: Path to save the report (will create directories if needed)
        """
        output_path = Path(output_path)

        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Report saved to: {output_path.absolute()}")

    def figure_to_base64(self, fig) -> str:
        """
        Convert matplotlib figure to base64 string for embedding in HTML.

        Args:
            fig: matplotlib Figure object

        Returns:
            Base64 encoded string
        """
        if not HAS_MATPLOTLIB:
            return ""

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return img_str
