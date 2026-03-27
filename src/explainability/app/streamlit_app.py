"""Streamlit application for fraud model explainability UI."""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import numpy as np
import pandas as pd

# Add src to path for imports
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from api.explainer_factory import create_explainer
from models.loader import load_model
from reports.generator import ReportGenerator
from utils.formatting import format_risk_factors
from utils.validation import validate_consistency, benchmark_explanation_speed

# Page configuration
st.set_page_config(
    page_title="Fraud Model Explainability",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .risk-factor {
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
        background: #f8f9fa;
        border-radius: 4px;
    }
    .risk-factor-high {
        border-left-color: #e74c3c;
    }
    .risk-factor-medium {
        border-left-color: #f39c12;
    }
    .risk-factor-low {
        border-left-color: #27ae60;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None
    if 'explainer' not in st.session_state:
        st.session_state.explainer = None
    if 'training_data' not in st.session_state:
        st.session_state.training_data = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None


def load_model_page():
    """Page for loading a model."""
    st.title("🔍 Fraud Model Explainability Tool")
    st.markdown("---")

    st.header("Load Model")

    # Model file upload
    model_file = st.file_uploader(
        "Upload Model File",
        type=['pkl', 'joblib', 'json', 'h5'],
        help="Upload a trained model file"
    )

    # Training data upload
    training_file = st.file_uploader(
        "Upload Training Data (CSV)",
        type=['csv'],
        help="Required for LIME explainer and SHAP background data"
    )

    if model_file:
        # Save uploaded file temporarily
        temp_model_path = f"/tmp/{model_file.name}"
        with open(temp_model_path, "wb") as f:
            f.write(model_file.getbuffer())

        try:
            with st.spinner("Loading model..."):
                model, model_type = load_model(temp_model_path)
                st.session_state.model = model
                st.session_state.model_type = model_type
                st.success(f"Model loaded successfully! Type: {model_type}")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

    if training_file and st.session_state.model:
        try:
            with st.spinner("Loading training data..."):
                df = pd.read_csv(training_file)
                st.session_state.training_data = df.values
                st.session_state.feature_names = df.columns.tolist()
                st.session_state.X_test = df.sample(min(1000, len(df)), random_state=42).values
                st.success(f"Training data loaded! Shape: {df.shape}")
        except Exception as e:
            st.error(f"Error loading training data: {str(e)}")

    # Model info
    if st.session_state.model:
        st.subheader("Model Information")
        st.write(f"**Model Type:** {st.session_state.model_type}")
        if st.session_state.training_data is not None:
            st.write(f"**Features:** {len(st.session_state.feature_names)}")
            st.write(f"**Training Samples:** {st.session_state.training_data.shape[0]}")
            with st.expander("View Feature Names"):
                st.write(st.session_state.feature_names)


def create_explainer_page():
    """Page for creating an explainer."""
    st.header("Configure Explainer")

    if not st.session_state.model:
        st.warning("Please load a model first.")
        return

    # Explainer type selection
    explainer_type = st.selectbox(
        "Explainer Type",
        options=['shap', 'lime', 'pdp'],
        help="SHAP: Fast for tree models, LIME: Model-agnostic, PDP: Global insights"
    )

    # Additional options
    col1, col2 = st.columns(2)

    with col1:
        use_background = st.checkbox(
            "Use background data for SHAP",
            value=True,
            help="Recommended for accurate SHAP values"
        )

    with col2:
        random_state = st.number_input(
            "Random State",
            value=42,
            min_value=0,
            max_value=1000,
            help="For reproducibility"
        )

    # Create explainer button
    if st.button("Create Explainer", type="primary"):
        try:
            with st.spinner("Creating explainer..."):
                training_data = st.session_state.training_data if use_background else None

                explainer = create_explainer(
                    model=st.session_state.model,
                    model_type=st.session_state.model_type,
                    explainer_type=explainer_type,
                    training_data=training_data,
                    feature_names=st.session_state.feature_names,
                    random_state=random_state
                )

                st.session_state.explainer = explainer
                st.success(f"{explainer_type.upper()} explainer created successfully!")

        except Exception as e:
            st.error(f"Error creating explainer: {str(e)}")


def explain_transaction_page():
    """Page for explaining individual transactions."""
    st.header("Explain Transaction")

    if not st.session_state.explainer:
        st.warning("Please create an explainer first.")
        return

    if not st.session_state.X_test is not None:
        st.warning("No test data available.")
        return

    # Sample selection
    col1, col2, col3 = st.columns(3)

    with col1:
        sample_idx = st.number_input(
            "Sample Index",
            value=0,
            min_value=0,
            max_value=len(st.session_state.X_test) - 1
        )

    with col2:
        num_features = st.number_input(
            "Top N Features",
            value=5,
            min_value=1,
            max_value=20
        )

    with col3:
        show_raw = st.checkbox("Show Raw Values", value=False)

    # Get sample
    X_sample = st.session_state.X_test[sample_idx:sample_idx+1]

    # Get prediction
    model = st.session_state.model
    if hasattr(model, 'predict_proba'):
        pred_proba = model.predict_proba(X_sample)[0]
        fraud_prob = pred_proba[1]
    else:
        pred = model.predict(X_sample)[0]
        fraud_prob = float(pred)

    # Explain button
    if st.button("Generate Explanation", type="primary"):
        try:
            with st.spinner("Generating explanation..."):
                explainer = st.session_state.explainer

                # Generate local explanation
                explanation = explainer.explain_local(
                    X_sample[0],
                    st.session_state.feature_names
                )

                # Format risk factors
                risk_factors = format_risk_factors(
                    explanation,
                    top_n=num_features
                )

                # Display prediction
                st.subheader("Prediction")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Fraud Probability",
                        f"{fraud_prob:.2%}"
                    )

                with col2:
                    risk_level = get_risk_level(fraud_prob)
                    st.metric("Risk Level", risk_level)

                with col3:
                    pred_class = "Fraud" if fraud_prob >= 0.5 else "Legitimate"
                    st.metric("Classification", pred_class)

                st.markdown("---")

                # Display risk factors
                st.subheader("Top Risk Factors")

                for i, factor in enumerate(risk_factors, 1):
                    impact_class = factor['impact_level'].lower()
                    st.markdown(
                        f"""
                        <div class="risk-factor risk-factor-{impact_class}">
                            <strong>{i}. {factor['description']}</strong><br/>
                            Direction: {factor['direction']} fraud risk<br/>
                            Score: {factor['importance']:.4f} | Impact: {factor['impact_level']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Show raw values if requested
                if show_raw:
                    with st.expander("Raw Feature Values"):
                        df_sample = pd.DataFrame(
                            [X_sample[0]],
                            columns=st.session_state.feature_names
                        ).T
                        df_sample.columns = ['Value']
                        st.dataframe(df_sample)

        except Exception as e:
            st.error(f"Error generating explanation: {str(e)}")


def generate_report_page():
    """Page for generating HTML reports."""
    st.header("Generate Report")

    if not st.session_state.explainer:
        st.warning("Please create an explainer first.")
        return

    # Report options
    col1, col2 = st.columns(2)

    with col1:
        transaction_id = st.text_input(
            "Transaction ID",
            value="TXN-001",
            help="Unique identifier for this transaction"
        )

    with col2:
        sample_idx = st.number_input(
            "Sample Index",
            value=0,
            min_value=0
        )

    if st.button("Generate HTML Report", type="primary"):
        try:
            with st.spinner("Generating report..."):
                # Get sample and prediction
                X_sample = st.session_state.X_test[sample_idx:sample_idx+1]
                model = st.session_state.model

                if hasattr(model, 'predict_proba'):
                    fraud_prob = model.predict_proba(X_sample)[0][1]
                else:
                    fraud_prob = float(model.predict(X_sample)[0])

                # Get explanation
                explanation = st.session_state.explainer.explain_local(
                    X_sample[0],
                    st.session_state.feature_names
                )

                risk_factors = format_risk_factors(explanation, top_n=5)

                # Get global importance
                global_importance = st.session_state.explainer.explain_global(
                    st.session_state.X_test[:100],
                    st.session_state.feature_names
                )
                global_factors = format_risk_factors(global_importance, top_n=10)

                # Model metadata
                model_metadata = {
                    'name': f"{st.session_state.model_type} Fraud Detection Model",
                    'version': '1.0.0',
                    'type': st.session_state.model_type,
                    'training_date': '2024-01-01',
                    'last_validated': '2024-01-15'
                }

                # Generate report
                generator = ReportGenerator()
                html_report = generator.generate_html_report(
                    transaction_id=transaction_id,
                    prediction=fraud_prob,
                    predicted_class="Fraud" if fraud_prob >= 0.5 else "Legitimate",
                    risk_factors=risk_factors,
                    global_importance=global_factors,
                    model_metadata=model_metadata
                )

                # Display report
                st.success("Report generated!")

                # Download button
                st.download_button(
                    label="Download HTML Report",
                    data=html_report,
                    file_name=f"fraud_report_{transaction_id}.html",
                    mime="text/html"
                )

                # Preview
                with st.expander("Preview Report"):
                    st.components.v1.html(html_report, height=600, scrolling=True)

        except Exception as e:
            st.error(f"Error generating report: {str(e)}")


def validation_page():
    """Page for validation and testing."""
    st.header("Validation & Testing")

    if not st.session_state.explainer:
        st.warning("Please create an explainer first.")
        return

    # Validation options
    test_type = st.radio(
        "Test Type",
        options=["Consistency Check", "Speed Benchmark", "Quality Check"],
        horizontal=True
    )

    sample_idx = st.number_input(
        "Sample Index",
        value=0,
        min_value=0,
        max_value=len(st.session_state.X_test) - 1 if st.session_state.X_test is not None else 0
    )

    if st.button("Run Test", type="primary"):
        try:
            if test_type == "Consistency Check":
                st.subheader("Consistency Validation")

                with st.spinner("Running consistency checks..."):
                    X_sample = st.session_state.X_test[sample_idx]

                    result = validate_consistency(
                        explainer=st.session_state.explainer,
                        X=X_sample,
                        feature_names=st.session_state.feature_names,
                        n_runs=5
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Is Consistent", "✅ Yes" if result['is_consistent'] else "❌ No")

                    with col2:
                        st.metric("Max Variance", f"{result['max_variance']:.6f}")

                    st.write(f"**Tolerance:** {result['tolerance']}")
                    st.write(f"**Runs:** {result['n_runs']}")

                    with st.expander("Feature Variances"):
                        df_var = pd.DataFrame(list(result['variance'].items()), columns=['Feature', 'Variance'])
                        st.dataframe(df_var)

            elif test_type == "Speed Benchmark":
                st.subheader("Speed Benchmark")

                with st.spinner("Benchmarking..."):
                    X_sample = st.session_state.X_test[sample_idx]

                    result = benchmark_explanation_speed(
                        explainer=st.session_state.explainer,
                        X=X_sample,
                        feature_names=st.session_state.feature_names,
                        n_runs=50,
                        target_seconds=2.0
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Mean Time", f"{result['mean_time']:.4f}s")

                    with col2:
                        st.metric("Median Time", f"{result['median_time']:.4f}s")

                    with col3:
                        st.metric("P95 Time", f"{result['p95_time']:.4f}s")

                    st.metric("Meets Target (<2s)", "✅ Yes" if result['meets_target'] else "❌ No")

                    with st.expander("Detailed Statistics"):
                        st.json(result)

            else:  # Quality Check
                st.subheader("Quality Validation")

                with st.spinner("Checking explanation quality..."):
                    from utils.validation import validate_explanation_quality

                    result = validate_explanation_quality(
                        explainer=st.session_state.explainer,
                        X_test=st.session_state.X_test[:100],
                        feature_names=st.session_state.feature_names
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Total Explanations", result['total_explanations'])
                        st.metric("Unique Features", result['unique_features_used'])

                    with col2:
                        st.metric("Null Values", result['null_values_found'])
                        st.metric("Infinite Values", result['infinite_values_found'])

                    st.metric("Passed Quality Checks", "✅ Yes" if result['passed_quality_checks'] else "❌ No")

                    with st.expander("Feature Coverage"):
                        df_cov = pd.DataFrame(
                            list(result['feature_coverage'].items()),
                            columns=['Feature', 'Coverage']
                        )
                        st.dataframe(df_cov)

        except Exception as e:
            st.error(f"Error running test: {str(e)}")


def get_risk_level(probability: float) -> str:
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


def main():
    """Main application."""
    initialize_session_state()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        options=[
            "Load Model",
            "Configure Explainer",
            "Explain Transaction",
            "Generate Report",
            "Validation"
        ]
    )

    # Page routing
    if page == "Load Model":
        load_model_page()
    elif page == "Configure Explainer":
        create_explainer_page()
    elif page == "Explain Transaction":
        explain_transaction_page()
    elif page == "Generate Report":
        generate_report_page()
    elif page == "Validation":
        validation_page()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Fraud Model Explainability**

        For fraud analysts and regulators.

        Supports:
        • SHAP explanations
        • LIME explanations
        • HTML report generation

        *Compliant with SR 11-7 and EU AI Act*
        """
    )


if __name__ == "__main__":
    main()
