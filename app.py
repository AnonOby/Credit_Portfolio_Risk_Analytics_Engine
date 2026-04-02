"""
Credit Portfolio Risk Analytics - Streamlit Dashboard

Entry point for the interactive web dashboard. Run with:
    streamlit run app.py

Or via the module path:
    streamlit run app.py --server.port 8501
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable by all sub-modules
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Multi-page navigation setup
# ---------------------------------------------------------------------------
# Streamlit >= 1.30 supports native multi-page apps via the pages/ directory.
# We place the page files in src/visualization/pages/ but Streamlit requires
# them under <entry_point_dir>/pages/.  Instead of duplicating files, we
# configure the entry point as a single-page app with sidebar navigation
# that dynamically imports each page module.
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Credit Portfolio Risk Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
st.sidebar.title("🏦 Credit Risk Dashboard")
st.sidebar.markdown("---")

page_options = {
    "📊 Portfolio Overview": "src.visualization.pages.01_portfolio_overview",
    "⚠️ Default Analysis": "src.visualization.pages.02_default_analysis",
    "📉 Risk Metrics": "src.visualization.pages.03_risk_metrics",
    "🤖 Model Performance": "src.visualization.pages.04_model_performance",
}

selected_page = st.sidebar.radio(
    "Navigation",
    list(page_options.keys()),
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("Credit Portfolio Risk Analytics Engine")
st.sidebar.caption("Data source: PostgreSQL (loans_master)")

# Dynamic import and exec
module_path = page_options[selected_page]
try:
    import importlib
    page_module = importlib.import_module(module_path)
    # The page module runs its own st.title / layout on import
except ImportError as exc:
    st.error("Failed to load page '{}': {}".format(selected_page, exc))
    st.info("Make sure you have installed all dependencies and the project "
            "root is on your PYTHONPATH.")