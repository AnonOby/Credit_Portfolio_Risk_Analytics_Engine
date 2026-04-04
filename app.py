"""
Credit Portfolio Risk Analytics - Streamlit Dashboard

Entry point for the interactive web dashboard. Run with:
    streamlit run app.py

Multi-page navigation via sidebar radio buttons.
Each page is dynamically imported and executed.
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable by all sub-modules
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Portfolio Risk Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Warm up database connection to avoid lazy loading issues on page switch
# ---------------------------------------------------------------------------
from src.database.connection import get_engine

@st.cache_resource
def _get_engine():
    """Return cached database engine."""
    return get_engine()

_engine = _get_engine()
if _engine is None:
    st.error("❌ Failed to connect to database. Please check PostgreSQL service and config.py.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🏦 Credit Risk Dashboard")
st.sidebar.markdown("---")

page_options = {
    "📊 Portfolio Overview": "src.visualization.pages.01_portfolio_overview",
    "⚠️ Default Analysis": "src.visualization.pages.02_default_analysis",
    "📐 Risk Metrics": "src.visualization.pages.03_risk_metrics",
    "📡 Model Performance": "src.visualization.pages.04_model_performance",
}

selected_page = st.sidebar.radio(
    "Navigation",
    list(page_options.keys()),
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("Credit Portfolio Risk Analytics Engine")
st.sidebar.caption("Data source: PostgreSQL (loans_master)")

# ---------------------------------------------------------------------------
# Dynamic page loader with error handling
# ---------------------------------------------------------------------------
module_path = page_options[selected_page]

try:
    import importlib

    # Clear any cached state that might interfere with page reload
    # (Streamlit automatically re-runs the script, but we force re-import)
    if module_path in sys.modules:
        del sys.modules[module_path]

    page_module = importlib.import_module(module_path)

    # The page module should contain its own layout and visualisations.
    # No additional actions needed here.
except ImportError as exc:
    st.error(f"Failed to load page '{selected_page}': {exc}")
    st.info(
        "Make sure you have installed all dependencies and the project "
        "root is on your PYTHONPATH."
    )
    st.stop()
except Exception as exc:
    st.error(f"Unexpected error while loading page '{selected_page}': {exc}")
    st.exception(exc)
    st.stop()