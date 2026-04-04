"""
Page 2 - Default Analysis

Deep-dive into default patterns across grades, time, purposes,
home ownership, DTI buckets, and FICO score ranges.
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.visualization.data_fetcher import DataFetcher
from src.visualization.charts import ChartBuilder

st.set_page_config(page_title="Default Analysis", page_icon="⚠️", layout="wide")

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _default_by_grade():
    return DataFetcher.default_rates_by_grade()

@st.cache_data(ttl=3600)
def _default_over_time():
    return DataFetcher.default_rates_over_time()

@st.cache_data(ttl=3600)
def _default_by_purpose():
    return DataFetcher.default_by_purpose()

@st.cache_data(ttl=3600)
def _default_by_home():
    return DataFetcher.default_by_home_ownership()

@st.cache_data(ttl=3600)
def _default_by_dti():
    return DataFetcher.default_by_dti_bucket()

@st.cache_data(ttl=3600)
def _default_by_fico():
    return DataFetcher.default_by_fico_bucket()


# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("⚠️ Default Analysis")
st.markdown("Explore default patterns across multiple dimensions.")

# -- KPI row ------------------------------------------------------------------
grade_df = _default_by_grade()
total_mature = grade_df["total_mature"].sum()
total_defaults = grade_df["defaults"].sum()
overall_dr = total_defaults / total_mature * 100 if total_mature > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Mature Loans", "{:,.0f}".format(total_mature))
col2.metric("Total Defaults", "{:,.0f}".format(total_defaults))
col3.metric("Overall Default Rate", "{:.2f}%".format(overall_dr))

st.divider()

# -- Grade default analysis ----------------------------------------------------
st.subheader("Default Rate by Grade")
st.plotly_chart(ChartBuilder.default_rate_by_grade(grade_df), use_container_width=True)

# -- Default rate over time ---------------------------------------------------
st.subheader("Default Rate Trend Over Time")
time_df = _default_over_time()
st.plotly_chart(ChartBuilder.default_trend(time_df), use_container_width=True)

# -- Default by purpose + home ownership ---------------------------------------
tab1, tab2 = st.tabs(["By Purpose", "By Home Ownership"])
with tab1:
    purpose_df = _default_by_purpose()
    st.plotly_chart(
        ChartBuilder.default_by_segment(purpose_df, "purpose",
                                        "Default Rate by Loan Purpose"),
        use_container_width=True,
    )
with tab2:
    home_df = _default_by_home()
    st.plotly_chart(
        ChartBuilder.default_by_segment(home_df, "home_ownership",
                                        "Default Rate by Home Ownership"),
        use_container_width=True,
    )

# -- Default by DTI + FICO -----------------------------------------------------
tab3, tab4 = st.tabs(["By DTI Bucket", "By FICO Score"])
with tab3:
    dti_df = _default_by_dti()
    dti_df["label"] = dti_df["dti_low"].astype(str) + "-" + dti_df["dti_high"].astype(str)
    st.plotly_chart(
        ChartBuilder.default_by_segment_bar(dti_df, "label",
                                            "Default Rate by DTI Range"),
        use_container_width=True,
    )
with tab4:
    fico_df = _default_by_fico()
    st.plotly_chart(ChartBuilder.default_by_fico(fico_df), use_container_width=True)

# -- Data tables ---------------------------------------------------------------
with st.expander("Default Rates by Grade (Raw)"):
    st.dataframe(grade_df, use_container_width=True)
with st.expander("Default Rates by Purpose (Raw)"):
    st.dataframe(purpose_df, use_container_width=True)