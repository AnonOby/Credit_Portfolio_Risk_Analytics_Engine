"""
Page 1 - Portfolio Overview

High-level dashboard summarising the entire loan portfolio:
grade distribution, loan status, term, purpose, issuance trend,
state map, and home ownership breakdown.
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

st.set_page_config(page_title="Portfolio Overview", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Caching wrapper around DataFetcher (cache per function for independent TTL)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _portfolio_summary():
    return DataFetcher.portfolio_summary()

@st.cache_data(ttl=3600)
def _grade_dist():
    return DataFetcher.grade_distribution()

@st.cache_data(ttl=3600)
def _status_dist():
    return DataFetcher.loan_status_distribution()

@st.cache_data(ttl=3600)
def _term_dist():
    return DataFetcher.term_distribution()

@st.cache_data(ttl=3600)
def _purpose_dist():
    return DataFetcher.purpose_distribution()

@st.cache_data(ttl=3600)
def _issuance():
    return DataFetcher.issuance_trend()

@st.cache_data(ttl=3600)
def _state_dist():
    return DataFetcher.state_distribution()

@st.cache_data(ttl=3600)
def _home_own():
    return DataFetcher.home_ownership_distribution()

@st.cache_data(ttl=3600)
def _funded_dist():
    return DataFetcher.funded_amount_distribution()


# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("📊 Portfolio Overview")
st.markdown("High-level summary of the **{:,.0f}**-loan portfolio.".format(
    _grade_dist()["count"].sum()))

# -- Top KPI row ----------------------------------------------------------------
grade_df = _portfolio_summary()
total_funded = grade_df["total_funded"].sum()
total_loans = grade_df["total_loans"].sum()
avg_rate = grade_df["avg_int_rate"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Loans", "{:,.0f}".format(total_loans))
col2.metric("Total Funded", "${:,.0f}".format(total_funded))
col3.metric("Avg Interest Rate", "{:.2f}%".format(avg_rate))
col4.metric("Grade Groups", str(len(grade_df)))

st.divider()

# -- Row 1: Grade distribution + Loan status ----------------------------------
tab1, tab2 = st.tabs(["By Grade", "By Term"])

with tab1:
    st.plotly_chart(ChartBuilder.grade_bar(grade_df), use_container_width=True)
with tab2:
    term_df = _term_dist()
    st.plotly_chart(ChartBuilder.term_pie(term_df), use_container_width=True)

# -- Row 2: Loan status + Funded amount distribution -------------------------
col_left, col_right = st.columns(2)
with col_left:
    status_df = _status_dist()
    st.plotly_chart(ChartBuilder.status_pie(status_df), use_container_width=True)
with col_right:
    funded_df = _funded_dist()
    st.plotly_chart(ChartBuilder.funded_hist(funded_df), use_container_width=True)

# -- Row 3: Issuance trend ----------------------------------------------------
st.subheader("Issuance Trend")
st.plotly_chart(ChartBuilder.issuance_trend(_issuance()), use_container_width=True)

# -- Row 4: Purpose + Home ownership -------------------------------------------
col_left, col_right = st.columns(2)
with col_left:
    purpose_df = _purpose_dist()
    st.plotly_chart(ChartBuilder.purpose_bar(purpose_df), use_container_width=True)
with col_right:
    home_df = _home_own()
    st.plotly_chart(ChartBuilder.home_ownership_bar(home_df), use_container_width=True)

# -- Row 5: US state map -------------------------------------------------------
st.subheader("Geographic Distribution")
state_df = _state_dist()
st.plotly_chart(ChartBuilder.state_map(state_df), use_container_width=True)

# -- Data table ----------------------------------------------------------------
with st.expander("View Raw Data"):
    st.dataframe(grade_df, use_container_width=True)