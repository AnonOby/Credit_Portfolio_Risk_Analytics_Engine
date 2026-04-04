"""
Page 3 - Risk Metrics

Expected Loss, LGD, concentration (HHI), interest rate analytics,
and stress-testing visualisation.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.visualization.data_fetcher import DataFetcher
from src.visualization.charts import ChartBuilder

st.set_page_config(page_title="Risk Metrics", page_icon="📉", layout="wide")

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _el():
    return DataFetcher.el_by_grade()

@st.cache_data(ttl=3600)
def _lgd():
    return DataFetcher.lgd_by_grade()

@st.cache_data(ttl=3600)
def _concentration():
    return DataFetcher.concentration_metrics()

@st.cache_data(ttl=3600)
def _int_rate():
    return DataFetcher.int_rate_by_grade()

@st.cache_data(ttl=3600)
def _portfolio_summary():
    return DataFetcher.portfolio_summary()


# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("📉 Risk Metrics")
st.markdown("Portfolio risk indicators: Expected Loss, LGD, concentration, and rate analytics.")

# -- KPI row ------------------------------------------------------------------
el_df = _el()
total_el = el_df["total_el"].sum()
total_ead = el_df["total_ead"].sum()
portfolio_el_pct = total_el / total_ead * 100 if total_ead > 0 else 0
lgd_df = _lgd()
avg_lgd = lgd_df["avg_lgd"].mean() * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Expected Loss", "${:,.0f}".format(total_el))
col2.metric("Portfolio EL %", "{:.2f}%".format(portfolio_el_pct))
col3.metric("Avg LGD", "{:.1f}%".format(avg_lgd))

st.divider()

# -- Expected Loss by grade ----------------------------------------------------
st.subheader("Expected Loss by Grade")
st.plotly_chart(ChartBuilder.el_by_grade(el_df), use_container_width=True)

# -- EL breakdown table --------------------------------------------------------
with st.expander("Expected Loss Breakdown"):
    display_el = el_df.copy()
    display_el["pd"] = display_el["pd"].apply(lambda x: "{:.1%}".format(x))
    display_el["lgd"] = display_el["lgd"].apply(lambda x: "{:.1%}".format(x))
    display_el["total_ead"] = display_el["total_ead"].apply(lambda x: "${:,.0f}".format(x))
    display_el["total_el"] = display_el["total_el"].apply(lambda x: "${:,.0f}".format(x))
    display_el["el_per_loan"] = display_el["el_per_loan"].apply(lambda x: "${:,.0f}".format(x))
    st.dataframe(display_el, use_container_width=True)

# -- LGD by grade --------------------------------------------------------------
st.subheader("Loss Given Default (LGD) by Grade")
st.plotly_chart(ChartBuilder.lgd_by_grade(lgd_df), use_container_width=True)

# -- Concentration (HHI) -------------------------------------------------------
st.subheader("Portfolio Concentration (HHI Index)")
conc_df = _concentration()
total_hhi = conc_df["hhi_contrib"].sum()
st.info("Total HHI Index: **{:.0f}** / 10,000  ({})".format(
    total_hhi,
    "Highly Concentrated" if total_hhi > 2500 else
    "Moderately Concentrated" if total_hhi > 1500 else "Well Diversified",
))
st.plotly_chart(ChartBuilder.concentration_hhi(conc_df), use_container_width=True)

# -- Interest rate analytics ---------------------------------------------------
st.subheader("Interest Rate Distribution by Grade")
rate_df = _int_rate()
st.plotly_chart(ChartBuilder.int_rate_box(rate_df), use_container_width=True)

# -- Stress test simulation (simple, client-side) ------------------------------
st.subheader("Simple Stress Test (Scenario Analysis)")
st.markdown(
    "Apply uniform PD/LGD multipliers to simulate adverse scenarios. "
    "Results are illustrative and computed entirely client-side."
)

# These variables should be defined earlier in the page
# el_df = DataFetcher.el_by_grade()
# total_el = el_df["total_el"].sum()

with st.form("stress_test_form"):
    col_a, col_b = st.columns(2)
    with col_a:
        pd_shock = st.slider("PD Multiplier", 1.0, 3.0, 1.5, 0.1)
    with col_b:
        lgd_shock = st.slider("LGD Multiplier", 1.0, 2.0, 1.2, 0.1)
    submitted = st.form_submit_button("Run Stress Test")

if submitted:
    stressed_el = el_df.copy()
    stressed_el["stressed_pd"] = np.minimum(stressed_el["pd"] * pd_shock, 1.0)
    stressed_el["stressed_lgd"] = np.minimum(stressed_el["lgd"] * lgd_shock, 1.0)
    stressed_el["stressed_el"] = (
        stressed_el["stressed_pd"] * stressed_el["stressed_lgd"] * stressed_el["total_ead"]
    )
    stressed_total = stressed_el["stressed_el"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline EL", "${:,.0f}".format(total_el))
    col2.metric("Stressed EL", "${:,.0f}".format(stressed_total))
    col3.metric("Increase", "+${:,.0f} ({:+.1f}%)".format(
        stressed_total - total_el,
        (stressed_total - total_el) / total_el * 100 if total_el > 0 else 0,
    ))

    import plotly.graph_objects as go
    colors = ["#2980b9", "#e74c3c"]
    fig_stress = go.Figure(data=[
        go.Bar(name="Baseline EL", x=el_df["grade"], y=el_df["total_el"],
               marker_color=colors[0]),
        go.Bar(name="Stressed EL", x=stressed_el["grade"], y=stressed_el["stressed_el"],
               marker_color=colors[1]),
    ])
    fig_stress.update_layout(barmode="group", title="Baseline vs Stressed EL by Grade",
                             template="plotly_white", height=450)
    fig_stress.update_yaxes(title_text="Expected Loss ($)", tickprefix="$")
    st.plotly_chart(fig_stress, use_container_width=True)