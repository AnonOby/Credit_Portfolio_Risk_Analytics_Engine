"""
Reusable Plotly chart components for the Credit Risk Dashboard.

Every function returns a plotly.graph_objects.Figure ready for
st.plotly_chart() rendering.  A consistent colour palette and
layout template are applied across all charts.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Consistent theme
# ---------------------------------------------------------------------------

# Grade colour palette (A=green -> G=red)
GRADE_COLORS = {
    "A": "#2ecc71", "B": "#27ae60", "C": "#f1c40f",
    "D": "#e67e22", "E": "#e74c3c", "F": "#c0392b", "G": "#8e44ad",
}

PRIMARY_COLOR = "#2980b9"
SECONDARY_COLOR = "#e74c3c"
TERTIARY_COLOR = "#2ecc71"
ACCENT_COLORS = px.colors.qualitative.Set2


def _apply_layout(fig, title, height=450):
    """Apply a consistent dark-header layout template."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#2c3e50")),
        template="plotly_white",
        height=height,
        margin=dict(l=40, r=20, t=60, b=40),
        font=dict(family="Segoe UI, Arial, sans-serif", size=12),
    )
    return fig


# ---------------------------------------------------------------------------
# Portfolio overview charts
# ---------------------------------------------------------------------------

def grade_bar_chart(df: pd.DataFrame, value_col: str = "total_funded",
                    title: str = "Total Funded Amount by Grade") -> go.Figure:
    """Bar chart coloured by grade for any metric in *df*."""
    colors = [GRADE_COLORS.get(g, "#95a5a6") for g in df["grade"]]
    fig = go.Figure(
        go.Bar(x=df["grade"], y=df[value_col], marker_color=colors,
               texttemplate="%{y:,.0f}", textposition="outside"),
    )
    fig.update_yaxes(title_text=value_col.replace("_", " ").title())
    _apply_layout(fig, title)
    return fig


def loan_status_pie(df: pd.DataFrame, title: str = "Loan Status Distribution") -> go.Figure:
    """Donut chart for loan_status distribution."""
    fig = go.Figure(
        go.Pie(labels=df["loan_status"], values=df["count"], hole=0.45,
               textinfo="label+percent", textposition="outside",
               marker_colors=ACCENT_COLORS),
    )
    _apply_layout(fig, title)
    return fig


def term_pie(df: pd.DataFrame, title: str = "Loan Term Distribution") -> go.Figure:
    """Donut chart for 36 vs 60 month terms."""
    fig = go.Figure(
        go.Pie(labels=df["term"].astype(str), values=df["count"], hole=0.4,
               textinfo="label+percent",
               marker_colors=["#3498db", "#e74c3c"]),
    )
    _apply_layout(fig, title)
    return fig


def purpose_bar_chart(df: pd.DataFrame, title: str = "Loans by Purpose") -> go.Figure:
    """Horizontal bar chart for loan purpose distribution."""
    df = df.sort_values("count")
    fig = go.Figure(
        go.Bar(y=df["purpose"], x=df["count"], orientation="h",
                marker_color=PRIMARY_COLOR, texttemplate="%{x:,.0f}",
                textposition="outside"),
    )
    _apply_layout(fig, title, height=max(400, len(df) * 25))
    return fig


def issuance_trend_chart(df: pd.DataFrame,
                         title: str = "Monthly Loan Issuance") -> go.Figure:
    """Dual-axis line chart: issuance count + total funded amount."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    df["month"] = pd.to_datetime(df["month"])
    fig.add_trace(
        go.Scatter(x=df["month"], y=df["count"], name="Loan Count",
                   line=dict(color=PRIMARY_COLOR, width=2), fill="tozeroy"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["month"], y=df["total_funded"], name="Total Funded ($)",
                   line=dict(color=SECONDARY_COLOR, width=2), fill="tozeroy"),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text="Total Funded ($)", secondary_y=True, tickprefix="$")
    fig.update_xaxes(title_text="")
    _apply_layout(fig, title, 500)
    return fig


def state_choropleth(df: pd.DataFrame,
                     title: str = "Portfolio by State") -> go.Figure:
    """US state choropleth map coloured by loan count."""
    fig = go.Figure(
        go.Choropleth(
            locations=df["addr_state"], z=df["count"],
            locationmode="USA-states", colorscale="Blues",
            colorbar_title="Loan Count",
        ),
    )
    fig.update_geos(scope="usa")
    _apply_layout(fig, title, 500)
    return fig


def home_ownership_bar(df: pd.DataFrame,
                       title: str = "Portfolio by Home Ownership") -> go.Figure:
    """Grouped bar: count + avg funded per home ownership type."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=df["home_ownership"], y=df["count"], name="Loan Count",
               marker_color=PRIMARY_COLOR),
    )
    fig2 = go.Figure(
        go.Bar(x=df["home_ownership"], y=df["avg_funded"], name="Avg Funded ($)",
               marker_color=SECONDARY_COLOR),
    )
    # Use subplots side-by-side
    fig_combined = make_subplots(rows=1, cols=2,
                                 subplot_titles=["Loan Count", "Avg Funded Amount ($)"])
    fig_combined.add_trace(fig.data[0], row=1, col=1)
    fig_combined.add_trace(fig2.data[0], row=1, col=2)
    _apply_layout(fig_combined, title)
    return fig_combined


def funded_histogram(df: pd.DataFrame,
                     title: str = "Funded Amount Distribution") -> go.Figure:
    """Histogram of funded amounts."""
    fig = go.Figure(
        go.Bar(x=df["low"].astype(str) + "-" + df["high"].astype(str),
               y=df["count"], marker_color=PRIMARY_COLOR),
    )
    fig.update_xaxes(title_text="Funded Amount Range ($)", tickangle=45)
    fig.update_yaxes(title_text="Count")
    _apply_layout(fig, title)
    return fig


# ---------------------------------------------------------------------------
# Default analysis charts
# ---------------------------------------------------------------------------

def default_rate_by_grade_bar(df: pd.DataFrame,
                              title: str = "Default Rate by Grade") -> go.Figure:
    """Stacked bar: total mature + defaulted, with default rate line."""
    colors = [GRADE_COLORS.get(g, "#95a5a6") for g in df["grade"]]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=df["grade"], y=df["total_mature"], name="Total Mature",
               marker_color=colors, texttemplate="%{y:,.0f}", textposition="outside"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=df["grade"], y=df["defaults"], name="Defaulted",
               marker_color=SECONDARY_COLOR,
               texttemplate="%{y:,.0f}", textposition="inside"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["grade"], y=df["default_rate"], name="Default Rate (%)",
                   line=dict(color="#2c3e50", width=3), mode="lines+markers+text",
                   texttemplate="%{y:.1f}%", textposition="top center"),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Loan Count", secondary_y=False)
    fig.update_yaxes(title_text="Default Rate (%)", secondary_y=True, range=[0, 50])
    _apply_layout(fig, title, 500)
    return fig


def default_trend_chart(df: pd.DataFrame,
                        title: str = "Default Rate Over Time") -> go.Figure:
    """Time series of monthly default rate with volume shading."""
    df["issue_month"] = pd.to_datetime(df["issue_month"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df["issue_month"], y=df["default_rate"],
                   name="Default Rate (%)", line=dict(color=SECONDARY_COLOR, width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=df["issue_month"], y=df["total"], name="Mature Loan Volume",
               marker_color=PRIMARY_COLOR, opacity=0.3),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Default Rate (%)", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)
    fig.update_xaxes(title_text="")
    _apply_layout(fig, title, 500)
    return fig


def default_by_segment_bar(df: pd.DataFrame, segment_col: str,
                           title: str = "Default Rate by Segment") -> go.Figure:
    """Horizontal bar chart coloured by default rate (low=green, high=red)."""
    df = df.sort_values("default_rate")
    colors = []
    max_dr = df["default_rate"].max()
    for _, row in df.iterrows():
        ratio = row["default_rate"] / max_dr if max_dr > 0 else 0
        # Interpolate green -> yellow -> red
        r = min(1.0, 2 * ratio)
        g = min(1.0, 2 * (1 - ratio))
        colors.append("rgb({},{},0)".format(int(r * 255), int(g * 255)))
    fig = go.Figure(
        go.Bar(y=df[segment_col], x=df["default_rate"], orientation="h",
                marker_color=colors,
                texttemplate="%{x:.1f}%", textposition="outside"),
    )
    fig.update_xaxes(title_text="Default Rate (%)")
    _apply_layout(fig, title, height=max(400, len(df) * 28))
    return fig


def default_by_fico_scatter(df: pd.DataFrame,
                            title: str = "Default Rate vs FICO Score") -> go.Figure:
    """Scatter with size representing volume, coloured by default rate."""
    df["label"] = df["fico_low"].astype(str) + "-" + df["fico_high"].astype(str)
    fig = go.Figure(
        go.Scatter(
            x=df["label"], y=df["default_rate"],
            mode="markers+lines+text",
            marker=dict(size=df["total"] / df["total"].max() * 40 + 5,
                        color=df["default_rate"], colorscale="RdYlGn_r",
                        showscale=True, colorbar_title="Default Rate %"),
            line=dict(color="#bdc3c7", width=1),
            texttemplate="%{y:.1f}%", textposition="top center",
        ),
    )
    fig.update_xaxes(title_text="FICO Score Range")
    fig.update_yaxes(title_text="Default Rate (%)")
    _apply_layout(fig, title)
    return fig


# ---------------------------------------------------------------------------
# Risk metrics charts
# ---------------------------------------------------------------------------

def lgd_by_grade_bar(df: pd.DataFrame,
                      title: str = "LGD Statistics by Grade") -> go.Figure:
    """Box-like range chart: P25-P75 bar with P50 marker."""
    colors = [GRADE_COLORS.get(g, "#95a5a6") for g in df["grade"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="P75", x=df["grade"], y=df["p75_lgd"] - df["p25_lgd"],
               base=df["p25_lgd"], marker_color=colors,
               texttemplate="%{customdata:.1%}", customdata=df["p75_lgd"],
               textposition="outside"),
    )
    fig.add_trace(
        go.Scatter(name="Median", x=df["grade"], y=df["p50_lgd"],
                   mode="markers+text", marker=dict(size=10, color="#2c3e50"),
                   texttemplate="%{y:.1%}", textposition="top center"),
    )
    fig.update_yaxes(title_text="LGD", tickformat=".0%")
    _apply_layout(fig, title)
    return fig


def el_by_grade_bar(df: pd.DataFrame,
                    title: str = "Expected Loss by Grade") -> go.Figure:
    """Stacked contribution: EL per loan + total EL with PD/LGD annotation."""
    colors = [GRADE_COLORS.get(g, "#95a5a6") for g in df["grade"]]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(name="Total EL ($)", x=df["grade"], y=df["total_el"],
               marker_color=colors, texttemplate="$%{y:,.0f}",
               textposition="outside"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(name="PD", x=df["grade"], y=df["pd"],
                   mode="lines+markers+text",
                   line=dict(color=SECONDARY_COLOR, width=2, dash="dot"),
                   texttemplate="%{y:.1%}", textposition="bottom center"),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(name="LGD", x=df["grade"], y=df["lgd"],
                   mode="lines+markers+text",
                   line=dict(color="#8e44ad", width=2, dash="dash"),
                   texttemplate="%{y:.1%}", textposition="top center"),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Expected Loss ($)", secondary_y=False, tickprefix="$")
    fig.update_yaxes(title_text="Rate", secondary_y=True, tickformat=".0%")
    _apply_layout(fig, title, 520)
    return fig


def concentration_hhi_chart(df: pd.DataFrame,
                            title: str = "Portfolio Concentration (HHI)") -> go.Figure:
    """Horizontal bar coloured by HHI contribution intensity."""
    df = df.sort_values("hhi_contrib")
    max_hhi = df["hhi_contrib"].max()
    colors = []
    for _, row in df.iterrows():
        ratio = row["hhi_contrib"] / max_hhi if max_hhi > 0 else 0
        r = min(1.0, 2 * ratio)
        g = min(1.0, 2 * (1 - ratio))
        colors.append("rgb({},{},0)".format(int(r * 255), int(g * 255)))
    fig = go.Figure(
        go.Bar(y=df["segment"], x=df["hhi_contrib"], orientation="h",
                marker_color=colors,
                texttemplate="%{x:.0f}", textposition="outside"),
    )
    fig.update_xaxes(title_text="HHI Contribution (points)")
    _apply_layout(fig, title, height=max(400, len(df) * 25))
    return fig


def interest_rate_box_by_grade(df: pd.DataFrame,
                               title: str = "Interest Rate Range by Grade") -> go.Figure:
    """Candlestick-style chart showing min/avg/max interest rate per grade."""
    colors = [GRADE_COLORS.get(g, "#95a5a6") for g in df["grade"]]
    fig = go.Figure()
    # Lower whisker to avg (lighter)
    fig.add_trace(
        go.Bar(name="Min-Avg Range", x=df["grade"],
               y=df["avg_rate"] - df["min_rate"], base=df["min_rate"],
               marker_color=colors, opacity=0.5),
    )
    # Avg to max (darker)
    fig.add_trace(
        go.Bar(name="Avg-Max Range", x=df["grade"],
               y=df["max_rate"] - df["avg_rate"], base=df["avg_rate"],
               marker_color=colors, opacity=0.9),
    )
    # Avg line
    fig.add_trace(
        go.Scatter(name="Average", x=df["grade"], y=df["avg_rate"],
                   mode="markers+text", marker=dict(size=8, color="#2c3e50"),
                   texttemplate="%{y:.2f}%", textposition="top center"),
    )
    fig.update_yaxes(title_text="Interest Rate (%)")
    fig.update_xaxes(title_text="Grade")
    _apply_layout(fig, title)
    return fig


def kpi_card_style() -> dict:
    """Return a dict of style options for Streamlit metric cards."""
    return {
        "label_font_size": "0.9rem",
        "value_font_size": "1.6rem",
    }


# ---------------------------------------------------------------------------
# Chart builder facade
# ---------------------------------------------------------------------------

class ChartBuilder:
    """
    Convenience façade that groups all chart factory functions.
    Usage:  ChartBuilder.grade_bar_chart(df)
    """

    grade_bar = staticmethod(grade_bar_chart)
    status_pie = staticmethod(loan_status_pie)
    term_pie = staticmethod(term_pie)
    purpose_bar = staticmethod(purpose_bar_chart)
    issuance_trend = staticmethod(issuance_trend_chart)
    state_map = staticmethod(state_choropleth)
    home_ownership_bar = staticmethod(home_ownership_bar)
    funded_hist = staticmethod(funded_histogram)
    default_rate_by_grade = staticmethod(default_rate_by_grade_bar)
    default_trend = staticmethod(default_trend_chart)
    default_by_segment = staticmethod(default_by_segment_bar)
    default_by_fico = staticmethod(default_by_fico_scatter)
    lgd_by_grade = staticmethod(lgd_by_grade_bar)
    el_by_grade = staticmethod(el_by_grade_bar)
    concentration_hhi = staticmethod(concentration_hhi_chart)
    int_rate_box = staticmethod(interest_rate_box_by_grade)