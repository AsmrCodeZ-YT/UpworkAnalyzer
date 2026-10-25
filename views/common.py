"""Shared presentation helpers for the dashboard pages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PALETTE = ["#4f6df5", "#0ea5e9", "#14b8a6", "#f59e0b", "#8b5cf6", "#ef4444", "#64748b", "#ec4899"]
INDUSTRY_COLORS = {
    "FinTech": "#4f6df5",
    "Finance": "#0ea5e9",
    "Banking": "#14b8a6",
    "Compliance": "#f59e0b",
    "Crypto": "#8b5cf6",
    "Cross-industry": "#64748b",
    "Other": "#94a3b8",
}
NICHE_COLORS = {
    "FinTech — Payments": "#4f6df5",
    "Finance — Financial Operations": "#0ea5e9",
    "Finance — Risk & Fraud": "#ef4444",
    "Finance — Investment Intelligence": "#38bdf8",
    "Banking — Lending & Credit": "#14b8a6",
    "Compliance — AML / KYC": "#f59e0b",
    "Crypto — On-chain Analytics": "#8b5cf6",
    "Crypto — DeFi": "#a855f7",
    "Crypto — Exchanges": "#c084fc",
    "Cross-industry — Data Engineering": "#64748b",
    "Other — General Analytics": "#94a3b8",
}
PROPOSAL_COLORS = {
    "0–5": "#16a34a",
    "6–10": "#65a30d",
    "11–20": "#eab308",
    "21–50": "#f97316",
    "50+": "#dc2626",
    "Unknown": "#94a3b8",
}
MUTED = "#64748b"


def chart(fig: go.Figure, height: int = 340, legend: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#1e293b"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(gridcolor="#e2e8f0", zeroline=False)
    fig.update_yaxes(gridcolor="#e2e8f0", zeroline=False)
    return fig


def show(fig: go.Figure, height: int = 340, legend: bool = False) -> None:
    st.plotly_chart(
        chart(fig, height=height, legend=legend),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f"#### {title}")
    if caption:
        st.caption(caption)


def money(value, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return "n/a"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M{suffix}"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K{suffix}"
    return f"${value:,.0f}{suffix}"


def pct(value, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}%"


def safe(value, fallback="n/a") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return fallback
    return value


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        with col:
            st.metric(label, value, help=help_text)


def note(text: str, kind: str = "info") -> None:
    if kind == "warning":
        st.warning(text, icon="⚠️")
    elif kind == "data":
        st.caption(text)
    else:
        st.info(text, icon="ℹ️")


def job_table(df: pd.DataFrame, n: int | None = None, show_score: bool = True) -> None:
    if df.empty:
        st.caption("No jobs match the current filters.")
        return
    view = df.copy()
    if n is not None:
        view = view.head(n)

    view["Rate / Budget"] = view["rate_label"]
    view["Posted"] = view["posted_date"].dt.strftime("%Y-%m-%d").fillna("unknown")
    view["Client spent"] = view["client_spent"].map(lambda v: money(v) if pd.notna(v) else "n/a")
    view["Score"] = view["job_score"].round(0)
    view["Data"] = view["score_coverage"].round(0)
    view["Link"] = view["url"]

    columns = ["title", "niche", "problem_primary", "Rate / Budget", "proposal_bucket",
               "Posted", "client_country", "Client spent", "Data"]
    if show_score:
        columns.insert(1, "Score")
    columns.append("Link")

    st.dataframe(
        view[columns],
        hide_index=True,
        width="stretch",
        column_config={
            "title": st.column_config.TextColumn("Job", width="large"),
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.0f", width="small"
            ),
            "niche": st.column_config.TextColumn("Niche", width="medium"),
            "problem_primary": st.column_config.TextColumn("Problem", width="medium"),
            "Rate / Budget": st.column_config.TextColumn("Rate / budget", width="small"),
            "proposal_bucket": st.column_config.TextColumn("Proposals", width="small"),
            "Posted": st.column_config.TextColumn("Posted", width="small"),
            "client_country": st.column_config.TextColumn("Country", width="small"),
            "Client spent": st.column_config.TextColumn("Client spent", width="small"),
            "Data": st.column_config.ProgressColumn(
                "Data", min_value=0, max_value=100, format="%.0f", width="small",
                help="Share of the job score based on available fields; low values mean more missing data.",
            ),
            "Link": st.column_config.LinkColumn("Link", display_text="Open ↗", width="small"),
        },
    )


def styled_table(df: pd.DataFrame, column_config: dict | None = None, height: int | None = None) -> None:
    kwargs = {
        "hide_index": True,
        "width": "stretch",
        "column_config": column_config or {},
    }
    if height is not None:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)


def no_data() -> None:
    st.warning("No data left after the current filters. Widen the filters in the sidebar to continue.")


def sparkline(values: list[float], color: str = "#4f6df5", height: int = 42) -> go.Figure:
    fig = go.Figure(go.Scatter(y=values, mode="lines", line=dict(color=color, width=2)))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False,
    )
    return fig
