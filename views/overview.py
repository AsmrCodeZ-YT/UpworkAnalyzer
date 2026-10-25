"""Page 1 — Market Overview."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as a
from views.common import (
    INDUSTRY_COLORS, MUTED, NICHE_COLORS, PROPOSAL_COLORS,
    job_table, kpi_row, money, no_data, note, pct, section, show,
)


def render(df: pd.DataFrame, full: pd.DataFrame) -> None:
    st.title("Market Overview")
    st.caption(
        "Where is the demand, where is the money, and what deserves attention first? "
        "All figures respect the global filters in the sidebar."
    )

    if df.empty:
        no_data()
        return

    relevant = df[df["relevant"]].copy()
    if relevant.empty:
        st.warning("No relevant jobs in the current selection. Try widening the industry / niche filters.")
        return

    hourly = relevant.loc[relevant["job_type"].eq("Hourly"), "hourly_mid"].dropna()
    fixed = relevant.loc[relevant["job_type"].eq("Fixed price"), "fixed_budget"].dropna()
    w30 = a.demand_window(df, 30)
    fresh7 = int(relevant["fresh_7d"].sum())
    fresh_low = int((relevant["fresh_7d"] & relevant["low_competition"]).sum())
    high_value_share = relevant["high_value"].mean() * 100

    growth = w30["growth"]
    growth_label = f"{growth:+.0f}%" if pd.notna(growth) else "n/a"
    niche_table = a.build_niche_table(df)
    top_niche = niche_table.iloc[0]["niche"] if not niche_table.empty else "n/a"

    kpi_row([
        ("Relevant jobs", f"{len(relevant):,}", "Unique jobs that actually match a target niche (not just the search query)."),
        ("30-day demand", f"{w30['current']:,}", f"Jobs posted in the last 30 days · change vs previous 30 days: {growth_label}"),
        ("Fresh & low competition", f"{fresh_low:,}", "Relevant jobs ≤ 7 days old with 0–10 proposals. Best quick-review pool."),
        ("Median hourly rate", money(hourly.median(), "/h") if len(hourly) else "n/a", f"Across {len(hourly)} hourly jobs. Fixed-price jobs excluded."),
        ("Median fixed budget", money(fixed.median()) if len(fixed) else "n/a", f"Across {len(fixed)} fixed-price jobs."),
    ])
    kpi_row([
        ("High-value share", pct(high_value_share), f"Hourly midpoint ≥ ${a.HIGH_VALUE_HOURLY:.0f}/h or fixed budget ≥ ${a.HIGH_VALUE_FIXED:,.0f}."),
        ("New in last 7 days", f"{fresh7:,}", "Relevant jobs posted within the last 7 days. Treat dates as estimates where noted."),
        ("Top niche by score", top_niche, "Bayesian-adjusted Niche Opportunity Score. See Niche Intelligence for components."),
        ("Payment-verified clients", pct(relevant["payment_verified"].mean() * 100), "Share of relevant jobs whose client has verified payment."),
        ("Long-term share", pct(relevant["long_term"].mean() * 100), "Jobs with duration > 6 months."),
    ])

    st.divider()

    left, right = st.columns([1.55, 1])

    with left:
        section("Demand over time", "Weekly job volume by posting date. Most dates are estimated from 'X days ago' strings, so treat the shape as directional.")
        dated = df[df["posted_date"].notna()].copy()
        if dated.empty:
            st.caption("No dates available.")
        else:
            weekly = (
                dated.set_index("posted_date")
                .resample("W-MON")["job_id"].count()
                .rename("jobs")
                .reset_index()
            )
            weekly = weekly[weekly["jobs"] > 0].tail(18)
            fig = go.Figure()
            fig.add_bar(
                x=weekly["posted_date"], y=weekly["jobs"],
                marker_color="#c7d2fe", name="All jobs", hovertemplate="Week of %{x|%b %d}<br>%{y} jobs<extra></extra>",
            )
            if len(weekly) >= 3:
                fig.add_trace(go.Scatter(
                    x=weekly["posted_date"], y=weekly["jobs"].rolling(3, min_periods=1).mean(),
                    mode="lines", name="3-week avg", line=dict(color="#4f6df5", width=2.5),
                    hovertemplate="Week of %{x|%b %d}<br>%{y:.1f} avg<extra></extra>",
                ))
            fig.update_yaxes(title=None, rangemode="tozero")
            fig.update_xaxes(title=None)
            show(fig, height=310, legend=True)

    with right:
        section("Demand windows", "Current vs. previous period. Growth is undefined when the previous period is empty.")
        rows = []
        for days in (7, 30, 90):
            win = a.demand_window(df, days)
            rows.append({
                "window": f"{days}d",
                "current": win["current"],
                "previous": win["previous"],
            })
        wdf = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_bar(x=wdf["window"], y=wdf["previous"], name="Previous", marker_color="#cbd5e1")
        fig.add_bar(x=wdf["window"], y=wdf["current"], name="Current", marker_color="#4f6df5")
        for i, row in wdf.iterrows():
            change = "n/a" if row["previous"] == 0 else f"{(row['current'] - row['previous']) / row['previous'] * 100:+.0f}%"
            fig.add_annotation(x=row["window"], y=max(row["current"], row["previous"]),
                               text=change, showarrow=False, yshift=12, font=dict(size=11, color=MUTED))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title=None, rangemode="tozero")
        fig.update_xaxes(title=None)
        show(fig, height=310, legend=True)

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        section("Demand by niche", "Unique relevant jobs per sub-niche.")
        counts = relevant.groupby("niche", as_index=False).size().sort_values("size")
        fig = px.bar(
            counts, x="size", y="niche", orientation="h",
            color="niche", color_discrete_map=NICHE_COLORS, text="size",
            labels={"size": "Jobs", "niche": ""},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title="Jobs")
        show(fig, height=360)

    with right:
        section("Demand vs. economic strength", "Bubble position compares volume with normalized economic power; hover shows the underlying rates.")
        if niche_table.empty:
            st.caption("Not enough data.")
        else:
            nt = niche_table.copy()
            nt["label_short"] = nt["niche"].str.replace(" — ", "<br>", regex=False)
            nt["hourly_text"] = nt["median_hourly"].map(lambda v: money(v, "/h") if pd.notna(v) else "n/a")
            nt["fixed_text"] = nt["median_fixed"].map(lambda v: money(v) if pd.notna(v) else "n/a")
            fig = px.scatter(
                nt, x="jobs", y="E_economics", size="jobs", color="niche",
                color_discrete_map=NICHE_COLORS, hover_name="niche",
                hover_data={"jobs": True, "median_hourly": False, "median_fixed": False,
                            "hourly_text": True, "fixed_text": True, "E_economics": ":.0f"},
                custom_data=["hourly_text", "fixed_text", "low_comp_share"],
                labels={"jobs": "Relevant jobs", "E_economics": "Economic strength (0–100)"},
            )
            fig.update_traces(
                marker=dict(line=dict(color="white", width=1.5)),
                hovertemplate=(
                    "<b>%{hovertext}</b><br>%{x} jobs · econ %{y:.0f}<br>"
                    "Median hourly: %{customdata[0]}<br>Median fixed: %{customdata[1]}<br>"
                    "Low competition: %{customdata[2]:.0f}%<extra></extra>"
                ),
            )
            fig.update_layout(showlegend=False, height=360)
            show(fig, height=360)

    st.divider()
    section("Top opportunities", "Ranked by the Individual Job Opportunity Score among relevant jobs. Open any job to validate the score manually.")
    top = relevant.sort_values("job_score", ascending=False)
    job_table(top, n=12)
    note(
        "The score is a triage aid, not a decision. It re-normalizes weights when data is missing and never ranks "
        "non-relevant jobs. Always check the actual brief before applying.",
        kind="data",
    )
    if len(full) and len(df) < len(full):
        st.caption(f"Showing {len(df):,} of {len(full):,} unique jobs after filters.")
