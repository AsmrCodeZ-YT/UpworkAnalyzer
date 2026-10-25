"""Page 2 — Niche Intelligence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as a
from views.common import (
    MUTED, NICHE_COLORS, job_table, kpi_row, money, no_data, note,
    pct, section, show, styled_table,
)


def render(df: pd.DataFrame, full: pd.DataFrame) -> None:
    st.title("Niche Intelligence")
    st.caption("Which sub-niche deserves specialisation? Compare demand, economics, competition, client quality, and freshness.")

    relevant = df[df["relevant"]].copy()
    if relevant.empty:
        no_data()
        return

    table = a.build_niche_table(df)
    if table.empty:
        no_data()
        return

    with st.expander("How the Niche Opportunity Score is built", expanded=False):
        st.markdown(
            "Raw score = **30% demand + 25% economics + 20% competition manageability + "
            "15% client quality + 10% freshness**. Every component is scaled to 0–100. "
            "The ranking then applies Bayesian shrinkage (prior 45, strength 4) so a niche with only "
            "one or two jobs cannot outrank a well-sampled niche. Both scores are shown."
        )

    display = table.copy()
    display["rank"] = display["rank"].astype(int)
    display["Niches"] = display["niche"]
    display["Jobs"] = display["jobs"]
    display["Share"] = display["share"].map(pct)
    display["Score"] = display["niche_score"].round(1)
    display["Raw score"] = display["niche_score_raw"].round(1)
    display["Median hourly"] = display["median_hourly"].map(lambda v: money(v, "/h"))
    display["Median fixed"] = display["median_fixed"].map(money)
    display["Low-comp share"] = display["low_comp_share"].map(pct)
    display["Verified"] = display["verified_share"].map(pct)
    display["Fresh 30d"] = display["fresh_30d_share"].map(pct)
    display["Top problem"] = display["top_problem"]
    display["Reliability"] = display["reliability"]

    styled_table(
        display[["rank", "Niches", "Jobs", "Share", "Score", "Raw score", "Median hourly", "Median fixed",
                 "Low-comp share", "Verified", "Fresh 30d", "Top problem", "Reliability"]],
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "Niches": st.column_config.TextColumn("Niche", width="medium"),
            "Score": st.column_config.ProgressColumn("Adjusted score", min_value=0, max_value=100, format="%.1f", width="small"),
            "Raw score": st.column_config.NumberColumn("Raw score", format="%.1f", width="small"),
            "Jobs": st.column_config.NumberColumn("Jobs", width="small"),
            "Share": st.column_config.TextColumn("Demand share", width="small"),
            "Median hourly": st.column_config.TextColumn("Median hourly", width="small"),
            "Median fixed": st.column_config.TextColumn("Median fixed", width="small"),
            "Low-comp share": st.column_config.TextColumn("Low comp.", width="small"),
            "Verified": st.column_config.TextColumn("Verified", width="small"),
            "Fresh 30d": st.column_config.TextColumn("Fresh 30d", width="small"),
            "Top problem": st.column_config.TextColumn("Top problem", width="medium"),
            "Reliability": st.column_config.TextColumn("Sample", width="medium"),
        },
    )
    note(
        "A high score on a niche with fewer than five jobs is fragile. The 'Reliability' column makes sample size explicit — "
        "compare it with the raw score before choosing a specialisation.",
        kind="data",
    )

    st.divider()

    left, right = st.columns([1.25, 1])

    with left:
        section("Score composition", "Weighted contribution of each component to the raw score.")
        comp = table.sort_values("niche_score_raw", ascending=True).copy()
        parts = pd.DataFrame({
            "Niche": comp["niche"],
            "Demand (30%)": comp["D_demand"] * 0.30,
            "Economics (25%)": comp["E_economics"] * 0.25,
            "Competition (20%)": comp["C_competition"] * 0.20,
            "Clients (15%)": comp["Q_client"] * 0.15,
            "Freshness (10%)": comp["T_freshness"] * 0.10,
        }).set_index("Niche")
        colors = ["#4f6df5", "#0ea5e9", "#14b8a6", "#f59e0b", "#8b5cf6"]
        fig = go.Figure()
        for column, color in zip(parts.columns, colors):
            fig.add_bar(x=parts[column], y=parts.index, name=column, orientation="h", marker_color=color)
        fig.update_layout(barmode="stack")
        fig.update_xaxes(title="Points (sum = raw score)")
        fig.update_yaxes(title=None)
        show(fig, height=390, legend=True)

    with right:
        section("Adjusted vs. raw score", "The gap shows how strongly small samples were shrunk toward the prior.")
        comp = table.sort_values("niche_score", ascending=True)
        fig = go.Figure()
        fig.add_bar(
            x=comp["niche_score_raw"], y=comp["niche"], orientation="h",
            name="Raw", marker_color="#cbd5e1",
        )
        fig.add_bar(
            x=comp["niche_score"], y=comp["niche"], orientation="h",
            name="Adjusted", marker_color=[NICHE_COLORS.get(n, "#4f6df5") for n in comp["niche"]],
        )
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title="Score", range=[0, 100])
        fig.update_yaxes(title=None)
        show(fig, height=390, legend=True)

    st.divider()

    tab_econ, tab_problems, tab_drill = st.tabs(["Economics", "Problems", "Drill-down"])

    with tab_econ:
        c1, c2 = st.columns(2)
        with c1:
            section("Median hourly rate", "Only hourly contracts. Fixed-price projects are excluded by design.")
            hourly = table.dropna(subset=["median_hourly"]).sort_values("median_hourly")
            fig = px.bar(
                hourly, x="median_hourly", y="niche", orientation="h",
                color="niche", color_discrete_map=NICHE_COLORS,
                labels={"median_hourly": "Median $/h", "niche": ""},
            )
            fig.update_layout(showlegend=False)
            show(fig, height=330)
        with c2:
            section("Median fixed budget", "Only fixed-price contracts. Units are not blended with hourly rates.")
            fixed = table.dropna(subset=["median_fixed"]).sort_values("median_fixed")
            fig = px.bar(
                fixed, x="median_fixed", y="niche", orientation="h",
                color="niche", color_discrete_map=NICHE_COLORS,
                labels={"median_fixed": "Median budget ($)", "niche": ""},
            )
            fig.update_layout(showlegend=False)
            show(fig, height=330)

        c1, c2 = st.columns([1, 1])
        with c1:
            section("High-value and long-term share", "Share of jobs above the value threshold and share of >6-month contracts.")
            melt = table.melt(
                id_vars="niche", value_vars=["high_value_share", "long_term_share"],
                var_name="metric", value_name="value",
            )
            melt["metric"] = melt["metric"].map({
                "high_value_share": "High value", "long_term_share": "Long term (>6mo)",
            })
            fig = px.bar(
                melt, x="value", y="niche", color="metric", orientation="h",
                barmode="group", color_discrete_sequence=["#4f6df5", "#14b8a6"],
                labels={"value": "Share of jobs (%)", "niche": "", "metric": ""},
            )
            show(fig, height=360, legend=True)
        with c2:
            section("Client quality", "Payment verification and median visible client spend by niche.")
            q = table.sort_values("Q_client")
            fig = go.Figure()
            fig.add_bar(x=q["verified_share"], y=q["niche"], orientation="h", name="Payment verified %", marker_color="#0ea5e9")
            fig.add_bar(x=q["Q_client"], y=q["niche"], orientation="h", name="Composite client score", marker_color="#f59e0b")
            fig.update_layout(barmode="group")
            fig.update_xaxes(title="Score / share", range=[0, 100])
            fig.update_yaxes(title=None)
            show(fig, height=360, legend=True)

    with tab_problems:
        section("Niche × business problem", "Multi-label extraction: a job can appear under more than one problem. Darker cells mean repeated demand.")
        exploded = relevant.explode("problems")
        exploded = exploded[exploded["problems"].notna()]
        matrix = pd.crosstab(exploded["niche"], exploded["problems"])
        matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]
        matrix = matrix[matrix.sum(axis=0).sort_values(ascending=False).index]
        fig = go.Figure(go.Heatmap(
            z=matrix.values, x=matrix.columns, y=matrix.index,
            colorscale="Blues", text=matrix.values, texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z} jobs<extra></extra>",
            colorbar=dict(title="Jobs"),
        ))
        fig.update_xaxes(tickangle=-35)
        show(fig, height=430)

        section("Problem economics by niche")
        rows = []
        for niche, g in exploded.groupby("niche"):
            for problem, h in g.groupby("problems"):
                if len(h) < 1:
                    continue
                hourly = h.loc[h["job_type"].eq("Hourly"), "hourly_mid"].dropna()
                fixed = h.loc[h["job_type"].eq("Fixed price"), "fixed_budget"].dropna()
                rows.append({
                    "Niche": niche, "Problem": problem, "Jobs": len(h),
                    "Median hourly": money(hourly.median(), "/h") if len(hourly) else "n/a",
                    "Median fixed": money(fixed.median()) if len(fixed) else "n/a",
                    "Low-comp share": pct(h["low_competition"].mean() * 100),
                    "High-value share": pct(h["high_value"].mean() * 100),
                })
        detail = pd.DataFrame(rows).sort_values(["Jobs"], ascending=False)
        styled_table(detail, height=320)

    with tab_drill:
        section("Drill down: industry → sub-niche → jobs", "The classification is rule-based and confidence-scored. Verify borderline jobs manually.")
        industries = table["industry"].drop_duplicates().tolist()
        industry = st.selectbox("Industry", industries, key="niche_industry")
        subs = table.loc[table["industry"].eq(industry), "sub_niche"].tolist()
        sub = st.selectbox("Sub-niche", subs, key=f"niche_sub_{industry}")
        selected_niche = f"{industry} — {sub}"
        niche_jobs = relevant[relevant["niche"].eq(selected_niche)]
        row = table[table["niche"].eq(selected_niche)].iloc[0]

        kpi_row([
            ("Jobs", f"{int(row['jobs'])}", "Relevant unique jobs in this niche"),
            ("Adjusted score", f"{row['niche_score']:.1f}", f"Raw score {row['niche_score_raw']:.1f}"),
            ("Median hourly", money(row["median_hourly"], "/h"), "Hourly contracts only"),
            ("Low competition", pct(row["low_comp_share"]), "Share with 0–10 proposals at collection time"),
            ("Fresh 30d", pct(row["fresh_30d_share"]), "Posted within 30 days of the last scrape"),
        ])
        if row["jobs"] < 5:
            note(
                f"This niche has only {int(row['jobs'])} jobs. Treat the score as a hypothesis and expand data collection before specialising.",
                kind="warning",
            )
        job_table(niche_jobs.sort_values("job_score", ascending=False), n=25)
