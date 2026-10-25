"""Page 5 — Data Quality & Collection Coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as a
from views.common import (
    MUTED, NICHE_COLORS, kpi_row, money, no_data, note, pct, section, show,
    styled_table,
)

CRITICAL_FIELDS = [
    "published_at", "posted_relative", "hourly_min", "hourly_max", "fixed_budget",
    "duration", "workload", "proposals", "client_total_spent", "client_reviews",
    "client_rating", "skills", "description", "client_country", "client_payment_verified",
]


def render(df: pd.DataFrame, full: pd.DataFrame) -> None:
    st.title("Data Quality & Collection Coverage")
    st.caption("Before trusting any insight, check whether it is an artifact of incomplete, duplicated or biased collection.")

    if df.empty:
        no_data()
        return

    dq = a.data_quality(df)
    kpi_row([
        ("Rows / unique jobs", f"{dq['rows']:,} / {dq['unique_jobs']:,}", "Rows are de-duplicated by job_id before analysis."),
        ("Unclassified", pct(dq["unclassified_share"]), f"{dq['unclassified']} jobs did not match any target niche."),
        ("Exact dates", pct((dq["exact_dates"] / dq["rows"] * 100) if dq["rows"] else 0), f"{dq['estimated_dates']} dates are estimated from relative strings; {dq['missing_dates']} missing."),
        ("Missing economics", pct(dq["missing_economics_share"]), f"{dq['missing_economics']} jobs have neither a usable hourly rate nor fixed budget."),
        ("Search queries", f"{dq['search_queries']:,}", f"Labels: {dq['search_labels'] or 'n/a'}. More queries = wider market coverage."),
    ])

    st.divider()
    note(
        "**Search bias alert.** This dataset was collected with a single query and location filter. It is not the whole "
        "FinTech/Crypto market — it is what that query surfaced. Treat every ranking as a property of this sample, not the market.",
        kind="warning",
    )
    if "search_query" in df:
        queries = df[["search_query", "search_label", "search_filters"]].drop_duplicates()
        with st.expander("Collection parameters", expanded=True):
            for _, row in queries.iterrows():
                st.markdown(f"**Query:** `{row['search_query']}` · **Label:** `{row['search_label']}`")
                st.code(str(row["search_filters"]), language=None)
            st.caption(f"Last scrape: {dq['last_scrape']:%Y-%m-%d %H:%M UTC} · Published range: {dq['date_min']:%Y-%m-%d} → {dq['date_max']:%Y-%m-%d}")

    absent = [n for n in a.NICHE_RULES
              if not any(r.startswith(n) for r in df.loc[df["relevant"], "niche"].unique())]

    st.divider()
    c1, c2 = st.columns([1, 1])

    with c1:
        section("Date provenance", "Only a small share of dates are exact timestamps. Time-series conclusions are directional.")
        sources = df["date_source"].value_counts().reset_index()
        sources.columns = ["source", "jobs"]
        fig = px.pie(
            sources, names="source", values="jobs", hole=0.55,
            color_discrete_sequence=["#4f6df5", "#f59e0b", "#94a3b8"],
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False)
        show(fig, height=310)

    with c2:
        section("Classification confidence", "Rule-based niche match confidence (0–100). Low-confidence jobs deserve manual review.")
        conf = df[["niche_confidence", "relevant"]].copy()
        fig = px.histogram(
            conf, x="niche_confidence", color="relevant", nbins=20,
            color_discrete_map={True: "#4f6df5", False: "#cbd5e1"},
            labels={"niche_confidence": "Confidence", "relevant": "Relevant"},
        )
        fig.update_layout(barmode="overlay")
        fig.update_traces(opacity=0.8)
        fig.update_yaxes(title="Jobs")
        show(fig, height=310, legend=True)

    st.divider()
    c1, c2 = st.columns([1, 1])

    with c1:
        section("Critical field completeness", "Share of missing values for the fields the analysis depends on.")
        rows = []
        for field in CRITICAL_FIELDS:
            if field not in df:
                continue
            missing = df[field].isna().sum()
            rows.append({"field": field, "missing": missing, "missing_share": missing / len(df) * 100})
        crit = pd.DataFrame(rows).sort_values("missing_share", ascending=True)
        fig = px.bar(
            crit, x="missing_share", y="field", orientation="h",
            color="missing_share", color_continuous_scale="OrRd", range_color=[0, 100],
            labels={"missing_share": "Missing (%)", "field": ""},
            text=crit["missing_share"].map(lambda v: f"{v:.0f}%"),
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(range=[0, 108])
        show(fig, height=420)

    with c2:
        section("Coverage by niche", "Niche counts reveal which target areas the collection barely touched.")
        relevant = df[df["relevant"]]
        counts = relevant.groupby("niche").size().reset_index(name="jobs").sort_values("jobs")
        if counts.empty:
            st.caption("No relevant jobs.")
        else:
            fig = px.bar(
                counts, x="jobs", y="niche", orientation="h",
                color="niche", color_discrete_map=NICHE_COLORS, text="jobs",
                labels={"jobs": "Jobs", "niche": ""},
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False)
            show(fig, height=420)

        if absent:
            st.caption("Target industries with **zero** classified jobs in this selection: " + ", ".join(absent))

    st.divider()
    section("Data integrity", "Duplicate identifiers, duplicate titles and missing critical values.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duplicate job IDs", f"{dq['duplicate_ids']:,}", help="Should be zero after de-duplication.")
    c2.metric("Duplicate titles", f"{dq['duplicate_titles']:,}", help="Could be re-posts or genuinely similar roles.")
    c3.metric("Unknown proposals", f"{dq['unknown_proposals']:,}", help="Proposal count could not be parsed into a band.")
    c4.metric("Low confidence", f"{dq['low_confidence_jobs']:,}", help="Jobs with classification confidence below 40.")

    duplicates = df[df["title"].duplicated(keep=False)].sort_values("title")
    if not duplicates.empty:
        with st.expander(f"Duplicate titles ({duplicates['title'].nunique()} title groups)", expanded=False):
            styled_table(
                duplicates[["title", "job_id", "posted_date", "url"]].assign(
                    posted_date=duplicates["posted_date"].dt.strftime("%Y-%m-%d")
                ).rename(columns={"posted_date": "posted", "job_id": "job id", "url": "link"}),
                column_config={"link": st.column_config.LinkColumn("Link", display_text="Open ↗")},
            )

    st.divider()
    section("Unclassified jobs", "These did not match a target niche above the relevance threshold. They may be genuine gaps — or noise from the broad search query.")
    unclassified = df[~df["relevant"]].sort_values("niche_confidence", ascending=False)
    view = unclassified[["title", "industry", "problem_primary", "rate_label", "client_country", "url"]].copy()
    view["posted"] = unclassified["posted_date"].dt.strftime("%Y-%m-%d").fillna("unknown")
    view = view.rename(columns={
        "title": "Job", "industry": "Industry", "problem_primary": "Problem",
        "rate_label": "Rate / budget", "client_country": "Country", "url": "Link",
    })
    styled_table(
        view[["Job", "Industry", "Problem", "Rate / budget", "posted", "Country", "Link"]],
        column_config={"Link": st.column_config.LinkColumn("Link", display_text="Open ↗")},
        height=340,
    )

    st.divider()
    section("Coverage checklist", "What to collect next to make the insights trustworthy.")
    checks = []
    if dq["search_queries"] <= 1:
        checks.append(("Search coverage", "Add the specialised queries from the brief (payments, reconciliation, on-chain, DeFi, AML, fraud…). One query cannot represent FinTech, Finance and Crypto."))
    if dq["exact_dates"] / max(dq["rows"], 1) < 0.5:
        checks.append(("Date precision", "Capture absolute publication timestamps. Relative strings make 7/30/90-day growth unreliable."))
    if dq["missing_economics_share"] > 10:
        checks.append(("Budget fields", f"{dq['missing_economics_share']:.0f}% of jobs lack usable economics. Do not treat unknown budgets as zero."))
    if dq["unclassified_share"] > 40:
        checks.append(("Classification", "A large share of jobs is unclassified. Review borderline cases and expand the taxonomy keywords."))
    if absent:
        checks.append(("Niche coverage", "No classified jobs for: " + ", ".join(absent) + ". Absence may be a collection gap, not low demand."))
    if duplicated_titles := dq["duplicate_titles"]:
        checks.append(("Duplicates", f"{duplicated_titles} duplicate titles found. Confirm they are not reposts inflating demand."))
    checks.append(("Snapshots", "One scrape is a snapshot. Re-collect the same job over time to measure proposal velocity and post/renew behaviour."))

    for title, text in checks:
        st.markdown(f"- **{title}:** {text}")

    st.divider()
    section("Export", "Download the filtered, enriched dataset (classification, scores and derived fields) for further analysis.")
    export_cols = [
        "job_id", "title", "url", "posted_date", "date_source", "job_type", "hourly_min",
        "hourly_max", "fixed_budget", "economic_value", "proposals", "proposal_bucket",
        "duration", "workload", "niche", "relevant", "niche_confidence", "problem_primary",
        "service", "job_score", "client_country", "client_payment_verified",
        "client_total_spent", "client_spent", "client_reviews", "client_rating",
    ]
    existing = [c for c in export_cols if c in df]
    csv = df[existing].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        data=csv,
        file_name="upwork_market_filtered.csv",
        mime="text/csv",
        width="content",
    )
    with st.expander("All-column completeness"):
        miss = a.missingness(df.drop(columns=[c for c in df.columns if c.startswith("score_")], errors="ignore"))
        miss["missing_share"] = miss["missing_share"].round(1)
        styled_table(
            miss.rename(columns={"field": "Field", "missing": "Missing", "missing_share": "Missing %", "populated": "Populated"}),
            column_config={"Missing %": st.column_config.NumberColumn(format="%.1f")},
            height=400,
        )
