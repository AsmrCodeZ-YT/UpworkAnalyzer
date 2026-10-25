"""Page 3 — Project & Service Discovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as a
from views.common import (
    MUTED, PALETTE, job_table, kpi_row, money, no_data, note,
    pct, section, show, styled_table,
)


def _service_table(df: pd.DataFrame) -> pd.DataFrame:
    table = a.build_problem_table(df)
    if table.empty:
        return table
    out = table.copy()
    out["Jobs"] = out["jobs"]
    out["Median hourly"] = out["median_hourly"].map(lambda v: money(v, "/h"))
    out["Median fixed"] = out["median_fixed"].map(money)
    out["Low-comp"] = out["low_comp_share"].map(pct)
    out["High-value"] = out["high_value_share"].map(pct)
    out["Score"] = out["median_job_score"].round(1)
    return out


def render(df: pd.DataFrame, full: pd.DataFrame) -> None:
    st.title("Project & Service Discovery")
    st.caption("Turn repeated client problems into concrete, sellable services — then inspect the real jobs behind each cluster.")

    relevant = df[df["relevant"]].copy()
    if relevant.empty:
        no_data()
        return

    service_table = _service_table(df)
    if service_table.empty:
        no_data()
        return

    skill_table = a.build_skill_table(df)
    kpi_row([
        ("Service archetypes", f"{len(service_table):,}", "Distinct business problems found in the filtered relevant jobs."),
        ("Top service", service_table.iloc[0]["service"], f"{int(service_table.iloc[0]['Jobs'])} matching jobs."),
        ("Problems / job", f"{relevant['problems'].map(len).mean():.1f}", "Average number of extracted business problems per relevant job."),
        ("Top skill", (skill_table.iloc[0]["skill"] if not skill_table.empty else "n/a"), "Most frequent extracted skill in the current selection."),
        ("Low-competition share", pct(relevant["low_competition"].mean() * 100), "Jobs with 0–10 proposals when collected."),
    ])

    st.divider()
    section(
        "Problem → service map",
        "Each row maps a repeated client problem to a productised service. Click a job link in the explorer below to inspect the evidence.",
    )
    styled_table(
        service_table[["problem", "service", "Jobs", "Median hourly", "Median fixed", "Low-comp", "High-value", "Score"]],
        column_config={
            "problem": st.column_config.TextColumn("Business problem", width="medium"),
            "service": st.column_config.TextColumn("Sellable service", width="large"),
            "Jobs": st.column_config.NumberColumn("Jobs", width="small"),
            "Median hourly": st.column_config.TextColumn("Median hourly", width="small"),
            "Median fixed": st.column_config.TextColumn("Median fixed", width="small"),
            "Low-comp": st.column_config.TextColumn("Low comp.", width="small"),
            "High-value": st.column_config.TextColumn("High value", width="small"),
            "Score": st.column_config.ProgressColumn("Median job score", min_value=0, max_value=100, format="%.0f", width="small"),
        },
    )
    with st.expander("What each service includes", expanded=False):
        for _, row in service_table.iterrows():
            st.markdown(f"**{row['service']}** · _{row['problem']}_")
            st.caption(row["service_description"])

    st.divider()
    section("Service demand × budget", "Bubble size = jobs; colour = low-competition share. Hourly and fixed-price economics are shown separately because they are different units.")
    tab_h, tab_f = st.tabs(["Hourly services", "Fixed-price services"])

    with tab_h:
        hourly = service_table.dropna(subset=["median_hourly"]).copy()
        if hourly.empty:
            st.caption("No hourly-priced services in this selection.")
        else:
            hourly["fixed_text"] = hourly["median_fixed"].map(lambda v: money(v) if pd.notna(v) else "n/a")
            fig = px.scatter(
                hourly, x="Jobs", y="median_hourly", size="Jobs", color="low_comp_share",
                hover_name="service", color_continuous_scale="RdYlGn",
                range_color=[0, 100], size_max=52,
                labels={"median_hourly": "Median hourly rate ($/h)", "low_comp_share": "Low-comp share (%)"},
                custom_data=["problem", "Low-comp", "fixed_text"],
            )
            fig.update_traces(
                marker=dict(line=dict(color="white", width=1.5)),
                hovertemplate=(
                    "<b>%{hovertext}</b><br>%{customdata[0]}<br>%{x} jobs · %{y:.0f} $/h<br>"
                    "Low competition: %{customdata[1]}<br>Median fixed: %{customdata[2]}<extra></extra>"
                ),
            )
            show(fig, height=390)

    with tab_f:
        fixed = service_table.dropna(subset=["median_fixed"]).copy()
        if fixed.empty:
            st.caption("No fixed-price services with a known budget in this selection.")
        else:
            fixed["hourly_text"] = fixed["median_hourly"].map(lambda v: money(v, "/h") if pd.notna(v) else "n/a")
            fig = px.scatter(
                fixed, x="Jobs", y="median_fixed", size="Jobs", color="low_comp_share",
                hover_name="service", color_continuous_scale="RdYlGn",
                range_color=[0, 100], size_max=52,
                labels={"median_fixed": "Median fixed budget ($)", "low_comp_share": "Low-comp share (%)"},
                custom_data=["problem", "Low-comp", "hourly_text"],
            )
            fig.update_traces(
                marker=dict(line=dict(color="white", width=1.5)),
                hovertemplate=(
                    "<b>%{hovertext}</b><br>%{customdata[0]}<br>%{x} jobs · $%{y:,.0f}<br>"
                    "Low competition: %{customdata[1]}<br>Median hourly: %{customdata[2]}<extra></extra>"
                ),
            )
            show(fig, height=390)

    st.divider()
    tab_skills, tab_phrases, tab_clusters = st.tabs(["Skills", "Phrases", "Text clusters"])

    with tab_skills:
        skills = a.build_skill_table(df, min_count=2)
        if skills.empty:
            st.caption("Not enough skill data.")
        else:
            top = skills.head(18).sort_values("jobs")
            fig = px.bar(
                top, x="jobs", y="skill", orientation="h",
                color="low_comp_share", color_continuous_scale="RdYlGn", range_color=[0, 100],
                labels={"jobs": "Jobs", "skill": "", "low_comp_share": "Low-comp share (%)"},
            )
            fig.update_layout(coloraxis_colorbar=dict(title="Low comp.<br>share (%)"))
            show(fig, height=440)

            c1, c2 = st.columns([1, 1])
            with c1:
                section("Skill economics")
                sk = skills.copy()
                sk["Median hourly"] = sk["median_hourly"].map(lambda v: money(v, "/h"))
                sk["Median fixed"] = sk["median_fixed"].map(money)
                sk["Low-comp"] = sk["low_comp_share"].map(pct)
                sk["Verified"] = sk["verified_share"].map(pct)
                sk["Score"] = sk["median_job_score"].round(1)
                styled_table(
                    sk[["skill", "jobs", "relevant_jobs", "Median hourly", "Median fixed", "Low-comp", "Verified", "Score"]].head(25),
                    column_config={
                        "skill": st.column_config.TextColumn("Skill", width="medium"),
                        "jobs": st.column_config.NumberColumn("Jobs", width="small"),
                        "relevant_jobs": st.column_config.NumberColumn("Relevant", width="small"),
                    },
                    height=420,
                )
            with c2:
                section("Skill co-occurrence", "How often two skills appear in the same job (diagonal = total jobs).")
                names, matrix = a.build_skill_cooccurrence(df, top_n=12)
                if names:
                    fig = go.Figure(go.Heatmap(
                        z=matrix, x=names, y=names, colorscale="Purples",
                        hovertemplate="%{y} + %{x}<br>%{z} jobs<extra></extra>",
                    ))
                    fig.update_xaxes(tickangle=-40)
                    show(fig, height=420)

    with tab_phrases:
        section("Recurring phrases", "TF-IDF ranked 2–3 word phrases from titles and descriptions. Median economics are shown for the jobs containing each phrase.")
        phrases = a.extract_phrases(df, ngram=(2, 3), top_n=25)
        if phrases.empty:
            st.caption("Not enough repeated phrases in this selection.")
        else:
            view = phrases.copy()
            view["median_hourly"] = view["median_hourly"].map(lambda v: money(v, "/h") if pd.notna(v) else "n/a")
            view["median_fixed"] = view["median_fixed"].map(money)
            view["low_comp_share"] = view["low_comp_share"].map(pct)
            view["tfidf"] = view["tfidf"].round(3)
            styled_table(
                view.rename(columns={
                    "phrase": "Phrase", "jobs": "Jobs", "tfidf": "Mean TF-IDF",
                    "median_hourly": "Median hourly", "median_fixed": "Median fixed",
                    "low_comp_share": "Low comp.",
                }),
                column_config={"Mean TF-IDF": st.column_config.NumberColumn(format="%.3f")},
                height=420,
            )

    with tab_clusters:
        section("Unsupervised project clusters", "TF-IDF + K-Means over titles, descriptions and skills. Clusters can reveal service themes that the rule-based taxonomy does not name.")
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            n_clusters = st.slider("Clusters", 2, 8, 5, key="cluster_n")
        with c2:
            scope = st.selectbox("Scope", ["Relevant jobs", "All filtered jobs"], key="cluster_scope")
        with c3:
            st.caption("Cluster names are generated from the strongest terms, not hand-labelled. Small samples can produce unstable clusters.")

        cluster_scope = relevant if scope == "Relevant jobs" else df
        cluster_df, work = a.cluster_jobs(cluster_scope, n_clusters=n_clusters)

        if cluster_df.empty or work.empty:
            st.caption("Not enough descriptive text to cluster.")
        else:
            view = cluster_df.copy()
            view["median_hourly"] = view["median_hourly"].map(lambda v: money(v, "/h"))
            view["median_fixed"] = view["median_fixed"].map(money)
            view["low_comp_share"] = view["low_comp_share"].map(pct)
            styled_table(
                view[["cluster", "jobs", "median_hourly", "median_fixed", "low_comp_share", "fresh_jobs", "top_problems", "sample_titles"]],
                column_config={
                    "cluster": st.column_config.TextColumn("Cluster (top terms)", width="medium"),
                    "jobs": st.column_config.NumberColumn("Jobs", width="small"),
                    "median_hourly": st.column_config.TextColumn("Median hourly", width="small"),
                    "median_fixed": st.column_config.TextColumn("Median fixed", width="small"),
                    "low_comp_share": st.column_config.TextColumn("Low comp.", width="small"),
                    "fresh_jobs": st.column_config.NumberColumn("Fresh 30d", width="small"),
                    "top_problems": st.column_config.TextColumn("Top problems", width="medium"),
                    "sample_titles": st.column_config.TextColumn("Sample titles", width="large"),
                },
            )
            cluster_labels = cluster_df["cluster"].tolist()
            selected = st.selectbox("Inspect cluster", cluster_labels, key=f"cluster_select_{scope}_{n_clusters}")
            chosen_id = cluster_df.loc[cluster_df["cluster"].eq(selected), "cluster_id"].iloc[0]
            jobs = work[work["cluster_id"].eq(chosen_id)]
            job_table(jobs.sort_values("job_score", ascending=False), show_score=True)

    st.divider()
    section("Job explorer", "Search the filtered jobs and inspect the briefs behind the services and clusters above.")
    query = st.text_input("Search title, description, skills or niche", placeholder="e.g. reconciliation, dbt, on-chain…")
    explore = relevant.copy()
    if query.strip():
        needle = query.strip().lower()
        mask = (
            explore["title"].str.lower().str.contains(needle, regex=False)
            | explore["description"].str.lower().str.contains(needle, regex=False)
            | explore["skills"].fillna("").str.lower().str.contains(needle, regex=False)
            | explore["niche"].str.lower().str.contains(needle, regex=False)
        )
        explore = explore[mask]
    sort_key = st.selectbox("Sort by", ["Job score", "Newest", "Median rate / budget", "Fewest proposals"], index=0)
    if sort_key == "Newest":
        explore = explore.sort_values("days_old", ascending=True, na_position="last")
    elif sort_key == "Median rate / budget":
        explore = explore.sort_values("economic_value", ascending=False, na_position="last")
    elif sort_key == "Fewest proposals":
        explore = explore.sort_values("proposal_mid", ascending=True, na_position="last")
    else:
        explore = explore.sort_values("job_score", ascending=False, na_position="last")
    st.caption(f"{len(explore)} matching relevant jobs.")
    job_table(explore, n=40)
    note(
        "Service suggestions are derived from the problem taxonomy, not from a trained language model. "
        "Validate them against the sample job briefs before packaging an offer.",
        kind="data",
    )
