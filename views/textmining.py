"""Page 6 — Text Mining & Keyword Intelligence."""

from __future__ import annotations

import html
import re

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import textmining as tm
from views.common import (
    MUTED, NICHE_COLORS, PALETTE, job_table, kpi_row, no_data, note,
    pct, section, show, styled_table,
)

ALL_NODES = "All terms"
METRIC_COLUMNS = {
    "Co-occurrence count": "jobs_with_both",
    "Conditional probability": "confidence_a_to_b",
    "Lift": "lift",
    "Jaccard similarity": "jaccard",
}
PRESETS = [
    "Finance vs All other jobs",
    "Crypto vs Finance",
    "DeFi vs Other crypto jobs",
    "Payments vs General Data Analytics",
    "Custom comparison",
]


def _safe_select(key: str, options: list[str], label: str, **kwargs):
    if st.session_state.get(key) not in options:
        st.session_state.pop(key, None)
    return st.selectbox(label, options, key=key, **kwargs)


def _safe_multi(key: str, options: list[str], label: str, default: list[str], **kwargs):
    current = st.session_state.get(key)
    if current is not None and any(value not in options for value in current):
        st.session_state.pop(key, None)
    clean_default = [value for value in default if value in options]
    return st.multiselect(label, options, default=clean_default, key=key, **kwargs)


def _period_totals(work: pd.DataFrame, window_days: int) -> tuple[int, int]:
    dates = pd.to_datetime(work["posted_date"], errors="coerce")
    latest = dates.max()
    if pd.isna(latest):
        return 0, 0
    current = int(((dates > latest - pd.to_timedelta(window_days, unit="D")) & (dates <= latest)).sum())
    previous = int((
        (dates > latest - pd.to_timedelta(2 * window_days, unit="D"))
        & (dates <= latest - pd.to_timedelta(window_days, unit="D"))
    ).sum())
    return current, previous


def _highlight(text: str, phrase: str) -> str:
    escaped = html.escape(str(text))
    needle = re.escape(html.escape(phrase))
    return re.sub(
        f"({needle})",
        r'<mark style="background:#fde68a;padding:0 2px;border-radius:3px;">\1</mark>',
        escaped,
        flags=re.IGNORECASE,
    )


def _short(label: str, limit: int = 24) -> str:
    return label if len(label) <= limit else label[: limit - 1] + "…"


def _trend_text(row) -> str:
    if row["trend_status"] == "New":
        return "new"
    if pd.notna(row["trend_change_pct"]):
        return f"{row['trend_change_pct']:+.0f}%"
    return "n/a"


def _date_text(value) -> str:
    return value.strftime("%Y-%m-%d") if pd.notna(value) else "n/a"


def _network_figure(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    metric: str,
    focus: str,
    scope: str,
    color_mode: str,
) -> go.Figure | None:
    if edges.empty:
        return None
    graph = nx.Graph()
    for row in edges.itertuples():
        graph.add_edge(row.phrase_a, row.phrase_b, weight=float(getattr(row, metric)))
    if graph.number_of_edges() == 0:
        return None
    positions = nx.spring_layout(
        graph, seed=42, k=0.9 / max(np.sqrt(graph.number_of_nodes()), 1),
        iterations=120, weight="weight",
    )

    node_view = nodes[nodes["node"].isin(graph.nodes)].copy()
    node_view = node_view.set_index("node")
    if color_mode == "Co-occurrence community":
        try:
            communities = nx.community.louvain_communities(graph, weight="weight", seed=42)
        except Exception:
            communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
        community_of = {}
        for index, community in enumerate(communities):
            for node in community:
                community_of[node] = f"Community {index + 1}"
        node_view["_group"] = [community_of.get(node, "Community 1") for node in node_view.index]
        group_colors = {f"Community {i + 1}": PALETTE[i % len(PALETTE)] for i in range(len(communities))}
    elif scope == "Skill → business problem":
        node_view["_group"] = node_view["node_type"]
        group_colors = {"Skill": "#4f6df5", "Business problem": "#f59e0b"}
    else:
        node_view["_group"] = node_view["top_niche"].fillna("n/a")
        group_colors = {group: NICHE_COLORS.get(group, MUTED) for group in node_view["_group"].unique()}

    metric_values = edges[metric].astype(float)
    low, high = float(metric_values.min()), float(metric_values.max())
    fig = go.Figure()
    for row in edges.itertuples():
        if row.phrase_a not in positions or row.phrase_b not in positions:
            continue
        x0, y0 = positions[row.phrase_a]
        x1, y1 = positions[row.phrase_b]
        connected = focus in (ALL_NODES, row.phrase_a, row.phrase_b)
        if focus == ALL_NODES:
            color = "rgba(148,163,184,0.45)"
        elif connected:
            color = "rgba(79,109,245,0.75)"
        else:
            color = "rgba(148,163,184,0.18)"
        width = 1.0 + 3.2 * ((float(getattr(row, metric)) - low) / max(high - low, 1e-9))
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color=color, width=width),
            text=(
                f"{row.phrase_a} ↔ {row.phrase_b}<br>"
                f"{row.jobs_with_both} jobs together<br>"
                f"Confidence A→B: {row.confidence_a_to_b:.0f}%<br>"
                f"Lift: {row.lift:.2f}× · Jaccard: {row.jaccard:.0f}%"
            ),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

    for group, subset in node_view.groupby("_group", sort=False):
        xs = [positions[node][0] for node in subset.index]
        ys = [positions[node][1] for node in subset.index]
        jobs = subset["jobs"].astype(float).to_numpy()
        line_width = [
            3.0 if focus != ALL_NODES and node == focus else 1.4 for node in subset.index
        ]
        line_color = [
            "#0f172a" if focus != ALL_NODES and node == focus else "white" for node in subset.index
        ]
        custom = [
            [node, int(subset.loc[node, "jobs"]), f"Group: {group}", str(subset.loc[node, "top_niche"])]
            for node in subset.index
        ]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            name=group,
            text=[_short(node) for node in subset.index],
            textposition="top center",
            textfont=dict(size=9, color="#334155"),
            marker=dict(
                size=11 + 2.8 * np.sqrt(jobs),
                color=group_colors.get(group, PALETTE[0]),
                line=dict(color=line_color, width=line_width),
                opacity=0.92,
            ),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[1]} jobs<br>"
                "%{customdata[2]}<br>Top niche: %{customdata[3]}<extra></extra>"
            ),
        ))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(hovermode="closest")
    return fig


def render(df: pd.DataFrame, full: pd.DataFrame) -> None:
    st.title("Text Mining & Keyword Intelligence")
    st.caption(
        "The language clients use in briefs: frequent terms, meaningful phrases, niche-specific wording, "
        "co-occurrence, trends and the raw sentences behind every match."
    )

    if df.empty:
        no_data()
        return

    max_jobs = max(2, min(20, len(df)))
    with st.container(border=True):
        row1 = st.columns([1.15, 1.55, 1.0, 1.0])
        with row1[0]:
            corpus_label = st.selectbox(
                "Corpus", ["Relevant jobs", "All filtered jobs"], key="tm_corpus",
                help="Relevant jobs only keeps the FinTech / Finance / Crypto classification; all jobs adds general analytics.",
            )
        with row1[1]:
            source_label = st.selectbox(
                "Text source", list(tm.SOURCE_OPTIONS), key="tm_source",
                help="Titles, descriptions and skills are mined separately because they carry different intent.",
            )
        with row1[2]:
            length_label = st.selectbox("Phrase length", list(tm.LENGTH_OPTIONS), key="tm_length")
        with row1[3]:
            min_jobs = st.slider(
                "Min jobs per term", 1, max_jobs, min(2, max_jobs), key="tm_min_jobs",
                help="Terms must appear in at least this many unique jobs.",
            )
        row2 = st.columns([1.0, 2.7])
        with row2[0]:
            window_label = st.selectbox("Trend window", ["7 days", "14 days", "30 days"], key="tm_window")
        with row2[1]:
            query = st.text_input(
                "Keyword search", placeholder="e.g. reconciliation, on-chain, dbt, risk…",
                key="tm_query",
            )
        st.caption(
            "All analysis runs on unique job IDs after the sidebar filters. "
            "Counts are document-level, not raw word repetitions, unless a column says otherwise."
        )

    relevant_only = corpus_label == "Relevant jobs"
    window_days = int(window_label.split()[0])
    work = tm.build_corpus(df, relevant_only=relevant_only, source=tm.SOURCE_OPTIONS[source_label])
    if work.empty:
        st.warning("No documents left for this corpus and text source. Try All filtered jobs or another text source.")
        return
    space = tm.build_term_space(work)
    if space is None:
        st.warning("No usable terms after tokenisation and stopword removal.")
        return

    table = tm.term_table(space, window_days=window_days)
    filtered = tm.filter_terms(table, min_jobs, tm.LENGTH_OPTIONS[length_label], query)
    current_total, previous_total = _period_totals(work, window_days)

    phrases = filtered[filtered["phrase_length"].between(2, 3)]
    emerging = filtered[
        filtered["coverage_change_pp"].gt(0)
        & filtered["current_job_count"].ge(1)
        & filtered["trend_status"].notna()
    ]
    kpi_row([
        ("Documents analyzed", f"{len(work):,}", f"Unique jobs after the sidebar filters. Corpus: {corpus_label.lower()}."),
        ("Unique terms", f"{int((table['phrase_length'] == 1).sum()):,}", "Distinct one-word vocabulary items after stopword removal."),
        ("Top phrases", f"{len(phrases):,}", f"Two- and three-word phrases with at least {min_jobs} jobs. Sort them below by NPMI."),
        ("Emerging terms", f"{len(emerging):,}", f"Terms with higher coverage in the latest {window_days} days than in the previous {window_days} days."),
    ])

    note(
        "Frequency is not commercial value. Generic words such as data, analyst or reporting can dominate counts; "
        "use distinctiveness, NPMI, context and the niche comparison before packaging a service. "
        "The corpus is small and dates are partly estimated — treat trends as directional.",
        kind="warning",
    )

    st.divider()
    section(
        "Keyword frequency",
        "Sort the table by any column. Job coverage is the share of analyzed jobs containing the term; distinctiveness rewards terms concentrated in one niche.",
    )
    if filtered.empty:
        st.warning("No terms match the current minimum job count and keyword search.")
        return

    freq = filtered.head(300).copy()
    freq["first_seen_at"] = freq["first_seen_at"].map(_date_text)
    freq["last_seen_at"] = freq["last_seen_at"].map(_date_text)
    freq_view = freq.rename(columns={
        "phrase": "Keyword", "phrase_length": "Words", "total_occurrences": "Occurrences",
        "job_count": "Jobs", "job_coverage_pct": "Job coverage", "npmi": "NPMI",
        "distinctiveness_score": "Distinctiveness", "top_niche": "Top niche",
        "first_seen_at": "First seen", "last_seen_at": "Last seen",
    })
    styled_table(
        freq_view[[
            "Keyword", "Words", "Occurrences", "Jobs", "Job coverage", "NPMI",
            "Distinctiveness", "Top niche", "First seen", "Last seen",
        ]],
        column_config={
            "Keyword": st.column_config.TextColumn("Keyword", width="medium"),
            "Words": st.column_config.NumberColumn("Words", width="small"),
            "Occurrences": st.column_config.NumberColumn("Occurrences", width="small"),
            "Jobs": st.column_config.NumberColumn("Jobs", width="small"),
            "Job coverage": st.column_config.NumberColumn("Job coverage", format="%.1f%%", width="small"),
            "NPMI": st.column_config.NumberColumn("NPMI", format="%.2f", width="small", help="Normalised pointwise mutual information; blank for single words."),
            "Distinctiveness": st.column_config.ProgressColumn("Distinctiveness", min_value=0, max_value=100, format="%.0f", width="small"),
            "Top niche": st.column_config.TextColumn("Top niche", width="medium"),
            "First seen": st.column_config.TextColumn("First seen", width="small"),
            "Last seen": st.column_config.TextColumn("Last seen", width="small"),
        },
        height=430,
    )

    section("Word cloud", "A quick orientation view, not the decision layer. The table beside it carries the real values.")
    cloud_row = st.columns([1.0, 1.0])
    with cloud_row[0]:
        cloud_mode = st.radio(
            "Size words by", ["Frequency", "Document coverage", "Distinctiveness"],
            horizontal=True, key="tm_cloud_mode",
        )
    with cloud_row[1]:
        cloud_scope = st.selectbox(
            "Cloud scope", ["All filtered jobs"] + sorted(work["industry"].unique()),
            key="tm_cloud_scope",
        )

    cloud_work = work if cloud_scope == "All filtered jobs" else work[work["industry"].eq(cloud_scope)]
    cloud_table = pd.DataFrame()
    if cloud_work.empty:
        st.caption("No jobs in this cloud scope.")
    else:
        cloud_space = tm.build_term_space(cloud_work)
        if cloud_space is not None:
            cloud_table = tm.filter_terms(
                tm.term_table(cloud_space, window_days), min_jobs,
                tm.LENGTH_OPTIONS[length_label], query,
            )
    if cloud_table.empty:
        st.caption("Not enough terms for a word cloud in this scope. Lower the minimum job count.")
    else:
        effective_mode = cloud_mode
        if cloud_mode == "Distinctiveness" and cloud_table["distinctiveness_score"].max() <= 0:
            effective_mode = "Document coverage"
            st.caption("No over-represented terms in this scope; using document coverage for the cloud.")
        image = tm.wordcloud_image(cloud_table, effective_mode, max_words=100)
        left, right = st.columns([1.25, 1.0])
        with left:
            if image is None:
                st.caption("The word cloud could not be generated for this selection.")
            else:
                st.image(image, width="stretch")
        with right:
            weighted = tm.weighted_terms(cloud_table, effective_mode).nlargest(30, "weight")
            cloud_view = weighted[["phrase", "job_count", "total_occurrences", "weight"]].rename(columns={
                "phrase": "Term", "job_count": "Jobs", "total_occurrences": "Occurrences",
                "weight": "Weight",
            })
            styled_table(
                cloud_view,
                column_config={
                    "Term": st.column_config.TextColumn("Term", width="medium"),
                    "Jobs": st.column_config.NumberColumn("Jobs", width="small"),
                    "Occurrences": st.column_config.NumberColumn("Occurrences", width="small"),
                    "Weight": st.column_config.NumberColumn("Weight", format="%.1f", width="small"),
                },
                height=380,
            )
        st.caption(f"Cloud scope: {cloud_scope}. Size metric: {effective_mode.lower()}.")

    st.divider()
    section(
        "Phrase mining",
        "Phrases are more specific than single words. NPMI tests whether words co-occur more often than chance; frequency and TF-IDF keep the table grounded.",
    )
    if phrases.empty:
        st.caption("No two- or three-word phrases meet the current filters.")
    else:
        phrase_sort = st.selectbox(
            "Sort phrases by", ["Frequency", "NPMI (collocation)", "Mean TF-IDF"],
            key="tm_phrase_sort",
        )
        phrase_view = phrases.copy()
        if phrase_sort == "NPMI (collocation)":
            phrase_view = phrase_view.dropna(subset=["npmi"]).sort_values(
                ["npmi", "job_count"], ascending=[False, False]
            )
        elif phrase_sort == "Mean TF-IDF":
            phrase_view = phrase_view.sort_values(["avg_tfidf", "job_count"], ascending=False)
        else:
            phrase_view = phrase_view.sort_values(["job_count", "total_occurrences"], ascending=False)
        phrase_view = phrase_view.head(100).copy()
        phrase_view["Trend"] = phrase_view.apply(_trend_text, axis=1)
        phrase_view["Example jobs"] = phrase_view["example_jobs"].map(lambda value: _short(value, 130))
        phrase_view = phrase_view.rename(columns={
            "phrase": "Phrase", "phrase_length": "Words", "total_occurrences": "Occurrences",
            "job_count": "Jobs", "job_coverage_pct": "Coverage", "avg_tfidf": "Mean TF-IDF",
            "npmi": "NPMI", "top_niche": "Top niche",
        })
        styled_table(
            phrase_view[[
                "Phrase", "Words", "Occurrences", "Jobs", "Coverage", "NPMI",
                "Mean TF-IDF", "Trend", "Top niche", "Example jobs",
            ]],
            column_config={
                "Phrase": st.column_config.TextColumn("Phrase", width="medium"),
                "Words": st.column_config.NumberColumn("Words", width="small"),
                "Occurrences": st.column_config.NumberColumn("Occurrences", width="small"),
                "Jobs": st.column_config.NumberColumn("Jobs", width="small"),
                "Coverage": st.column_config.NumberColumn("Coverage", format="%.1f%%", width="small"),
                "NPMI": st.column_config.NumberColumn("NPMI", format="%.2f", width="small"),
                "Mean TF-IDF": st.column_config.NumberColumn("Mean TF-IDF", format="%.3f", width="small"),
                "Trend": st.column_config.TextColumn("Trend", width="small"),
                "Top niche": st.column_config.TextColumn("Top niche", width="medium"),
                "Example jobs": st.column_config.TextColumn("Example jobs", width="large"),
            },
            height=430,
        )
        st.caption(
            "NPMI is computed from document frequencies and clipped to [-1, 1]. "
            "A high NPMI on a tiny phrase can still be incidental; use the minimum job count and read the examples."
        )

    st.divider()
    section(
        "Distinctive keywords by niche",
        "Compare two groups and rank the terms that are over-represented in one relative to the other. The sign is directional, not causal.",
    )
    preset = st.selectbox("Compare", PRESETS, key="tm_compare")
    niches = sorted(work["niche"].dropna().unique())
    mask_a = mask_b = None
    label_a = label_b = "Group A"
    if preset == "Custom comparison":
        custom = st.columns(2)
        with custom[0]:
            values_a = _safe_multi("tm_group_a", niches, "Group A niches", niches[:2])
        with custom[1]:
            values_b = _safe_multi("tm_group_b", niches, "Group B niches", niches[2:4] if len(niches) > 2 else niches[:1])
        mask_a = work["niche"].isin(values_a)
        mask_b = work["niche"].isin(values_b)
        label_a = ", ".join(values_a) if values_a else "Group A"
        label_b = ", ".join(values_b) if values_b else "Group B"
    elif preset == "Finance vs All other jobs":
        mask_a = work["industry"].eq("Finance")
        mask_b = ~mask_a
        label_a, label_b = "Finance", "All other jobs"
    elif preset == "Crypto vs Finance":
        mask_a = work["industry"].eq("Crypto")
        mask_b = work["industry"].eq("Finance")
        label_a, label_b = "Crypto", "Finance"
    elif preset == "DeFi vs Other crypto jobs":
        mask_a = work["sub_niche"].eq("DeFi")
        mask_b = work["industry"].eq("Crypto") & ~work["sub_niche"].eq("DeFi")
        label_a, label_b = "DeFi", "Other crypto jobs"
    else:
        mask_a = work["sub_niche"].eq("Payments")
        mask_b = work["sub_niche"].eq("General Analytics")
        label_a, label_b = "Payments", "General Data Analytics"

    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
    if n_a == 0 or n_b == 0:
        st.warning(
            f"This comparison is empty in the current corpus ({label_a}: {n_a} jobs; {label_b}: {n_b} jobs). "
            "Try All filtered jobs, another preset, or widen the sidebar filters."
        )
    else:
        direction_label = st.radio(
            "Show terms distinctive for", [label_a, label_b], horizontal=True, key="tm_compare_direction"
        )
        direction = "A" if direction_label == label_a else "B"
        compared = tm.compare_terms(
            space, table, mask_a, mask_b, min_jobs=min_jobs, direction=direction, top_n=30,
        )
        if compared.empty:
            st.caption("No terms pass the minimum-job threshold for this comparison.")
        else:
            compared = compared[compared["phrase"].isin(set(filtered["phrase"]))].head(25)
        if compared.empty:
            st.caption("No terms pass the language filters in this comparison.")
        else:
            compared_view = compared.copy()
            compared_view["coverage_a"] = compared_view["coverage_a"].map(lambda value: f"{value:.1f}%")
            compared_view["coverage_b"] = compared_view["coverage_b"].map(lambda value: f"{value:.1f}%")
            compared_view["coverage_diff_pp"] = compared_view["coverage_diff_pp"].map(lambda value: f"{value:+.1f}")
            compared_view["lift"] = compared_view["lift"].map(lambda value: f"{value:.2f}×")
            compared_view["log_odds_z"] = compared_view["log_odds_z"].round(2)
            styled_table(
                compared_view.rename(columns={
                    "phrase": "Term", "coverage_a": f"Coverage in {label_a}",
                    "coverage_b": f"Coverage in {label_b}", "jobs_a": f"Jobs in {label_a}",
                    "jobs_b": f"Jobs in {label_b}", "coverage_diff_pp": "Difference (pp)",
                    "lift": "Lift", "log_odds_z": "Log-odds z", "top_niche": "Top niche",
                })[[
                    "Term", f"Jobs in {label_a}", f"Jobs in {label_b}",
                    f"Coverage in {label_a}", f"Coverage in {label_b}",
                    "Difference (pp)", "Lift", "Log-odds z", "Top niche",
                ]],
                column_config={
                    "Term": st.column_config.TextColumn("Term", width="medium"),
                    "Log-odds z": st.column_config.NumberColumn("Log-odds z", format="%.2f", width="small"),
                    "Lift": st.column_config.TextColumn("Lift", width="small"),
                    "Top niche": st.column_config.TextColumn("Top niche", width="medium"),
                },
                height=360,
            )
            chart_data = compared.copy()
            chart_data["size"] = chart_data["jobs_a"].clip(lower=1)
            fig = px.scatter(
                chart_data, x="coverage_a", y="coverage_b", size="size",
                color="log_odds_z", color_continuous_scale="RdBu_r", size_max=38,
                hover_name="phrase",
                labels={
                    "coverage_a": f"Coverage in {label_a} (%)",
                    "coverage_b": f"Coverage in {label_b} (%)",
                    "log_odds_z": "Log-odds z",
                },
            )
            limit = max(float(chart_data["coverage_a"].max()), float(chart_data["coverage_b"].max()), 10.0) * 1.08
            fig.add_shape(type="line", x0=0, y0=0, x1=limit, y1=limit, line=dict(color="#cbd5e1", dash="dot"))
            fig.update_traces(marker=dict(line=dict(color="white", width=1)))
            show(fig, height=380)
            st.caption(
                f"Group sizes: {label_a} = {n_a} jobs, {label_b} = {n_b} jobs. "
                "Log-odds uses a 0.5 smoothing prior. With this sample, small groups produce unstable extremes."
            )

    st.divider()
    section(
        "Keyword co-occurrence & network",
        "Move beyond pairs of adjacent words: discover which skills, tools, problems and phrases tend to appear together in the same project.",
    )
    controls = st.columns(4)
    with controls[0]:
        scope_label = st.selectbox(
            "Relationship scope", ["Same job", "Same sentence", "Skill → business problem"],
            key="tm_scope",
        )
    with controls[1]:
        metric_label = st.selectbox("Edge weight", list(METRIC_COLUMNS), key="tm_metric")
    with controls[2]:
        graph_terms_n = st.slider("Terms in graph", 6, 24, 14, key="tm_graph_terms")
    with controls[3]:
        min_both = st.slider("Minimum co-occurring jobs", 1, 10, 2, key="tm_min_both")
    metric = METRIC_COLUMNS[metric_label]

    if scope_label == "Skill → business problem":
        edges, nodes = tm.skill_problem_edges(work, min_jobs=min_both, top_skills=graph_terms_n)
        graph_terms = nodes["node"].tolist() if not nodes.empty else []
        focus_label = "Focus skill or problem"
    else:
        graph_terms = filtered.sort_values("job_count", ascending=False)["phrase"].head(graph_terms_n).tolist()
        edges, nodes = tm.term_edges(space, graph_terms, scope=scope_label, min_jobs=min_both)
        focus_label = "Focus keyword"

    if edges.empty:
        st.caption(
            "No relationship passes the current minimum co-occurring jobs. "
            "Lower the threshold, include more terms, or switch relationship scope."
        )
    else:
        focus_options = [ALL_NODES] + nodes.sort_values("jobs", ascending=False)["node"].tolist()
        focus = _safe_select("tm_graph_focus", focus_options, focus_label)
        view_edges = edges if focus == ALL_NODES else edges[
            edges["phrase_a"].eq(focus) | edges["phrase_b"].eq(focus)
        ]
        if view_edges.empty:
            st.caption(f"{focus} has no relationship above the current threshold.")
            view_edges = edges.iloc[0:0]
        view_nodes = nodes[
            nodes["node"].isin(set(view_edges["phrase_a"]).union(set(view_edges["phrase_b"])))
        ].copy()

        if not view_edges.empty:
            if scope_label == "Skill → business problem":
                color_options = ["Node type", "Co-occurrence community"]
            else:
                color_options = ["Dominant niche", "Co-occurrence community"]
            color_mode = st.radio("Colour nodes by", color_options, horizontal=True, key="tm_color_mode")
            figure = _network_figure(view_edges, view_nodes, metric, focus, scope_label, color_mode)
            if figure is not None:
                show(figure, height=530, legend=True)
            st.caption(
                "Node size = unique jobs containing the term. Edge width = the selected relationship metric. "
                "The graph is filtered to your focus node when one is selected. "
                "Louvain communities are discovered from the graph, not from the pre-existing taxonomy."
            )

            table_edges = view_edges.sort_values(metric, ascending=False).head(40).copy()
            if focus != ALL_NODES:
                focus_is_a = table_edges["phrase_a"].eq(focus)
                table_edges["focus_to_other"] = np.where(
                    focus_is_a, table_edges["confidence_a_to_b"], table_edges["confidence_b_to_a"]
                )
                table_edges["other_to_focus"] = np.where(
                    focus_is_a, table_edges["confidence_b_to_a"], table_edges["confidence_a_to_b"]
                )
            table_edges["support"] = table_edges["support"].map(lambda value: f"{value:.1f}%")
            table_edges["confidence_a_to_b"] = table_edges["confidence_a_to_b"].map(lambda value: f"{value:.1f}%")
            table_edges["confidence_b_to_a"] = table_edges["confidence_b_to_a"].map(lambda value: f"{value:.1f}%")
            table_edges["lift"] = table_edges["lift"].map(lambda value: f"{value:.2f}×")
            table_edges["jaccard"] = table_edges["jaccard"].map(lambda value: f"{value:.1f}%")
            if focus != ALL_NODES:
                table_edges["focus_to_other"] = table_edges["focus_to_other"].map(lambda value: f"{value:.1f}%")
                table_edges["other_to_focus"] = table_edges["other_to_focus"].map(lambda value: f"{value:.1f}%")
            columns = ["phrase_a", "phrase_b", "jobs_with_both", "support", "confidence_a_to_b", "confidence_b_to_a", "lift", "jaccard"]
            if focus != ALL_NODES:
                columns += ["focus_to_other", "other_to_focus"]
            styled_table(
                table_edges[columns].rename(columns={
                    "phrase_a": "Node A", "phrase_b": "Node B", "jobs_with_both": "Both",
                    "confidence_a_to_b": "A → B", "confidence_b_to_a": "B → A",
                    "focus_to_other": "Focus → other", "other_to_focus": "Other → focus",
                }),
                column_config={
                    "Node A": st.column_config.TextColumn("Node A", width="medium"),
                    "Node B": st.column_config.TextColumn("Node B", width="medium"),
                    "Both": st.column_config.NumberColumn("Co-occurring jobs", width="small"),
                    "Lift": st.column_config.TextColumn("Lift", width="small"),
                },
                height=330,
            )
            st.caption(
                "Support = share of all analyzed jobs containing both. A→B is the share of A jobs that also contain B. "
                "Lift above 1 means the pair is more common than independent frequencies predict. "
                "Jaccard is the overlap of the two job sets."
            )

            edge_labels = [
                f"{row.phrase_a} ↔ {row.phrase_b}" for row in view_edges.itertuples()
            ]
            edge_label = _safe_select("tm_edge", edge_labels, "Inspect the jobs behind a relationship")
            chosen = view_edges.iloc[edge_labels.index(edge_label)]
            evidence = work[work["job_id"].isin(chosen["job_ids"])].copy()
            context_text = "Both nodes appear in these jobs."
            if scope_label == "Same sentence":
                context_text = "Both terms appear in the same sentence in these jobs."
            elif scope_label == "Skill → business problem":
                context_text = "These jobs contain the skill and were classified with the business problem."
            st.caption(f"{len(evidence)} common jobs. {context_text}")
            job_table(evidence.sort_values("job_score", ascending=False, na_position="last"), n=20)

            with st.expander("Node detail", expanded=False):
                node_view = view_nodes.sort_values("jobs", ascending=False).rename(columns={
                    "node": "Node", "jobs": "Jobs", "top_niche": "Niche / type",
                    "node_type": "Type",
                })
                styled_table(
                    node_view[["Node", "Jobs", "Niche / type", "Type"]],
                    column_config={"Node": st.column_config.TextColumn("Node", width="medium")},
                    height=300,
                )
            note(
                "Lift is unstable when either node appears in very few jobs. Treat the graph as a hypothesis map: "
                "open the evidence jobs before using a relationship in an offer.",
                kind="data",
            )

    st.divider()
    section(
        "Keyword trend & context explorer",
        "Choose one phrase to see whether its coverage is rising and read the actual sentences where clients used it.",
    )
    focus_options = filtered.sort_values("job_count", ascending=False)["phrase"].head(250).tolist()
    focus = _safe_select("tm_focus", focus_options, "Focus phrase / keyword")
    focus_row = filtered[filtered["phrase"].eq(focus)].iloc[0]
    kpi_row([
        ("Focus phrase", focus, f"{int(focus_row['job_count'])} jobs · {pct(focus_row['job_coverage_pct'], 1)} coverage"),
        ("Current coverage", pct(focus_row["current_coverage"], 1), f"Latest {window_days} days: {int(focus_row['current_job_count'])} of {current_total} dated jobs."),
        ("Previous coverage", pct(focus_row["previous_coverage"], 1), f"Previous {window_days} days: {int(focus_row['previous_job_count'])} of {previous_total} dated jobs."),
        ("Coverage change", f"{focus_row['coverage_change_pp']:+.1f} pp" if pd.notna(focus_row["coverage_change_pp"]) else "n/a", "Percentage-point change, not relative growth."),
        ("First seen", _date_text(focus_row["first_seen_at"]), "First dated observation in this filtered corpus, not necessarily first appearance in the market."),
    ])

    trend = tm.weekly_trend(space, focus)
    if trend.empty:
        st.caption("No dated jobs contain this phrase, so a time trend cannot be drawn.")
    else:
        figure = make_subplots(specs=[[{"secondary_y": True}]])
        figure.add_trace(
            go.Bar(
                x=trend["week"], y=trend["jobs"], name="Jobs collected",
                marker_color="#cbd5e1",
                hovertemplate="Week of %{x|%Y-%m-%d}<br>%{y} jobs collected<extra></extra>",
            ),
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=trend["week"], y=trend["coverage"], name="Phrase coverage",
                mode="lines+markers", line=dict(color="#4f6df5", width=3), marker=dict(size=7),
                hovertemplate="Week of %{x|%Y-%m-%d}<br>%{y:.1f}% coverage<extra></extra>",
            ),
            secondary_y=True,
        )
        figure.update_yaxes(title_text="Jobs collected", secondary_y=False, rangemode="tozero")
        figure.update_yaxes(title_text="Coverage (%)", secondary_y=True, range=[0, 105])
        show(figure, height=340, legend=True)
        st.caption(
            "Bars show how many jobs were collected in each week; the line is the share containing the focus phrase. "
            "Coverage controls for collection volume. A week with few jobs can swing sharply."
        )

    emerging_table = filtered[
        filtered["coverage_change_pp"].gt(0)
        & filtered["current_job_count"].ge(1)
        & filtered["coverage_change_pp"].notna()
    ].sort_values("coverage_change_pp", ascending=False).head(25).copy()
    if emerging_table.empty:
        st.caption("No term shows rising coverage in the current trend window.")
    else:
        emerging_table["trend_change_pct"] = emerging_table.apply(_trend_text, axis=1)
        emerging_table["first_seen_at"] = emerging_table["first_seen_at"].map(_date_text)
        emerging_table["last_seen_at"] = emerging_table["last_seen_at"].map(_date_text)
        styled_table(
            emerging_table.rename(columns={
                "phrase": "Term", "current_job_count": "Current jobs",
                "previous_job_count": "Previous jobs", "current_coverage": "Current coverage",
                "previous_coverage": "Previous coverage", "coverage_change_pp": "Change (pp)",
                "trend_change_pct": "Relative change", "first_seen_at": "First seen",
                "last_seen_at": "Last seen", "top_niche": "Top niche",
            })[[
                "Term", "Current jobs", "Previous jobs", "Current coverage", "Previous coverage",
                "Change (pp)", "Relative change", "First seen", "Last seen", "Top niche",
            ]],
            column_config={
                "Term": st.column_config.TextColumn("Term", width="medium"),
                "Current coverage": st.column_config.NumberColumn("Current coverage", format="%.1f%%", width="small"),
                "Previous coverage": st.column_config.NumberColumn("Previous coverage", format="%.1f%%", width="small"),
                "Change (pp)": st.column_config.NumberColumn("Change (pp)", format="%+.1f", width="small"),
                "Top niche": st.column_config.TextColumn("Top niche", width="medium"),
            },
            height=340,
        )
        st.caption(
            f"Largest coverage gains in the latest {window_days} days. Terms that were absent in the previous period "
            "are marked new; that can reflect a real shift or simply a short collection history."
        )

    matches = tm.context_matches(space, focus)
    if matches.empty:
        st.caption("No readable context found for this phrase.")
    else:
        match_controls = st.columns([1.0, 1.0])
        with match_controls[0]:
            sort_matches = st.selectbox(
                "Sort matches by", ["Job score", "Newest", "Fewest proposals"], key="tm_context_sort"
            )
        with match_controls[1]:
            max_matches = max(1, min(30, len(matches)))
            show_matches = st.slider(
                "Matches to show", 1, max_matches, min(10, max_matches), key="tm_context_count"
            )
        if sort_matches == "Newest":
            matches = matches.sort_values("posted_date", ascending=False, na_position="last")
        elif sort_matches == "Fewest proposals":
            matches = matches.sort_values("proposal_mid", ascending=True, na_position="last")
        else:
            matches = matches.sort_values("job_score", ascending=False, na_position="last")

        left, right = st.columns([1.35, 0.9])
        with left:
            st.caption(f"{len(matches)} matching job(s). Use the sidebar to narrow by industry, date, rate or proposals.")
            with st.container(height=560, border=False):
                for _, match in matches.head(show_matches).iterrows():
                    st.markdown(f"**{html.escape(str(match['title']))}**")
                    score_text = f"{match['job_score']:.0f}/100" if pd.notna(match["job_score"]) else "n/a"
                    st.caption(
                        f"{_date_text(match['posted_date'])} · {match['niche']} · "
                        f"{match['rate_label']} · {match['proposal_bucket']} proposals · "
                        f"score {score_text} · match in {match['source'].lower()}"
                    )
                    st.markdown(
                        f"<div style='line-height:1.55;color:#1e293b;background:#f8fafc;"
                        f"border-left:3px solid #4f6df5;padding:8px 12px;border-radius:4px;'>"
                        f"{_highlight(match['context'], focus)}</div>",
                        unsafe_allow_html=True,
                    )
                    url = str(match["url"] or "")
                    if url.startswith("http"):
                        st.markdown(f"[Open the job on Upwork ↗]({url})")
                    st.divider()
        with right:
            section("Frequent co-occurring terms", "Terms that share jobs with the focus phrase. Overlapping n-grams are possible.")
            co_terms = tm.cooccurring_terms(space, focus, top_n=12)
            if co_terms.empty:
                st.caption("No co-occurring terms in this selection.")
            else:
                co_view = co_terms.copy()
                co_view["confidence"] = co_view["confidence"].map(lambda value: f"{value:.0f}%")
                co_view["lift"] = co_view["lift"].map(lambda value: f"{value:.2f}×")
                styled_table(
                    co_view.rename(columns={
                        "phrase": "Term", "jobs_with_both": "Shared jobs", "jobs": "Jobs",
                        "confidence": "Focus→term", "lift": "Lift",
                    }),
                    column_config={"Term": st.column_config.TextColumn("Term", width="medium")},
                    height=300,
                )
            section("Skills in these jobs", "Skills attached to the matching ads.")
            skills = tm.related_skills(space, focus, top_n=12)
            if skills.empty:
                st.caption("No skill metadata in the matching jobs.")
            else:
                styled_table(
                    skills.rename(columns={"skill": "Skill", "jobs": "Jobs"}),
                    column_config={"Skill": st.column_config.TextColumn("Skill", width="medium")},
                    height=300,
                )

    st.divider()
    note(
        "This page describes language patterns in one search snapshot. It does not prove demand, willingness to pay "
        "or causality. Validate high-value phrases against the linked briefs, then check the Niche Intelligence and "
        "Clients & Competition pages before committing to a service.",
        kind="data",
    )
