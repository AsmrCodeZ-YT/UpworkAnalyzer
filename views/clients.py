"""Page 4 — Clients & Competition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as a
from views.common import (
    MUTED, NICHE_COLORS, PROPOSAL_COLORS, job_table, kpi_row, money, no_data,
    note, pct, section, show, styled_table,
)


def _spend_band(value: float) -> str:
    if pd.isna(value):
        return "Unknown"
    if value <= 0:
        return "$0"
    if value < 1_000:
        return "<$1K"
    if value < 10_000:
        return "$1K–10K"
    if value < 50_000:
        return "$10K–50K"
    if value < 100_000:
        return "$50K–100K"
    if value < 500_000:
        return "$100K–500K"
    return "$500K+"


def render(df: pd.DataFrame, full: pd.DataFrame) -> None:
    st.title("Clients & Competition")
    st.caption("Not every job is worth pursuing. Compare proposal pressure, client quality and contract structure before spending connects.")

    relevant = df[df["relevant"]].copy()
    if relevant.empty:
        no_data()
        return

    hourly = relevant.loc[relevant["job_type"].eq("Hourly"), "hourly_mid"].dropna()
    fixed = relevant.loc[relevant["job_type"].eq("Fixed price"), "fixed_budget"].dropna()
    rated = relevant[(relevant["client_rating"] > 0) & (relevant["client_reviews"] > 0)]
    weighted_rating = (
        (rated["client_rating"] * rated["client_reviews"]).sum() / rated["client_reviews"].sum()
        if not rated.empty else np.nan
    )

    kpi_row([
        ("Payment verified", pct(relevant["payment_verified"].mean() * 100), "Share of relevant jobs with a payment-verified client."),
        ("Median client spend", money(relevant["client_spent"].median()), "Median visible lifetime spend of clients (includes fallback from spend bands)."),
        ("Weighted rating", f"{weighted_rating:.2f} / 5" if pd.notna(weighted_rating) else "n/a", f"Rating weighted by review count across {len(rated)} rated clients."),
        ("Low-competition jobs", pct(relevant["low_competition"].mean() * 100), "Share with 0–10 proposals at collection time."),
        ("Long-term contracts", pct(relevant["long_term"].mean() * 100), "Share with duration > 6 months."),
    ])

    st.divider()
    c1, c2 = st.columns([1.1, 1])

    with c1:
        section("Proposal pressure", "Upwork reports proposal counts in bands. The snapshot only reflects the moment of collection.")
        counts = (
            relevant["proposal_bucket"].value_counts()
            .reindex(a.PROPOSAL_ORDER).dropna().astype(int).reset_index()
        )
        counts.columns = ["bucket", "jobs"]
        fig = px.bar(
            counts, x="bucket", y="jobs", color="bucket",
            color_discrete_map=PROPOSAL_COLORS, text="jobs",
            labels={"bucket": "Proposals", "jobs": "Jobs"},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title=None, rangemode="tozero")
        show(fig, height=330)

    with c2:
        section("Contract mix", "Hourly vs. fixed-price demand and the duration profile of relevant jobs.")
        mix = relevant["job_type"].value_counts().rename_axis("type").reset_index(name="jobs")
        fig = px.pie(
            mix, names="type", values="jobs", hole=0.55,
            color="type", color_discrete_sequence=["#4f6df5", "#14b8a6"],
        )
        fig.update_traces(texttemplate="%{label}<br>%{value} jobs", textinfo="text")
        fig.update_layout(showlegend=False)
        show(fig, height=330)

    st.divider()
    section("Budget vs. competition", "Do higher-value jobs really attract more proposals? Hourly and fixed-price contracts are separated.")
    tab_h, tab_f = st.tabs(["Hourly rate", "Fixed budget"])
    with tab_h:
        data = relevant.dropna(subset=["hourly_mid"]).copy()
        data = data[data["proposal_bucket"].ne("Unknown")]
        if data.empty:
            st.caption("No hourly data.")
        else:
            order = [b for b in a.PROPOSAL_ORDER if b in data["proposal_bucket"].unique()]
            fig = px.box(
                data, x="proposal_bucket", y="hourly_mid", color="proposal_bucket",
                color_discrete_map=PROPOSAL_COLORS, category_orders={"proposal_bucket": order},
                points="all", labels={"proposal_bucket": "Proposals", "hourly_mid": "Hourly rate ($/h)"},
            )
            fig.update_traces(marker=dict(size=5, opacity=0.55))
            fig.update_layout(showlegend=False)
            show(fig, height=380)
    with tab_f:
        data = relevant.dropna(subset=["fixed_budget"]).copy()
        data = data[data["proposal_bucket"].ne("Unknown")]
        if data.empty:
            st.caption("No fixed-price data.")
        else:
            order = [b for b in a.PROPOSAL_ORDER if b in data["proposal_bucket"].unique()]
            fig = px.box(
                data, x="proposal_bucket", y="fixed_budget", color="proposal_bucket",
                color_discrete_map=PROPOSAL_COLORS, category_orders={"proposal_bucket": order},
                points="all", labels={"proposal_bucket": "Proposals", "fixed_budget": "Fixed budget ($)"},
            )
            fig.update_traces(marker=dict(size=5, opacity=0.55))
            fig.update_layout(showlegend=False)
            show(fig, height=380)

    st.divider()
    c1, c2 = st.columns([1, 1])

    with c1:
        section("Client spend bands", "Visible lifetime spend. $0 and unknown are not the same: unknown means the field was absent.")
        bands = relevant["client_spent"].map(_spend_band)
        order = ["$0", "<$1K", "$1K–10K", "$10K–50K", "$50K–100K", "$100K–500K", "$500K+", "Unknown"]
        band_df = bands.value_counts().reindex(order).dropna().astype(int).reset_index()
        band_df.columns = ["band", "jobs"]
        fig = px.bar(
            band_df, x="band", y="jobs",
            color="band", color_discrete_sequence=px.colors.sequential.Blues_r,
            text="jobs", labels={"band": "Client lifetime spend", "jobs": "Jobs"},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title=None, rangemode="tozero")
        show(fig, height=330)

    with c2:
        section("Rating vs. review count", "A 5.0 rating from one review is not the same signal as 4.9 from a hundred reviews. Bubble size = reviews.")
        rated = relevant[relevant["client_reviews"] > 0].copy()
        if rated.empty:
            st.caption("No rated clients in this selection.")
        else:
            fig = px.scatter(
                rated, x="client_reviews", y="client_rating", size="client_reviews",
                color="payment_verified", color_discrete_map={True: "#0ea5e9", False: "#ef4444"},
                hover_name="title", size_max=28,
                labels={"client_reviews": "Reviews", "client_rating": "Rating", "payment_verified": "Payment verified"},
            )
            fig.update_xaxes(type="log")
            fig.update_yaxes(range=[0, 5.3])
            show(fig, height=330, legend=True)

    st.divider()
    section("Client quality by niche", "Composite of payment verification, visible spend and review-weighted rating. Small samples are noisy.")
    qrows = []
    for niche, g in relevant.groupby("niche"):
        rated_g = g[(g["client_rating"] > 0) & (g["client_reviews"] > 0)]
        rating = (
            (rated_g["client_rating"] * rated_g["client_reviews"]).sum() / rated_g["client_reviews"].sum()
            if not rated_g.empty else np.nan
        )
        qrows.append({
            "Niche": niche,
            "Jobs": len(g),
            "Verified": pct(g["payment_verified"].mean() * 100),
            "Median spend": money(g["client_spent"].median()),
            "Rating": f"{rating:.2f}" if pd.notna(rating) else "n/a",
            "Reviews": int(g["client_reviews"].fillna(0).sum()),
            "Low comp.": pct(g["low_competition"].mean() * 100),
            "Long term": pct(g["long_term"].mean() * 100),
            "Client score": round(g["client_quality"].median(), 1),
        })
    styled_table(
        pd.DataFrame(qrows).sort_values("Client score", ascending=False),
        column_config={
            "Niche": st.column_config.TextColumn("Niche", width="medium"),
            "Client score": st.column_config.ProgressColumn("Client score", min_value=0, max_value=100, format="%.1f", width="small"),
        },
    )

    st.divider()
    c1, c2 = st.columns([1, 1])

    with c1:
        section("Demand by client country", "The dataset is biased toward the US due to the search filter (location=United States).")
        countries = relevant["client_country"].fillna("Unknown").value_counts().head(12).reset_index()
        countries.columns = ["country", "jobs"]
        fig = px.bar(
            countries.sort_values("jobs"), x="jobs", y="country", orientation="h",
            color="jobs", color_continuous_scale="Blues", text="jobs",
            labels={"country": "", "jobs": "Jobs"},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False)
        show(fig, height=330)

    with c2:
        section("Duration and workload", "Long-term indicators matter for repeat revenue, but they are not a guarantee of hiring.")
        duration = relevant["duration"].fillna("Unknown").value_counts().reset_index()
        duration.columns = ["duration", "jobs"]
        fig = px.bar(
            duration, x="duration", y="jobs", text="jobs",
            color="duration", color_discrete_sequence=px.colors.sequential.Teal,
            labels={"duration": "", "jobs": "Jobs"},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title=None, rangemode="tozero")
        show(fig, height=330)

    st.divider()
    with st.expander("Revenue potential scenario (not a forecast)", expanded=False):
        c1, c2, c3 = st.columns(3)
        default_rate = float(hourly.median()) if len(hourly) else 50.0
        with c1:
            rate = st.number_input("Hourly rate ($/h)", min_value=0.0, value=round(default_rate, 0), step=5.0)
        with c2:
            hours = st.slider("Billable hours per week", 1, 40, 20)
        with c3:
            weeks = st.slider("Weeks of engagement", 1, 52, 12)
        scenario = rate * hours * weeks
        st.metric("Scenario revenue", money(scenario))
        st.caption(
            "This is arithmetic, not a forecast. It ignores proposal competition, hiring probability, unpaid discovery time, "
            "scope changes and payment risk. Use it only to compare contract structures."
        )

    st.divider()
    section("Fresh, low-competition jobs", "≤ 7 days old with 0–10 proposals. The best pool for a quick, targeted proposal.")
    fresh = relevant[relevant["fresh_7d"] & relevant["low_competition"]].sort_values("job_score", ascending=False)
    if fresh.empty:
        fresh = relevant[relevant["fresh_30d"] & relevant["low_competition"]].sort_values("job_score", ascending=False)
        st.caption("No jobs met the 7-day rule; showing ≤ 30 days with 0–10 proposals instead.")
    job_table(fresh, n=20)
    note(
        "Proposal counts are a snapshot. Two proposals one hour after posting is not the same as two proposals three days later, "
        "and competitor quality is invisible here. Use this as a prioritisation aid, not a probability of winning.",
        kind="data",
    )
