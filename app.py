"""Upwork FinTech / Finance / Crypto Market Intelligence Dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import analysis as a
from views import clients, discovery, niches, overview, quality, textmining

DATA_PATH = Path(__file__).parent / "mkb.csv"

st.set_page_config(
    page_title="Upwork Market Intelligence",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Cleaning, classifying and scoring jobs…")
def load_data(path: str, mtime: float) -> pd.DataFrame:
    return a.prepare_data(path)


FILTER_KEYS = [
    "f_date", "f_unknown_dates", "f_industry", "f_subniche", "f_problem", "f_skill",
    "f_jobtype", "f_hourly", "f_fixed", "f_keep_unknown", "f_proposals",
    "f_country", "f_verified", "f_duration", "f_query",
]


def _date_bounds(df: pd.DataFrame):
    dates = df["posted_date"].dropna()
    if dates.empty:
        return None, None
    return dates.min().date(), dates.max().date()


def render_filters(df: pd.DataFrame) -> dict:
    with st.sidebar:
        st.markdown("### Market Intelligence")
        st.caption("Upwork · FinTech / Finance / Crypto")
        st.divider()

        reset = st.button("Reset filters", width="stretch", type="tertiary")
        if reset:
            for key in FILTER_KEYS:
                st.session_state.pop(key, None)
            st.rerun()

        st.markdown("**Timing**")
        lo, hi = _date_bounds(df)
        include_unknown_dates = True
        if lo is not None:
            date_value = st.date_input(
                "Posted date range",
                value=(lo, hi),
                min_value=lo,
                max_value=hi,
                key="f_date",
                help="83 of 93 dates are estimated from relative strings. Interpret accordingly.",
            )
            if isinstance(date_value, (tuple, list)):
                if len(date_value) == 2:
                    start, end = date_value
                elif len(date_value) == 1:
                    start = end = date_value[0]
                else:
                    start, end = lo, hi
            else:
                start = end = date_value
            include_unknown_dates = st.checkbox(
                "Keep jobs with unknown date", value=True, key="f_unknown_dates"
            )
        else:
            start = end = None

        st.markdown("**Market**")
        industries = sorted(df["industry"].unique())
        selected_industries = st.multiselect("Industry", industries, key="f_industry")
        subs = sorted(df["sub_niche"].unique())
        selected_subs = st.multiselect("Sub-niche", subs, key="f_subniche")
        problems = [p for p in a.PROBLEM_ORDER if p in set(df["problem_primary"])]
        selected_problems = st.multiselect("Problem type", problems, key="f_problem")
        all_skills = sorted({s for lst in df["skill_list"] for s in lst})
        selected_skills = st.multiselect("Skill", all_skills, key="f_skill", help="Matches if any selected skill is present.")

        st.markdown("**Contract & budget**")
        job_types = [t for t in ["Hourly", "Fixed price"] if t in set(df["job_type"])]
        selected_types = st.multiselect("Job type", job_types + [t for t in df["job_type"].unique() if t not in job_types], key="f_jobtype")

        hourly_series = df.loc[df["job_type"].eq("Hourly"), "hourly_mid"].dropna()
        fixed_series = df.loc[df["job_type"].eq("Fixed price"), "fixed_budget"].dropna()
        hourly_range = (0.0, float(np.ceil(hourly_series.max())) if not hourly_series.empty else 200.0)
        fixed_range = (0.0, float(np.ceil(fixed_series.max())) if not fixed_series.empty else 5000.0)
        if not hourly_series.empty:
            hourly_range = st.slider(
                "Hourly rate ($/h)", 0.0, float(np.ceil(hourly_series.max())),
                (0.0, float(np.ceil(hourly_series.max()))), step=5.0, key="f_hourly",
            )
        if not fixed_series.empty:
            step = 100.0 if fixed_series.max() > 1000 else 25.0
            fixed_range = st.slider(
                "Fixed budget ($)", 0.0, float(np.ceil(fixed_series.max() / step) * step),
                (0.0, float(np.ceil(fixed_series.max() / step) * step)), step=step, key="f_fixed",
            )
        keep_unknown = st.checkbox("Keep jobs with unknown rate / budget", value=True, key="f_keep_unknown")

        selected_proposals = st.multiselect(
            "Proposal bucket", [b for b in a.PROPOSAL_ORDER if b in set(df["proposal_bucket"])], key="f_proposals"
        )

        st.markdown("**Client & contract terms**")
        countries = sorted(df["client_country"].dropna().unique())
        selected_countries = st.multiselect("Client country", countries, key="f_country")
        verified = st.selectbox(
            "Payment verified", ["All", "Verified only", "Not verified"], key="f_verified"
        )
        durations = sorted(df["duration"].dropna().unique())
        selected_durations = st.multiselect("Duration", durations, key="f_duration")

        st.markdown("**Collection**")
        queries = sorted(df["search_query"].dropna().unique()) if "search_query" in df else []
        selected_queries = st.multiselect("Source search query", queries, key="f_query")

        st.divider()
        st.caption(
            "Scores are decision aids built on a single, search-biased snapshot. "
            "See **Data Quality & Collection Coverage** before acting on any ranking."
        )

    return {
        "start": start, "end": end, "include_unknown_dates": include_unknown_dates,
        "industries": selected_industries, "subs": selected_subs,
        "problems": selected_problems, "skills": selected_skills,
        "job_types": selected_types, "hourly": hourly_range, "fixed": fixed_range,
        "keep_unknown": keep_unknown, "proposals": selected_proposals,
        "countries": selected_countries, "verified": verified,
        "durations": selected_durations, "queries": selected_queries,
    }


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    out = df.copy()

    if f["start"] and f["end"]:
        start = pd.Timestamp(f["start"])
        end = pd.Timestamp(f["end"]) + pd.to_timedelta(1, unit="D")
        in_range = out["posted_date"].between(start, end, inclusive="left")
        if f["include_unknown_dates"]:
            in_range = in_range | out["posted_date"].isna()
        out = out[in_range]

    if f["industries"]:
        out = out[out["industry"].isin(f["industries"])]
    if f["subs"]:
        out = out[out["sub_niche"].isin(f["subs"])]
    if f["problems"]:
        out = out[out["problem_primary"].isin(f["problems"])]
    if f["skills"]:
        wanted = set(f["skills"])
        out = out[out["skill_list"].map(lambda skills: bool(wanted.intersection(skills)))]
    if f["job_types"]:
        out = out[out["job_type"].isin(f["job_types"])]

    hourly_lo, hourly_hi = f["hourly"]
    fixed_lo, fixed_hi = f["fixed"]
    is_hourly = out["job_type"].eq("Hourly")
    is_fixed = out["job_type"].eq("Fixed price")
    hourly_ok = out["hourly_mid"].between(hourly_lo, hourly_hi)
    fixed_ok = out["fixed_budget"].between(fixed_lo, fixed_hi)
    if f["keep_unknown"]:
        hourly_ok = hourly_ok | out["hourly_mid"].isna()
        fixed_ok = fixed_ok | out["fixed_budget"].isna()
    out = out[(~is_hourly | hourly_ok) & (~is_fixed | fixed_ok)]

    if f["proposals"]:
        out = out[out["proposal_bucket"].isin(f["proposals"])]
    if f["countries"]:
        out = out[out["client_country"].isin(f["countries"])]
    if f["verified"] == "Verified only":
        out = out[out["payment_verified"]]
    elif f["verified"] == "Not verified":
        out = out[~out["payment_verified"]]
    if f["durations"]:
        out = out[out["duration"].isin(f["durations"])]
    if f["queries"] and "search_query" in out:
        out = out[out["search_query"].isin(f["queries"])]

    return out.reset_index(drop=True)


def main() -> None:
    data = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    filters = render_filters(data)
    filtered = apply_filters(data, filters)

    with st.sidebar:
        st.caption(f"{len(filtered):,} of {len(data):,} unique jobs in scope")

    pages = [
        st.Page(lambda: overview.render(filtered, data), title="Market Overview", url_path="overview", default=True),
        st.Page(lambda: niches.render(filtered, data), title="Niche Intelligence", url_path="niches"),
        st.Page(lambda: discovery.render(filtered, data), title="Project & Service Discovery", url_path="discovery"),
        st.Page(lambda: textmining.render(filtered, data), title="Text Mining", url_path="text-mining"),
        st.Page(lambda: clients.render(filtered, data), title="Clients & Competition", url_path="clients"),
        st.Page(lambda: quality.render(filtered, data), title="Data Quality & Coverage", url_path="quality"),
    ]
    navigation = st.navigation(pages, position="sidebar")
    navigation.run()


if __name__ == "__main__":
    main()
