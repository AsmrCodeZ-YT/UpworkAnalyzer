"""Data preparation, classification, scoring and mining for the Upwork market dashboard."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

# Keywords are (phrase, weight). Single words use word boundaries, phrases are
# allowed to match anywhere.
NICHE_RULES: dict[str, dict[str, tuple[str, list[tuple[str, float]]]]] = {
    "FinTech": {
        "Payments": (
            "Payment Analytics, Settlement Reconciliation",
            [
                ("payment reconciliation", 5), ("payment analytics", 5),
                ("payment", 3), ("settlement", 3),
                ("stripe", 3), ("paypal", 3), ("adyen", 3), ("braintree", 3),
                ("dlocal", 3), ("merchant", 2), ("gateway", 2),
                ("chargeback", 3), ("payout", 2), ("remittance", 3),
                ("billing", 2), ("invoicing", 2), ("fintech", 3),
                ("subscription billing", 3), ("transaction fees", 2),
            ],
        )
    },
    "Finance": {
        "Financial Operations": (
            "Reconciliation, Financial Reporting, Data Quality",
            [
                ("reconciliation", 4), ("reconcile", 4), ("financial reporting", 5),
                ("financial data quality", 5), ("month-end close", 4),
                ("journal entries", 4), ("general ledger", 3), ("gaap", 3),
                ("ifrs", 3), ("financial operations", 5), ("accounting", 2),
                ("bookkeeping", 2), ("balance sheet", 3), ("cash flow", 3),
                ("profit and loss", 3), ("p&l", 2), ("variance analysis", 3),
                ("financial analysis", 2), ("financial analyst", 2),
                ("financial data", 2), ("audit", 1), ("ledger", 2),
            ],
        ),
        "Risk & Fraud": (
            "Fraud Analytics, Risk Reporting, Anomaly Detection",
            [
                ("fraud", 5), ("fraud analytics", 5), ("risk analytics", 5),
                ("risk analysis", 3), ("risk reporting", 4),
                ("anomaly detection", 4), ("transaction monitoring", 3),
                ("operational risk", 3), ("risk model", 3),
                ("risk data", 3), ("chargeback", 2), ("suspicious transactions", 3),
            ],
        ),
        "Investment Intelligence": (
            "Market Intelligence, Investment Data, M&A Due Diligence",
            [
                ("investment research", 5), ("investment data", 5), ("m&a", 4),
                ("due diligence", 4), ("market intelligence", 5),
                ("equity research", 5), ("private equity", 4),
                ("venture capital", 4), ("asset management", 4),
                ("investment", 3), ("valuation", 2), ("portfolio", 1),
                ("market research", 2), ("investors", 2), ("financial modeling", 2),
            ],
        ),
    },
    "Banking": {
        "Lending & Credit": (
            "Credit Risk, Loan Portfolio Reporting",
            [
                ("credit risk", 5), ("lending", 4), ("loan portfolio", 5),
                ("loan", 4), ("underwriting", 4), ("basel", 5),
                ("stress testing", 4), ("banking", 3),
                ("portfolio monitoring", 3), ("credit", 2),
            ],
        )
    },
    "Compliance": {
        "AML / KYC": (
            "AML/KYC Data Pipelines, Suspicious Pattern Analysis",
            [
                ("anti-money laundering", 6), ("aml", 5), ("kyc", 5),
                ("sanctions", 5), ("regulatory reporting", 5),
                ("suspicious activity", 5), ("customer due diligence", 4),
                ("transaction monitoring", 3), ("compliance", 2),
                ("regulatory", 2), ("gdpr", 2), ("audit", 1),
            ],
        )
    },
    "Crypto": {
        "On-chain Analytics": (
            "Wallet, Transaction and Asset Flow Analytics",
            [
                ("on-chain", 6), ("onchain", 6), ("blockchain", 5),
                ("wallet", 3), ("bitcoin", 3),
                ("ethereum", 3), ("etherscan", 4), ("dune", 3),
                ("solana", 3), ("evm", 3), ("token flows", 5),
                ("address clustering", 5), ("web3", 3), ("nft", 2),
                ("crypto", 2), ("smart contract", 3),
            ],
        ),
        "DeFi": (
            "Liquidity, TVL, Protocol and Yield Analytics",
            [
                ("defi", 6), ("protocol analytics", 6), ("tvl", 5),
                ("total value locked", 5), ("liquidity pool", 4),
                ("lending protocol", 4), ("yield farming", 5),
                ("uniswap", 4), ("aave", 4), ("staking", 3),
                ("dex", 3), ("liquidity", 2), ("yield", 2),
            ],
        ),
        "Exchanges": (
            "Order Book, Trading Volume and Liquidity Analytics",
            [
                ("crypto exchange", 6), ("order book", 5),
                ("market making", 4), ("trading volume", 4),
                ("trading fees", 4), ("binance", 4), ("coinbase", 3),
                ("cex", 4), ("exchange", 2), ("liquidity", 2),
            ],
        ),
    },
    "Cross-industry": {
        "Data Engineering": (
            "ETL/ELT, Data Quality, Automation, Integration",
            [
                ("etl", 5), ("elt", 5), ("data pipeline", 4),
                ("airflow", 4), ("dbt", 4), ("data warehouse", 4),
                ("snowflake", 1.5), ("bigquery", 1.5), ("databricks", 2),
                ("data ingestion", 4), ("data integration", 3),
                ("data quality", 3), ("data engineering", 4),
                ("data pipeline", 4), ("api integration", 2),
                ("web scraping", 2), ("data scraping", 2),
                ("data cleaning", 1), ("data modeling", 2.5),
                ("kafka", 4), ("spark", 3), ("automation", 1),
                ("python", 0.25), ("sql", 0.25),
            ],
        )
    },
}

PROBLEM_RULES: dict[str, list[tuple[str, float]]] = {
    "Reconciliation & Settlement": [
        ("reconciliation", 5), ("reconcile", 5), ("settlement", 4),
        ("transactions do not match", 5), ("mismatch", 3), ("discrepanc", 3),
        ("chargeback", 2), ("ledger", 2), ("journal entries", 3),
    ],
    "Financial Reporting & Automation": [
        ("financial reporting", 5), ("reporting automation", 5),
        ("automated reporting", 4), ("manual spreadsheet", 4),
        ("month-end", 3), ("management reporting", 3),
        ("financial statement", 4), ("p&l", 3), ("balance sheet", 3),
        ("variance analysis", 3), ("financial report", 3),
    ],
    "Data Quality & Validation": [
        ("data quality", 5), ("data validation", 4), ("quality control", 4),
        ("data cleaning", 3), ("inconsistent", 3), ("validate", 2),
        ("data integrity", 4), ("duplicate", 2), ("accuracy", 1.5),
        ("audit", 1), ("data governance", 4),
    ],
    "Data Pipeline & Integration": [
        ("data pipeline", 5), ("etl", 5), ("elt", 5), ("data integration", 4),
        ("data ingestion", 4), ("airflow", 4), ("dbt", 4),
        ("combine data", 3), ("multiple sources", 2),
        ("data warehouse", 4), ("api integration", 3), ("automation", 1),
    ],
    "Fraud & Risk Analytics": [
        ("fraud", 5), ("risk analytics", 4), ("risk analysis", 3),
        ("anomaly detection", 4), ("suspicious transactions", 4),
        ("transaction monitoring", 4), ("risk reporting", 3),
        ("risk model", 3), ("credit risk", 3), ("operational risk", 3),
    ],
    "On-chain & Wallet Analytics": [
        ("on-chain", 6), ("onchain", 6), ("wallet", 4), ("blockchain", 5),
        ("etherscan", 5), ("address clustering", 5), ("token flows", 5),
        ("crypto", 3), ("web3", 3), ("nft", 2),
    ],
    "DeFi & Liquidity Analytics": [
        ("defi", 6), ("tvl", 5), ("total value locked", 5),
        ("protocol analytics", 6), ("liquidity pool", 4),
        ("yield farming", 5), ("staking", 3), ("liquidity", 2),
    ],
    "Market & Investment Research": [
        ("market research", 4), ("investment research", 5),
        ("market intelligence", 5), ("due diligence", 4), ("m&a", 4),
        ("competitor analysis", 3), ("investment", 2),
        ("equity research", 4), ("valuation", 2), ("industry research", 3),
    ],
    "Dashboard & BI": [
        ("dashboard", 3), ("power bi", 3), ("tableau", 3), ("looker", 3),
        ("business intelligence", 3), ("kpi", 2), ("report", 1),
        ("data visualization", 2), ("visualization", 2),
    ],
    "Compliance & AML Monitoring": [
        ("aml", 5), ("kyc", 5), ("anti-money laundering", 6),
        ("sanctions", 4), ("regulatory reporting", 4),
        ("suspicious activity", 4), ("compliance", 2), ("gdpr", 2),
    ],
    "Credit & Lending Analytics": [
        ("credit risk", 5), ("loan", 4), ("lending", 4),
        ("underwriting", 4), ("portfolio monitoring", 3), ("basel", 4),
    ],
    "General Data Analysis": [
        ("data analysis", 2), ("data analyst", 2), ("analytics", 1),
        ("reporting", 1), ("sql", 1), ("excel", 1), ("python", 1),
    ],
}

PROBLEM_SERVICES: dict[str, tuple[str, str]] = {
    "Reconciliation & Settlement": (
        "Payment & Settlement Reconciliation Pipeline",
        "Standardize payment/settlement data, match transactions across systems and produce an exception report for finance teams.",
    ),
    "Financial Reporting & Automation": (
        "Automated Financial Reporting Suite",
        "Replace manual spreadsheet reporting with a validated, repeatable reporting workflow and consistent metric definitions.",
    ),
    "Data Quality & Validation": (
        "Financial Data Quality Control System",
        "Detect duplicates, missing values and mismatches, then establish validation rules and a monitoring report.",
    ),
    "Data Pipeline & Integration": (
        "Financial Data Integration Pipeline",
        "Ingest, combine and model data from multiple platforms into a reliable warehouse-ready dataset.",
    ),
    "Fraud & Risk Analytics": (
        "Fraud & Risk Monitoring Analytics",
        "Build anomaly detection, transaction monitoring and risk reporting for suspicious activity.",
    ),
    "On-chain & Wallet Analytics": (
        "On-chain Wallet & Flow Analytics",
        "Decode wallet activity, trace asset flows and build address-level analytics for crypto intelligence.",
    ),
    "DeFi & Liquidity Analytics": (
        "DeFi Protocol & Liquidity Dashboard",
        "Track TVL, liquidity, volumes and protocol performance with a decision-ready dashboard.",
    ),
    "Market & Investment Research": (
        "Investment Data Intelligence Service",
        "Collect, structure and analyze market/investment data for research, M&A and due diligence.",
    ),
    "Dashboard & BI": (
        "Executive Analytics Dashboard",
        "Turn fragmented operational data into a governed KPI dashboard for recurring decision-making.",
    ),
    "Compliance & AML Monitoring": (
        "AML / KYC Data & Monitoring Pipeline",
        "Structure compliance data, monitor suspicious patterns and automate regulatory reporting.",
    ),
    "Credit & Lending Analytics": (
        "Credit Risk & Loan Portfolio Analytics",
        "Analyze credit exposure, portfolio performance and risk indicators for lending decisions.",
    ),
    "General Data Analysis": (
        "General Data Analysis & Reporting",
        "Clean, analyze and report on business data for ad-hoc or recurring decisions.",
    ),
}

PROBLEM_ORDER = list(PROBLEM_RULES.keys())
PROPOSAL_ORDER = ["0–5", "6–10", "11–20", "21–50", "50+", "Unknown"]
PROPOSAL_MID = {"0–5": 2.5, "6–10": 8, "11–20": 15.5, "21–50": 35, "50+": 65}
PROPOSAL_MANAGEABILITY = {"0–5": 100, "6–10": 82, "11–20": 62, "21–50": 35, "50+": 12}
PROPOSAL_MAP = {
    "less than 5": "0–5", "fewer than 5": "0–5",
    "5 to 10": "6–10",
    "10 to 15": "11–20", "15 to 20": "11–20",
    "20 to 50": "21–50",
    "50+": "50+",
}

HIGH_VALUE_HOURLY = 50.0
HIGH_VALUE_FIXED = 2000.0
LOW_COMPETITION = {"0–5", "6–10"}

DURATION_SCORE = {
    "More than 6 months": 100,
    "3 to 6 months": 75,
    "1 to 3 months": 50,
    "Less than 1 month": 20,
}

CURRENCY_MULTIPLIER = {"K": 1_000, "M": 1_000_000}


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _compile(phrase: str) -> re.Pattern:
    escaped = re.escape(phrase.lower())
    if re.fullmatch(r"[a-z0-9]+", phrase):
        suffix = "s?" if not phrase.endswith("s") else ""
        return re.compile(rf"\b{escaped}{suffix}\b")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")


_COMPILED_NICHE = {
    industry: {
        sub: {
            "label": info[0],
            "keywords": [(_compile(kw), weight) for kw, weight in info[1]],
        }
        for sub, info in subs.items()
    }
    for industry, subs in NICHE_RULES.items()
}
_COMPILED_PROBLEMS = {name: [(_compile(kw), weight) for kw, weight in rules] for name, rules in PROBLEM_RULES.items()}


def _parse_money_text(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper().replace(",", "").replace("$", "")
    if text in {"", "NAN"}:
        return np.nan
    match = re.search(r"([\d.]+)\s*([KM]?)", text)
    if not match:
        return np.nan
    amount = float(match.group(1))
    amount *= CURRENCY_MULTIPLIER.get(match.group(2), 1)
    return amount


def _parse_relative(text, reference: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(text) or pd.isna(reference):
        return pd.NaT
    value = str(text).strip().lower()
    if value in {"just posted", "today"}:
        return reference
    if value == "yesterday":
        return reference - pd.to_timedelta(1, unit="D")
    if value == "last week":
        return reference - pd.to_timedelta(7, unit="D")
    if value == "last month":
        return reference - pd.to_timedelta(30, unit="D")
    if value == "last quarter":
        return reference - pd.to_timedelta(90, unit="D")
    match = re.search(r"(\d+)\s*(minute|hour|day|week|month|quarter)", value)
    if not match:
        return pd.NaT
    amount = int(match.group(1))
    unit = match.group(2)
    factors = {"minute": 1 / 1440, "hour": 1 / 24, "day": 1, "week": 7, "month": 30, "quarter": 90}
    return reference - pd.to_timedelta(amount * factors[unit], unit="D")


def _split_skills(value) -> list[str]:
    if pd.isna(value):
        return []
    raw = [s.strip() for s in str(value).split(",")]
    return [s for s in raw if s and not re.fullmatch(r"\+\d+", s)]


def _hits(text: str, patterns) -> float:
    return sum(weight for pattern, weight in patterns if pattern.search(text))


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def prepare_data(path: str | Path = "mkb.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["job_id"]).copy()
    df["job_id"] = df["job_id"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["job_id"], keep="first").reset_index(drop=True)

    numeric = [
        "hourly_min", "hourly_max", "fixed_budget", "weekly_budget",
        "client_total_spent", "client_reviews", "client_rating", "freelancers_to_hire",
    ]
    for col in numeric:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["job_type"] = df["job_type"].fillna("Unknown").str.strip()
    df["client_payment_verified"] = df["client_payment_verified"].fillna("no").str.lower()
    df["payment_verified"] = df["client_payment_verified"].eq("yes")

    df["proposal_bucket"] = (
        df["proposals"].fillna("Unknown").astype(str).str.strip().str.lower().map(PROPOSAL_MAP).fillna("Unknown")
    )
    df["proposal_mid"] = df["proposal_bucket"].map(PROPOSAL_MID)
    df["low_competition"] = df["proposal_bucket"].isin(LOW_COMPETITION)

    df["hourly_mid"] = df[["hourly_min", "hourly_max"]].mean(axis=1, skipna=True)
    df["fixed_budget"] = df["fixed_budget"].where(df["fixed_budget"] > 0)
    df["economic_value"] = np.where(df["job_type"].eq("Fixed price"), df["fixed_budget"], df["hourly_mid"])
    df["rate_label"] = np.where(
        df["job_type"].eq("Fixed price"),
        df["fixed_budget"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "n/a"),
        df["hourly_mid"].map(lambda v: f"${v:,.0f}/h" if pd.notna(v) else "n/a"),
    )
    df["high_value"] = np.where(
        df["job_type"].eq("Fixed price"),
        df["fixed_budget"] >= HIGH_VALUE_FIXED,
        df["hourly_mid"] >= HIGH_VALUE_HOURLY,
    )
    df.loc[df["economic_value"].isna(), "high_value"] = False

    # Client spend: prefer the exact numeric field, fall back to the displayed band.
    parsed_spend = df["client_total_spent_text"].map(_parse_money_text)
    df["client_spent"] = df["client_total_spent"].where(df["client_total_spent"].notna(), parsed_spend)
    df["client_spent_known"] = df["client_spent"].notna()

    df["long_term"] = df["duration"].eq("More than 6 months")
    df["recurring_score"] = df["duration"].map(DURATION_SCORE)
    workload_bonus = df["workload"].fillna("").str.contains("30\\+", regex=True)
    df["recurring_score"] = df["recurring_score"].where(
        df["recurring_score"].notna(), np.where(workload_bonus, 70, np.nan)
    )

    # Dates -----------------------------------------------------------------
    scraped = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True).dt.tz_localize(None)
    published = pd.to_datetime(df["published_at"], errors="coerce", utc=True).dt.tz_localize(None)
    created = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.tz_localize(None)
    estimated = [
        _parse_relative(rel, ref) for rel, ref in zip(df["posted_relative"], scraped)
    ]
    estimated = pd.Series(estimated, index=df.index)

    df["posted_date"] = published
    df["date_source"] = np.where(published.notna(), "Exact (published_at)", "Missing")
    mask = df["posted_date"].isna() & created.notna()
    df.loc[mask, "posted_date"] = created[mask]
    df.loc[mask, "date_source"] = "Exact (created_at)"
    mask = df["posted_date"].isna() & estimated.notna()
    df.loc[mask, "posted_date"] = estimated[mask]
    df.loc[mask, "date_source"] = "Estimated (relative)"

    last_scrape = scraped.max()
    df["last_scrape"] = last_scrape
    df["days_old"] = (last_scrape - df["posted_date"]).dt.total_seconds() / 86400
    df.loc[df["days_old"] < 0, "days_old"] = 0

    df["fresh_7d"] = df["days_old"] <= 7.5
    df["fresh_30d"] = df["days_old"] <= 31
    df["fresh_bucket"] = pd.cut(
        df["days_old"],
        bins=[-1, 2, 7, 14, 30, 90, np.inf],
        labels=["0–2 days", "3–7 days", "8–14 days", "15–30 days", "31–90 days", "90+ days"],
    )
    df["fresh_bucket"] = df["fresh_bucket"].cat.add_categories("Unknown").fillna("Unknown")

    # Skills ----------------------------------------------------------------
    df["skill_list"] = df["skills"].map(_split_skills)
    df["skill_count"] = df["skill_list"].map(len)

    # Text fields -----------------------------------------------------------
    df["title"] = df["title"].fillna("Untitled")
    df["description"] = df["description"].fillna("")
    df["title_l"] = df["title"].str.lower()
    df["skills_l"] = df["skills"].fillna("").str.lower()
    df["desc_l"] = df["description"].str.lower()
    df["text"] = (df["title"] + " " + df["description"] + " " + df["skills"].fillna("")).str.lower()

    _classify(df)
    _score_jobs(df)
    return df


def _classify(df: pd.DataFrame) -> None:
    industries, subs, niches, raws, confidences, relevant = [], [], [], [], [], []
    primary_problems, all_problems = [], []

    for row in df.itertuples():
        candidates = []
        for industry, sub_map in _COMPILED_NICHE.items():
            for sub, info in sub_map.items():
                t_hits = _hits(row.title_l, info["keywords"])
                s_hits = _hits(row.skills_l, info["keywords"])
                d_hits = _hits(row.desc_l, info["keywords"])
                desc_distinct = sum(1 for pattern, _ in info["keywords"] if pattern.search(row.desc_l))
                score = 3.0 * t_hits + 2.0 * s_hits + d_hits
                support = t_hits > 0 or s_hits > 0
                candidates.append({
                    "score": score, "industry": industry, "sub": sub, "label": info["label"],
                    "support": support, "desc_distinct": desc_distinct,
                })

        supported = [c for c in candidates if c["support"] and c["score"] >= 4.0]
        if supported:
            best = max(supported, key=lambda c: c["score"])
            is_relevant = True
        else:
            best = max(candidates, key=lambda c: c["score"])
            # Description-only matches need several independent signals to qualify.
            is_relevant = best["score"] >= 12.0 and best["desc_distinct"] >= 3

        if not is_relevant and best["score"] < 2.5:
            best = {"score": best["score"], "industry": "Other", "sub": "General Analytics",
                    "label": "General Analytics"}
        elif not is_relevant:
            best = {"score": best["score"], "industry": "Other", "sub": "General Analytics",
                    "label": "General Analytics"}

        raw, industry, sub = best["score"], best["industry"], best["sub"]
        industries.append(industry)
        subs.append(sub)
        niches.append(f"{industry} — {sub}")
        raws.append(raw)
        confidences.append(float(np.clip(raw / 12.0 * 100, 0, 100)))
        relevant.append(bool(is_relevant))

        problem_scores = {}
        for problem, patterns in _COMPILED_PROBLEMS.items():
            problem_scores[problem] = (
                3.0 * _hits(row.title_l, patterns)
                + 2.0 * _hits(row.skills_l, patterns)
                + 1.0 * _hits(row.desc_l, patterns)
            )
        winners = [p for p, s in sorted(problem_scores.items(), key=lambda kv: kv[1], reverse=True) if s >= 3.0]
        if not winners:
            winners = ["General Data Analysis"]
        primary_problems.append(winners[0])
        all_problems.append(winners[:4])

    df["industry"] = industries
    df["sub_niche"] = subs
    df["niche"] = niches
    df["niche_raw"] = raws
    df["niche_confidence"] = confidences
    df["relevant"] = relevant
    df["problem_primary"] = primary_problems
    df["problems"] = all_problems
    df["service"] = df["problem_primary"].map(lambda p: PROBLEM_SERVICES.get(p, ("Other", ""))[0])
    df["service_description"] = df["problem_primary"].map(lambda p: PROBLEM_SERVICES.get(p, ("Other", ""))[1])


def _score_jobs(df: pd.DataFrame) -> None:
    # Economic percentile within contract type.
    df["economic_pct"] = np.nan
    for job_type in df["job_type"].unique():
        mask = df["job_type"].eq(job_type) & df["economic_value"].notna()
        if mask.sum() > 1:
            df.loc[mask, "economic_pct"] = df.loc[mask, "economic_value"].rank(pct=True) * 100
        elif mask.sum() == 1:
            df.loc[mask, "economic_pct"] = 60.0

    # Client quality.
    spent_pct = pd.Series(np.nan, index=df.index)
    known = df["client_spent_known"]
    if known.sum() > 1:
        spent_pct[known] = df.loc[known, "client_spent"].rank(pct=True) * 100
    elif known.sum() == 1:
        spent_pct[known] = 60.0
    df["client_spent_pct"] = spent_pct

    rating = df["client_rating"].where(df["client_rating"] > 0)
    reviews = df["client_reviews"].fillna(0)
    rating_score = (rating / 5 * 100).fillna(50)
    reliability = (reviews / 10).clip(upper=1)
    rating_score = rating_score * reliability + 50 * (1 - reliability)

    verified_score = np.where(df["payment_verified"], 100, 25)
    client_parts = pd.DataFrame({
        "verified": verified_score,
        "spent": spent_pct,
        "rating": rating_score,
    })
    weights = pd.Series({"verified": 0.40, "spent": 0.40, "rating": 0.20})
    client_num = (client_parts * weights).sum(axis=1, skipna=True)
    client_den = (client_parts.notna() * weights).sum(axis=1)
    df["client_quality"] = (client_num / client_den.replace(0, np.nan)).fillna(50)
    df["client_quality_coverage"] = client_den * 100

    df["competition_score"] = df["proposal_bucket"].map(PROPOSAL_MANAGEABILITY)
    df["freshness_score"] = np.clip(100 - df["days_old"].fillna(45) * 2.4, 8, 100)
    df.loc[df["days_old"].isna(), "freshness_score"] = np.nan

    components = pd.DataFrame({
        "fit": df["niche_confidence"],
        "economic": df["economic_pct"],
        "client": df["client_quality"],
        "competition": df["competition_score"],
        "freshness": df["freshness_score"],
        "recurring": df["recurring_score"],
    })
    component_weights = pd.Series({
        "fit": 0.30, "economic": 0.20, "client": 0.15,
        "competition": 0.15, "freshness": 0.10, "recurring": 0.10,
    })
    weighted = (components * component_weights).sum(axis=1, skipna=True)
    coverage = (components.notna() * component_weights).sum(axis=1)
    df["job_score"] = (weighted / coverage.replace(0, np.nan)).clip(upper=100)
    df["job_score"] = df["job_score"].where(df["relevant"], np.nan)
    df["score_coverage"] = coverage * 100
    missing = components.isna()
    df["score_missing"] = missing.apply(
        lambda r: ", ".join(component_weights.index[i] for i, val in enumerate(r) if val), axis=1
    )
    df["job_rank"] = df["job_score"].rank(ascending=False, method="min").astype("Int64")
    for col in ["fit", "economic", "client", "competition", "freshness", "recurring"]:
        df[f"score_{col}"] = components[col]


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def _weighted_rating(group: pd.DataFrame) -> float:
    valid = group[(group["client_rating"] > 0) & (group["client_reviews"] > 0)]
    if valid.empty:
        return np.nan
    return float((valid["client_rating"] * valid["client_reviews"]).sum() / valid["client_reviews"].sum())


def build_niche_table(df: pd.DataFrame) -> pd.DataFrame:
    relevant = df[df["relevant"]].copy()
    if relevant.empty:
        return pd.DataFrame()

    rows = []
    for niche, g in relevant.groupby("niche"):
        hourly = g.loc[g["job_type"].eq("Hourly"), "hourly_mid"].dropna()
        fixed = g.loc[g["job_type"].eq("Fixed price"), "fixed_budget"].dropna()
        dates_known = g["days_old"].notna()
        fresh_share = g.loc[dates_known, "fresh_30d"].mean() * 100 if dates_known.any() else np.nan
        total_reviews = g["client_reviews"].fillna(0).sum()
        rows.append({
            "niche": niche,
            "industry": g["industry"].iloc[0],
            "sub_niche": g["sub_niche"].iloc[0],
            "jobs": len(g),
            "client_countries": g["client_country"].nunique(),
            "hourly_jobs": len(hourly),
            "median_hourly": hourly.median() if len(hourly) else np.nan,
            "hourly_p25": hourly.quantile(0.25) if len(hourly) else np.nan,
            "hourly_p75": hourly.quantile(0.75) if len(hourly) else np.nan,
            "fixed_jobs": len(fixed),
            "median_fixed": fixed.median() if len(fixed) else np.nan,
            "low_comp_share": g["low_competition"].mean() * 100,
            "high_comp_share": g["proposal_bucket"].eq("50+").mean() * 100,
            "median_proposals": g["proposal_mid"].median(),
            "verified_share": g["payment_verified"].mean() * 100,
            "median_spent": g["client_spent"].dropna().median() if g["client_spent"].notna().any() else np.nan,
            "rating": _weighted_rating(g),
            "reviews": total_reviews,
            "fresh_30d_share": fresh_share,
            "long_term_share": g["long_term"].mean() * 100,
            "high_value_share": g["high_value"].mean() * 100,
            "top_problem": g["problem_primary"].mode().iloc[0] if not g["problem_primary"].mode().empty else "n/a",
            "top_skills": ", ".join(pd.Series([s for lst in g["skill_list"] for s in lst]).value_counts().head(4).index),
            "job_score_median": g["job_score"].median(),
        })
    table = pd.DataFrame(rows)
    total_relevant = max(len(relevant), 1)
    table["share"] = table["jobs"] / total_relevant * 100

    def ratio(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        if valid.empty:
            return pd.Series(np.nan, index=series.index)
        peak = valid.max()
        if peak <= 0:
            return pd.Series(0.0, index=series.index)
        return (series / peak * 100).clip(0, 100)

    max_jobs = table["jobs"].max()
    table["D_demand"] = np.log1p(table["jobs"]) / np.log1p(max_jobs) * 100 if max_jobs > 0 else 0

    hourly_norm = ratio(table["median_hourly"])
    fixed_norm = ratio(table["median_fixed"])
    econ_parts = pd.DataFrame({"hourly": hourly_norm, "fixed": fixed_norm})
    econ_weights = pd.Series({"hourly": 0.55, "fixed": 0.45})
    table["E_economics"] = (econ_parts * econ_weights).sum(axis=1, skipna=True) / (
        (econ_parts.notna() * econ_weights).sum(axis=1).replace(0, np.nan)
    )
    table["E_economics"] = table["E_economics"].fillna(0)

    table["C_competition"] = table["low_comp_share"]

    spent_norm = ratio(table["median_spent"])
    rating_norm = (table["rating"] / 5 * 100).fillna(50)
    quality = pd.DataFrame({
        "verified": table["verified_share"],
        "spent": spent_norm,
        "rating": rating_norm,
    })
    q_weights = pd.Series({"verified": 0.40, "spent": 0.40, "rating": 0.20})
    table["Q_client"] = (quality * q_weights).sum(axis=1, skipna=True) / (
        (quality.notna() * q_weights).sum(axis=1).replace(0, np.nan)
    )
    table["Q_client"] = table["Q_client"].fillna(50)

    table["T_freshness"] = table["fresh_30d_share"].fillna(50)
    table["niche_score_raw"] = (
        0.30 * table["D_demand"]
        + 0.25 * table["E_economics"]
        + 0.20 * table["C_competition"]
        + 0.15 * table["Q_client"]
        + 0.10 * table["T_freshness"]
    )
    # Bayesian shrinkage: tiny niches should not outrank well-sampled ones.
    prior, strength = 45.0, 4.0
    table["niche_score"] = (
        table["jobs"] / (table["jobs"] + strength) * table["niche_score_raw"]
        + strength / (table["jobs"] + strength) * prior
    )
    table["reliability"] = np.select(
        [table["jobs"] >= 10, table["jobs"] >= 5],
        ["Strong sample", "Moderate sample"],
        default="Low sample — interpret carefully",
    )
    table = table.sort_values("niche_score", ascending=False).reset_index(drop=True)
    table["rank"] = table.index + 1
    return table


def demand_window(df: pd.DataFrame, days: int) -> dict:
    last = df["last_scrape"].max()
    if pd.isna(last):
        return {"current": 0, "previous": 0, "growth": np.nan, "has_dates": False}
    current = ((df["days_old"] >= 0) & (df["days_old"] <= days)).sum()
    previous = ((df["days_old"] > days) & (df["days_old"] <= 2 * days)).sum()
    if previous > 0:
        growth = (current - previous) / previous * 100
    else:
        growth = np.nan
    return {"current": int(current), "previous": int(previous), "growth": growth, "has_dates": True}


def build_problem_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    exploded = df.explode("problems")
    exploded = exploded[exploded["problems"].notna() & exploded["relevant"]]
    rows = []
    for problem, g in exploded.groupby("problems"):
        hourly = g.loc[g["job_type"].eq("Hourly"), "hourly_mid"].dropna()
        fixed = g.loc[g["job_type"].eq("Fixed price"), "fixed_budget"].dropna()
        service, description = PROBLEM_SERVICES.get(problem, ("Other", ""))
        rows.append({
            "problem": problem,
            "service": service,
            "service_description": description,
            "jobs": len(g),
            "share": len(g) / max(len(df[df["relevant"]]), 1) * 100,
            "median_hourly": hourly.median() if len(hourly) else np.nan,
            "median_fixed": fixed.median() if len(fixed) else np.nan,
            "low_comp_share": g["low_competition"].mean() * 100,
            "verified_share": g["payment_verified"].mean() * 100,
            "high_value_share": g["high_value"].mean() * 100,
            "fresh_jobs": int(g["fresh_30d"].sum()),
            "median_job_score": g["job_score"].median(),
            "examples": g.sort_values("job_score", ascending=False)["job_id"].head(5).tolist(),
        })
    table = pd.DataFrame(rows).sort_values(["jobs", "median_job_score"], ascending=[False, False])
    return table.reset_index(drop=True)


def build_skill_table(df: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    exploded = df.explode("skill_list").dropna(subset=["skill_list"])
    exploded = exploded[exploded["skill_list"].str.len() > 0]
    rows = []
    for skill, g in exploded.groupby("skill_list"):
        if len(g) < min_count:
            continue
        hourly = g.loc[g["job_type"].eq("Hourly"), "hourly_mid"].dropna()
        fixed = g.loc[g["job_type"].eq("Fixed price"), "fixed_budget"].dropna()
        rows.append({
            "skill": skill,
            "jobs": len(g),
            "relevant_jobs": int(g["relevant"].sum()),
            "median_hourly": hourly.median() if len(hourly) else np.nan,
            "median_fixed": fixed.median() if len(fixed) else np.nan,
            "low_comp_share": g["low_competition"].mean() * 100,
            "median_job_score": g["job_score"].median() if g["job_score"].notna().any() else np.nan,
            "verified_share": g["payment_verified"].mean() * 100,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["jobs", "median_hourly"], ascending=[False, False]).reset_index(drop=True)


def build_skill_cooccurrence(df: pd.DataFrame, top_n: int = 14) -> tuple[list[str], np.ndarray]:
    counter: dict[str, int] = {}
    for skills in df["skill_list"]:
        for skill in set(skills):
            counter[skill] = counter.get(skill, 0) + 1
    top = [s for s, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]
    index = {s: i for i, s in enumerate(top)}
    matrix = np.zeros((len(top), len(top)), dtype=int)
    for skills in df["skill_list"]:
        present = [index[s] for s in set(skills) if s in index]
        for i in present:
            for j in present:
                matrix[i, j] += 1
    return top, matrix


def extract_phrases(df: pd.DataFrame, ngram: tuple[int, int] = (2, 3), top_n: int = 20) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer

    if df.empty:
        return pd.DataFrame()
    corpus = (df["title"] + ". " + df["description"]).str.lower().tolist()
    stop_words = list(TfidfVectorizer(stop_words="english").get_stop_words())
    extra = {
        "project", "looking", "need", "needed", "work", "working", "job", "role",
        "experience", "experienced", "required", "requirements", "must", "will",
        "please", "able", "ability", "strong", "good", "great", "help", "team",
        "time", "new", "using", "use", "used", "including", "include", "based",
        "months", "month", "week", "weeks", "hour", "hours", "day", "days",
        "long", "term", "part", "full", "remote", "client", "clients", "company",
    }
    stop_words = list(set(stop_words) | extra)
    try:
        vectorizer = TfidfVectorizer(
            stop_words=stop_words, ngram_range=ngram, min_df=2, max_features=800,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
        )
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return pd.DataFrame()
    terms = np.array(vectorizer.get_feature_names_out())
    document_frequency = (matrix > 0).sum(axis=0).A1
    mean_tfidf = matrix.mean(axis=0).A1
    order = np.lexsort((terms, -mean_tfidf))[:top_n]
    rows = []
    for idx in order:
        phrase = terms[idx]
        mask = np.asarray(matrix[:, idx].todense()).ravel() > 0
        subset = df.iloc[np.where(mask)[0]]
        hourly = subset.loc[subset["job_type"].eq("Hourly"), "hourly_mid"].dropna()
        fixed = subset.loc[subset["job_type"].eq("Fixed price"), "fixed_budget"].dropna()
        rows.append({
            "phrase": phrase,
            "jobs": int(document_frequency[idx]),
            "tfidf": float(mean_tfidf[idx]),
            "median_hourly": hourly.median() if len(hourly) else np.nan,
            "median_fixed": fixed.median() if len(fixed) else np.nan,
            "low_comp_share": subset["low_competition"].mean() * 100,
        })
    return pd.DataFrame(rows)


def cluster_jobs(df: pd.DataFrame, n_clusters: int = 5, scope: str = "All jobs") -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    work = df[df["description"].str.len() > 40].copy()
    if work.empty or n_clusters < 2:
        return pd.DataFrame(), work

    corpus = (work["title"] + ". " + work["description"] + " " + work["skills"].fillna("")).str.lower()
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=2, max_features=600,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
    )
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return pd.DataFrame(), work
    n_clusters = min(n_clusters, max(2, matrix.shape[0] // 3), matrix.shape[0])
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)
    work["cluster_id"] = labels

    terms = np.array(vectorizer.get_feature_names_out())
    rows = []
    for cluster_id in range(n_clusters):
        g = work[work["cluster_id"].eq(cluster_id)]
        if g.empty:
            continue
        centroid = model.cluster_centers_[cluster_id]
        top_terms = terms[np.argsort(centroid)[::-1][:6]]
        label = " · ".join(top_terms[:3])
        hourly = g.loc[g["job_type"].eq("Hourly"), "hourly_mid"].dropna()
        fixed = g.loc[g["job_type"].eq("Fixed price"), "fixed_budget"].dropna()
        rows.append({
            "cluster_id": cluster_id,
            "cluster": label,
            "top_terms": ", ".join(top_terms),
            "jobs": len(g),
            "median_hourly": hourly.median() if len(hourly) else np.nan,
            "median_fixed": fixed.median() if len(fixed) else np.nan,
            "low_comp_share": g["low_competition"].mean() * 100,
            "fresh_jobs": int(g["fresh_30d"].sum()),
            "top_problems": ", ".join(g["problem_primary"].value_counts().head(2).index),
            "sample_titles": "; ".join(g["title"].head(3).tolist()),
            "job_ids": g["job_id"].tolist(),
        })
    clusters = pd.DataFrame(rows).sort_values("jobs", ascending=False).reset_index(drop=True)
    return clusters, work


def data_quality(df: pd.DataFrame) -> dict:
    total = len(df)
    dates = df["posted_date"].dropna()
    return {
        "rows": total,
        "unique_jobs": df["job_id"].nunique(),
        "duplicate_ids": total - df["job_id"].nunique(),
        "duplicate_titles": int(df["title"].duplicated().sum()),
        "relevant_jobs": int(df["relevant"].sum()),
        "unclassified": int((~df["relevant"]).sum()),
        "unclassified_share": (~df["relevant"]).mean() * 100 if total else 0,
        "exact_dates": int(df["date_source"].str.startswith("Exact").sum()),
        "estimated_dates": int(df["date_source"].eq("Estimated (relative)").sum()),
        "missing_dates": int(df["posted_date"].isna().sum()),
        "missing_economics": int(df["economic_value"].isna().sum()),
        "missing_economics_share": df["economic_value"].isna().mean() * 100 if total else 0,
        "missing_client_spend": int((~df["client_spent_known"]).sum()),
        "missing_duration": int(df["duration"].isna().sum()),
        "date_min": dates.min() if len(dates) else pd.NaT,
        "date_max": dates.max() if len(dates) else pd.NaT,
        "last_scrape": df["last_scrape"].max(),
        "search_queries": df["search_query"].nunique() if "search_query" in df else 0,
        "search_labels": ", ".join(sorted(df["search_label"].dropna().unique())) if "search_label" in df else "",
        "mean_confidence": df["niche_confidence"].mean(),
        "low_confidence_jobs": int((df["niche_confidence"] < 40).sum()),
        "client_verified_share": df["payment_verified"].mean() * 100,
        "long_term_share": df["long_term"].mean() * 100,
        "low_comp_share": df["low_competition"].mean() * 100,
        "unknown_proposals": int(df["proposal_bucket"].eq("Unknown").sum()),
    }


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col.startswith("_"):
            continue
        missing = df[col].isna().sum()
        rows.append({
            "field": col,
            "missing": int(missing),
            "missing_share": missing / len(df) * 100 if len(df) else 0,
            "populated": len(df) - missing,
        })
    return pd.DataFrame(rows).sort_values("missing_share", ascending=False).reset_index(drop=True)
