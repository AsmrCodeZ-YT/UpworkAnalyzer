"""Text mining, phrase statistics, keyword comparison and co-occurrence."""

from __future__ import annotations

import html
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

SOURCE_OPTIONS = {
    "Title + description": "title_description",
    "Title only": "title",
    "Description only": "description",
    "Skills only": "skills",
    "Title + description + skills": "all",
}

LENGTH_OPTIONS = {
    "All lengths": (1, 3),
    "1 word": (1, 1),
    "2 words": (2, 2),
    "3 words": (3, 3),
    "1–2 words": (1, 2),
    "2–3 words": (2, 3),
}

CUSTOM_STOPWORDS = {
    "able", "ability", "across", "also", "applicant", "apply", "based",
    "bonus", "candidate", "client", "clients", "company", "currently", "day",
    "days", "eg", "etc", "experience", "experienced", "full", "great", "help",
    "hour", "hourly", "hours", "ideal", "ie", "include", "included",
    "including", "job", "long", "looking", "may", "month", "monthly", "months",
    "must", "need", "needed", "needs", "new", "one", "opportunity", "part",
    "per", "please", "plus", "preferred", "project", "projects", "remote",
    "require", "required", "requirement", "requirements", "role", "seeking",
    "should", "someone", "strong", "team", "three", "time", "times", "two",
    "use", "used", "using", "via", "want", "week", "weekly", "weeks", "will",
    "within", "work", "working", "would", "year", "years",
}
STOPWORDS = set(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#&./'\-]*")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\s*;\s*")


@dataclass
class TermSpace:
    work: pd.DataFrame
    terms: np.ndarray
    counts: sparse.csr_matrix
    presence: sparse.csr_matrix
    tfidf: sparse.csr_matrix
    index: dict[str, int]


def clean_text(value) -> str:
    text = html.unescape(str(value or ""))
    text = URL_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def raw_tokens(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_RE.findall(clean_text(text).lower()):
        token = token.strip(".-/'")
        if len(token) < 2 or token.isdigit() or token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _analyzer(ngram_range: tuple[int, int]):
    low, high = ngram_range

    def analyze(text: str) -> list[str]:
        tokens = raw_tokens(text)
        grams = []
        for n in range(low, high + 1):
            for start in range(len(tokens) - n + 1):
                window = tokens[start:start + n]
                if any(token in STOPWORDS for token in window):
                    continue
                grams.append(" ".join(window))
        return grams

    return analyze


def build_corpus(
    df: pd.DataFrame,
    relevant_only: bool = True,
    source: str = "title_description",
) -> pd.DataFrame:
    work = df[df["relevant"]].copy() if relevant_only else df.copy()
    work = work.drop_duplicates(subset=["job_id"]).reset_index(drop=True)
    title = work["title"].map(clean_text)
    description = work["description"].map(clean_text)
    skills = work["skills"].fillna("").str.replace(",", " . ", regex=False).map(clean_text)
    if source == "title":
        text = title
    elif source == "description":
        text = description
    elif source == "skills":
        text = skills
    elif source == "all":
        text = title + " . " + description + " . " + skills
    else:
        text = title + " . " + description
    work["mining_text"] = text
    work = work[work["mining_text"].str.len() > 0].reset_index(drop=True)
    return work


def build_term_space(
    work: pd.DataFrame,
    ngram_range: tuple[int, int] = (1, 3),
    max_features: int = 5000,
) -> TermSpace | None:
    if work.empty:
        return None
    n_docs = len(work)
    max_df = 0.95 if n_docs >= 5 else 1.0
    vectorizer = CountVectorizer(
        analyzer=_analyzer(ngram_range),
        lowercase=False,
        min_df=1,
        max_df=max_df,
        max_features=max_features,
        dtype=np.int32,
    )
    try:
        counts = vectorizer.fit_transform(work["mining_text"])
    except ValueError:
        return None
    terms = vectorizer.get_feature_names_out()
    if len(terms) == 0:
        return None
    presence = (counts > 0).astype(np.int8)
    dfreq = np.asarray(presence.sum(axis=0)).ravel()
    tf = counts.astype(np.float64).copy()
    tf.data = 1.0 + np.log(tf.data)
    idf = np.log((1 + n_docs) / (1 + dfreq)) + 1.0
    tfidf = tf @ sparse.diags(idf)
    norm = np.sqrt(np.asarray(tfidf.multiply(tfidf).sum(axis=1)).ravel())
    tfidf = sparse.diags(1.0 / np.maximum(norm, 1e-12)) @ tfidf
    index = {term: i for i, term in enumerate(terms)}
    return TermSpace(
        work=work,
        terms=terms,
        counts=counts.tocsr(),
        presence=presence.tocsr(),
        tfidf=tfidf.tocsr(),
        index=index,
    )


def term_table(space: TermSpace, window_days: int = 30) -> pd.DataFrame:
    work = space.work
    n_docs = len(work)
    presence = space.presence.toarray().astype(bool)
    counts = np.asarray(space.counts.sum(axis=0)).ravel()
    dfreq = presence.sum(axis=0)
    mean_tfidf = np.asarray(space.tfidf.sum(axis=0)).ravel() / np.maximum(dfreq, 1)

    niches = work["niche"].astype("category")
    categories = niches.cat.categories.to_numpy()
    codes = niches.cat.codes.to_numpy()
    onehot = np.zeros((n_docs, len(categories)), dtype=np.int32)
    onehot[np.arange(n_docs), codes] = 1
    term_niche = presence.astype(np.int32).T @ onehot
    niche_sizes = onehot.sum(axis=0)
    top_niche_idx = term_niche.argmax(axis=1)
    top_niche_count = term_niche[np.arange(len(space.terms)), top_niche_idx]
    market_coverage = dfreq / max(n_docs, 1)
    niche_coverage = top_niche_count / np.maximum(niche_sizes[top_niche_idx], 1)
    lift = np.where(market_coverage > 0, niche_coverage / np.maximum(market_coverage, 1e-9), 0)
    distinctiveness = np.clip(100 * (1 - 1 / np.maximum(lift, 1.0)), 0, 100)
    unigram_jobs = {
        term: int(dfreq[position])
        for term, position in space.index.items()
        if " " not in term
    }

    dates = pd.to_datetime(work["posted_date"], errors="coerce")
    latest = dates.max()
    current_mask = np.zeros(n_docs, dtype=bool)
    previous_mask = np.zeros(n_docs, dtype=bool)
    if pd.notna(latest):
        current_mask = ((dates > latest - pd.to_timedelta(window_days, unit="D")) & (dates <= latest)).to_numpy()
        previous_mask = (
            (dates > latest - pd.to_timedelta(2 * window_days, unit="D"))
            & (dates <= latest - pd.to_timedelta(window_days, unit="D"))
        ).to_numpy()
    current_total = int(current_mask.sum())
    previous_total = int(previous_mask.sum())
    current_counts = presence[current_mask].sum(axis=0)
    previous_counts = presence[previous_mask].sum(axis=0)

    rows = []
    for i, term in enumerate(space.terms):
        job_idx = np.flatnonzero(presence[:, i])
        job_count = int(len(job_idx))
        top_idx = top_niche_idx[i]
        current_jobs = int(current_counts[i])
        previous_jobs = int(previous_counts[i])
        current_cov = 100 * current_jobs / current_total if current_total else np.nan
        previous_cov = 100 * previous_jobs / previous_total if previous_total else np.nan
        change_pp = (
            current_cov - previous_cov
            if pd.notna(current_cov) and pd.notna(previous_cov)
            else np.nan
        )
        if pd.notna(change_pp) and pd.notna(previous_cov):
            if previous_cov > 0:
                change_pct = 100 * (current_cov - previous_cov) / previous_cov
                status = "Up" if change_pp > 0 else ("Down" if change_pp < 0 else "Flat")
            elif current_cov > 0:
                change_pct = np.nan
                status = "New"
            else:
                change_pct = 0.0
                status = "Flat"
        else:
            change_pct = np.nan
            status = "n/a"
        tokens = term.split(" ")
        npmi = np.nan
        if len(tokens) > 1 and all(token in unigram_jobs for token in tokens):
            p_phrase = job_count / max(n_docs, 1)
            p_parts = np.prod([unigram_jobs[token] / max(n_docs, 1) for token in tokens])
            if p_phrase > 0 and p_parts > 0 and p_phrase < 1:
                npmi = float(np.clip(np.log(p_phrase / p_parts) / -np.log(p_phrase), -1, 1))
        term_dates = dates.iloc[job_idx].dropna()
        rows.append({
            "phrase": term,
            "phrase_length": term.count(" ") + 1,
            "total_occurrences": int(counts[i]),
            "job_count": job_count,
            "job_coverage_pct": 100 * job_count / max(n_docs, 1),
            "avg_tfidf": float(mean_tfidf[i]),
            "npmi": npmi,
            "top_niche": categories[top_idx],
            "top_niche_jobs": int(top_niche_count[i]),
            "dominant_share": 100 * top_niche_count[i] / max(job_count, 1),
            "distinctiveness_score": float(distinctiveness[i]),
            "first_seen_at": term_dates.min() if len(term_dates) else pd.NaT,
            "last_seen_at": term_dates.max() if len(term_dates) else pd.NaT,
            "current_job_count": current_jobs,
            "previous_job_count": previous_jobs,
            "current_coverage": current_cov,
            "previous_coverage": previous_cov,
            "coverage_change_pp": change_pp,
            "trend_change_pct": change_pct,
            "trend_status": status,
            "example_jobs": "; ".join(work["title"].iloc[job_idx[:3]].astype(str).tolist()),
        })
    return pd.DataFrame(rows).sort_values(
        ["job_count", "total_occurrences"], ascending=False
    ).reset_index(drop=True)


def filter_terms(
    table: pd.DataFrame,
    min_jobs: int = 2,
    length_range: tuple[int, int] = (1, 3),
    query: str = "",
) -> pd.DataFrame:
    if table.empty:
        return table
    low, high = length_range
    out = table[
        table["job_count"].ge(min_jobs)
        & table["phrase_length"].between(low, high)
    ]
    if query.strip():
        out = out[out["phrase"].str.contains(query.strip(), case=False, regex=False)]
    return out.reset_index(drop=True)


def compare_terms(
    space: TermSpace,
    table: pd.DataFrame,
    mask_a: pd.Series,
    mask_b: pd.Series,
    min_jobs: int = 2,
    direction: str = "A",
    top_n: int = 25,
) -> pd.DataFrame:
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
    if n_a == 0 or n_b == 0:
        return pd.DataFrame()
    presence = space.presence.toarray().astype(bool)
    a = presence[mask_a].sum(axis=0)
    c = presence[mask_b].sum(axis=0)
    a1 = a + 0.5
    b1 = n_a - a + 0.5
    c1 = c + 0.5
    d1 = n_b - c + 0.5
    log_odds = np.log(a1 / b1) - np.log(c1 / d1)
    std_err = np.sqrt(1 / a1 + 1 / b1 + 1 / c1 + 1 / d1)
    z = log_odds / std_err
    coverage_a = 100 * a / n_a
    coverage_b = 100 * c / n_b
    lift = (a1 / (n_a + 1)) / (c1 / (n_b + 1))
    top_niche = table.set_index("phrase").reindex(space.terms)["top_niche"].to_numpy()
    out = pd.DataFrame({
        "phrase": space.terms,
        "jobs_a": a.astype(int),
        "jobs_b": c.astype(int),
        "coverage_a": coverage_a,
        "coverage_b": coverage_b,
        "coverage_diff_pp": coverage_a - coverage_b,
        "lift": lift,
        "log_odds_z": z,
        "top_niche": top_niche,
    })
    if direction == "B":
        out = out[(out["jobs_b"] >= min_jobs) & (out["log_odds_z"] < 0)]
        out = out.sort_values("log_odds_z", ascending=True)
    else:
        out = out[(out["jobs_a"] >= min_jobs) & (out["log_odds_z"] > 0)]
        out = out.sort_values("log_odds_z", ascending=False)
    return out.head(top_n).reset_index(drop=True)


def _term_pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", re.IGNORECASE)


def _sentence_pair_jobs(work: pd.DataFrame, terms: list[str]) -> dict[tuple[int, int], set[int]]:
    patterns = [_term_pattern(term) for term in terms]
    pair_jobs: dict[tuple[int, int], set[int]] = defaultdict(set)
    for job_idx, text in enumerate(work["mining_text"]):
        for sentence in SENTENCE_SPLIT_RE.split(text):
            present = [i for i, pattern in enumerate(patterns) if pattern.search(sentence)]
            if len(present) < 2:
                continue
            for i, j in combinations(present, 2):
                pair_jobs[(i, j)].add(job_idx)
    return pair_jobs


def term_edges(
    space: TermSpace,
    terms: list[str],
    scope: str = "Same job",
    min_jobs: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    terms = [term for term in terms if term in space.index]
    if len(terms) < 2:
        return pd.DataFrame(), pd.DataFrame()
    idx = [space.index[term] for term in terms]
    presence = space.presence.toarray().astype(bool)[:, idx]
    n_docs = len(space.work)
    jobs = presence.sum(axis=0)
    if scope == "Same sentence":
        pair_jobs = _sentence_pair_jobs(space.work, terms)
    else:
        pair_jobs = {
            (i, j): set(np.flatnonzero(presence[:, i] & presence[:, j]).tolist())
            for i, j in combinations(range(len(terms)), 2)
        }
    rows = []
    for i, j in combinations(range(len(terms)), 2):
        common = pair_jobs.get((i, j), set())
        both = len(common)
        if both < min_jobs:
            continue
        jobs_a, jobs_b = int(jobs[i]), int(jobs[j])
        rows.append({
            "phrase_a": terms[i], "phrase_b": terms[j], "scope": scope,
            "jobs_with_a": jobs_a, "jobs_with_b": jobs_b, "jobs_with_both": both,
            "support": 100 * both / max(n_docs, 1),
            "confidence_a_to_b": 100 * both / max(jobs_a, 1),
            "confidence_b_to_a": 100 * both / max(jobs_b, 1),
            "lift": both * n_docs / max(jobs_a * jobs_b, 1),
            "jaccard": 100 * both / max(jobs_a + jobs_b - both, 1),
            "job_ids": [space.work["job_id"].iloc[k] for k in sorted(common)],
        })
    edges = pd.DataFrame(rows)
    nodes = pd.DataFrame({
        "node": terms,
        "jobs": jobs,
        "top_niche": [
            space.work.loc[presence[:, k], "niche"].mode().iloc[0] if jobs[k] else "n/a"
            for k in range(len(terms))
        ],
        "node_type": "Term",
    })
    if not edges.empty:
        edges = edges.sort_values(["jobs_with_both", "lift"], ascending=False).reset_index(drop=True)
    return edges, nodes


def skill_problem_edges(
    work: pd.DataFrame,
    min_jobs: int = 2,
    top_skills: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    skill_counts = Counter(skill for skills in work["skill_list"] for skill in set(skills))
    problem_counts = Counter(problem for problems in work["problems"] for problem in set(problems))
    skills = [s for s, c in skill_counts.most_common(top_skills) if c >= min_jobs]
    problems = [p for p, c in problem_counts.most_common() if c >= min_jobs and p != "General Data Analysis"]
    if not skills or not problems:
        return pd.DataFrame(), pd.DataFrame()
    skill_set = set(skills)
    problem_set = set(problems)
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_jobs: dict[tuple[str, str], list[int]] = defaultdict(list)
    for job_idx, row in enumerate(work.itertuples()):
        job_skills = skill_set.intersection(row.skill_list)
        job_problems = problem_set.intersection(row.problems)
        for skill in job_skills:
            for problem in job_problems:
                pair_counts[(skill, problem)] += 1
                pair_jobs[(skill, problem)].append(job_idx)
    n_docs = len(work)
    rows = []
    for (skill, problem), both in pair_counts.items():
        if both < min_jobs:
            continue
        jobs_a = skill_counts[skill]
        jobs_b = problem_counts[problem]
        rows.append({
            "phrase_a": skill, "phrase_b": problem, "scope": "Skill → business problem",
            "jobs_with_a": jobs_a, "jobs_with_b": jobs_b, "jobs_with_both": both,
            "support": 100 * both / n_docs,
            "confidence_a_to_b": 100 * both / jobs_a,
            "confidence_b_to_a": 100 * both / jobs_b,
            "lift": both * n_docs / (jobs_a * jobs_b),
            "jaccard": 100 * both / (jobs_a + jobs_b - both),
            "job_ids": work["job_id"].iloc[pair_jobs[(skill, problem)]].tolist(),
        })
    nodes = pd.DataFrame(
        [{"node": skill, "jobs": skill_counts[skill], "top_niche": "Skill", "node_type": "Skill"} for skill in skills]
        + [
            {"node": problem, "jobs": problem_counts[problem], "top_niche": "Business problem", "node_type": "Business problem"}
            for problem in problems
        ]
    )
    edges = pd.DataFrame(rows)
    if not edges.empty:
        edges = edges.sort_values(["jobs_with_both", "lift"], ascending=False).reset_index(drop=True)
    return edges, nodes


def _sentence_context(text: str, phrase: str) -> str | None:
    pattern = _term_pattern(phrase)
    for sentence in SENTENCE_SPLIT_RE.split(text):
        if pattern.search(sentence):
            if len(sentence) > 520:
                match = pattern.search(sentence)
                start = max(0, match.start() - 190)
                end = min(len(sentence), match.end() + 260)
                if start > 0:
                    start = sentence.find(" ", start)
                    start = 0 if start < 0 else start + 1
                if end < len(sentence):
                    end = sentence.rfind(" ", 0, end)
                return ("… " if start > 0 else "") + sentence[start:end].strip() + (" …" if end < len(sentence) else "")
            return sentence.strip()
    return None


def context_matches(space: TermSpace, phrase: str) -> pd.DataFrame:
    if phrase not in space.index:
        return pd.DataFrame()
    idx = space.index[phrase]
    presence = space.presence.toarray().astype(bool)[:, idx]
    pattern = _term_pattern(phrase)
    rows = []
    for job_idx in np.flatnonzero(presence):
        row = space.work.iloc[job_idx]
        title = clean_text(row["title"])
        description = clean_text(row["description"])
        skills = clean_text(row["skills"])
        source = "Description"
        snippet = _sentence_context(description, phrase)
        if snippet is None:
            source = "Title"
            snippet = title if pattern.search(title) else None
        if snippet is None:
            source = "Skills"
            snippet = f"Skills: {skills}" if pattern.search(skills) else None
        if snippet is None:
            source = "Text"
            snippet = description[:320] or title
        rows.append({
            "job_id": row["job_id"],
            "title": row["title"],
            "phrase": phrase,
            "source": source,
            "context": snippet,
            "posted_date": row["posted_date"],
            "niche": row["niche"],
            "rate_label": row["rate_label"],
            "proposal_bucket": row["proposal_bucket"],
            "proposal_mid": row["proposal_mid"],
            "job_score": row["job_score"],
            "url": row["url"],
        })
    return pd.DataFrame(rows)


def cooccurring_terms(space: TermSpace, phrase: str, top_n: int = 12) -> pd.DataFrame:
    if phrase not in space.index:
        return pd.DataFrame()
    presence = space.presence.toarray().astype(bool)
    idx = space.index[phrase]
    anchor = presence[:, idx]
    both = presence.astype(np.int32).T @ anchor.astype(np.int32)
    jobs = presence.sum(axis=0)
    n_docs = len(space.work)
    jobs_anchor = int(anchor.sum())
    out = pd.DataFrame({
        "phrase": space.terms,
        "jobs_with_both": both,
        "jobs": jobs,
        "confidence": 100 * both / max(jobs_anchor, 1),
        "lift": both * n_docs / np.maximum(jobs * jobs_anchor, 1),
    })
    out = out[out["phrase"].ne(phrase) & out["jobs_with_both"].gt(0)]
    return out.sort_values(["jobs_with_both", "lift"], ascending=False).head(top_n).reset_index(drop=True)


def related_skills(space: TermSpace, phrase: str, top_n: int = 12) -> pd.DataFrame:
    if phrase not in space.index:
        return pd.DataFrame()
    presence = space.presence.toarray().astype(bool)[:, space.index[phrase]]
    matches = space.work[presence]
    counter = Counter(skill for skills in matches["skill_list"] for skill in set(skills))
    rows = [{"skill": skill, "jobs": count} for skill, count in counter.most_common(top_n)]
    return pd.DataFrame(rows)


def weekly_trend(space: TermSpace, phrase: str) -> pd.DataFrame:
    if phrase not in space.index:
        return pd.DataFrame()
    presence = space.presence.toarray().astype(bool)[:, space.index[phrase]]
    frame = pd.DataFrame({
        "posted_date": pd.to_datetime(space.work["posted_date"], errors="coerce"),
        "has_phrase": presence,
    }).dropna(subset=["posted_date"])
    if frame.empty:
        return pd.DataFrame()
    frame["week"] = frame["posted_date"].dt.floor("7D")
    grouped = frame.groupby("week", as_index=False).agg(
        jobs=("has_phrase", "size"),
        jobs_with_phrase=("has_phrase", "sum"),
    )
    grouped["coverage"] = 100 * grouped["jobs_with_phrase"] / grouped["jobs"]
    return grouped


def weighted_terms(table: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = table.copy()
    if mode == "Frequency":
        out["weight"] = out["total_occurrences"].astype(float)
    elif mode == "Document coverage":
        out["weight"] = out["job_count"].astype(float)
    else:
        out["weight"] = out["distinctiveness_score"].astype(float) + 1.0
    return out[out["weight"] > 0]


def wordcloud_image(table: pd.DataFrame, mode: str = "Frequency", max_words: int = 100):
    from wordcloud import WordCloud

    weighted = weighted_terms(table, mode).nlargest(max_words, "weight")
    if weighted.empty:
        return None
    frequencies = {
        str(phrase): float(weight)
        for phrase, weight in zip(weighted["phrase"], weighted["weight"])
    }
    if not frequencies:
        return None
    colormap = "viridis" if mode != "Distinctiveness" else "plasma"
    cloud = WordCloud(
        width=1100,
        height=380,
        background_color="white",
        max_words=max_words,
        random_state=42,
        colormap=colormap,
        prefer_horizontal=0.88,
        margin=4,
    )
    cloud.generate_from_frequencies(frequencies)
    return cloud.to_array()
