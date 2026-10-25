# Upwork FinTech / Finance / Crypto Market Intelligence

An explainable, local Streamlit dashboard for exploring a collected snapshot of Upwork job ads in FinTech, Finance, Banking, Compliance, Crypto, and cross-industry data work.

The project turns the raw file `mkb.csv` into an enriched analytical table, then exposes it through six pages:

1. Market Overview
2. Niche Intelligence
3. Project & Service Discovery
4. Text Mining & Keyword Intelligence
5. Clients & Competition
6. Data Quality & Collection Coverage

It is a **decision-support and research tool**, not a market forecast, a live scraper, or a production pricing engine.

## Documentation map

| File | Purpose |
| --- | --- |
| `README.md` | Setup, usage, analytical findings, and recommended next steps. |
| `artitechur.md` | Technical architecture, data flow, schemas, formulas, and extension guide. |
| `readme1.md` | Original product specification for the main dashboard. |
| `readme2.md` | Original specification for the Text Mining page. |
| `mkb.csv` | The raw job-ad snapshot used by the app. |

`readme1.md` and `readme2.md` are retained as source specifications and are not runtime dependencies.

## Snapshot at a glance

These figures describe the **current `mkb.csv` file** and will change if the dataset is re-collected.

| Metric | Value |
| --- | --- |
| Raw unique jobs | 93 |
| Relevant jobs | 35 (37.6%) |
| Unclassified / general analytics | 58 (62.4%) |
| Job types | 74 hourly, 19 fixed price |
| Collected date range | 2026-04-27 to 2026-10-24 |
| Exact dates | 10 of 93 |
| Estimated dates | 83 of 93 |
| Search query | Single query: `data analyst` |
| Location filter | United States |
| Clients in the US | 86 of 93 rows |
| Median hourly rate, all jobs | $30/h (57 known) |
| Median hourly rate, relevant jobs | $17.5/h (19 known) |
| Median fixed budget, all jobs | $100 (19 known) |
| Median fixed budget, relevant jobs | $150 (9 known) |
| Payment-verified clients, relevant | 88.6% |
| Low-competition jobs, relevant | 5.7% (0–10 proposals) |
| Long-term contracts, relevant | 51.4% |
| High-value jobs, relevant | 8.6% |
| Relevant jobs in the last 7 days | 15 |
| Relevant jobs in the last 30 days | 34 |
| Relevant jobs that are fresh and low competition | 1 |
| Median individual job score, relevant | 60.7 / 100 |

### Relevant niche distribution

| Niche | Jobs |
| --- | --- |
| Cross-industry — Data Engineering | 16 |
| Finance — Financial Operations | 8 |
| Finance — Investment Intelligence | 8 |
| Banking — Lending & Credit | 1 |
| Compliance — AML / KYC | 1 |
| FinTech — Payments | 1 |
| Crypto — On-chain / DeFi / Exchanges | 0 |

The absence of Crypto jobs is a **collection result**, not evidence that Crypto demand is zero. The dataset was produced with one broad query and one location filter.

## What the dashboard can answer

- How many relevant jobs are in the sample, and how fresh are they?
- Which niches have the strongest combination of demand, economics, competition, client quality, and freshness?
- Which business problems repeat, and what services could be packaged around them?
- Which skills, phrases, and keyword relationships appear together?
- Which client and contract characteristics make a job easier or harder to pursue?
- How complete and biased is the underlying data?

## Quick start

### Requirements

- Python 3.11 or newer; tested with Python 3.14.
- No API keys, database, or external service are required.

### Install and run

```bash
cd /home/thecodez/0/Python/DataSave/testproject

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

If the existing environment is available:

```bash
cd /home/thecodez/0/Python/DataSave/testproject
/home/thecodez/0/Python/DataSave/.venv/bin/streamlit run app.py
```

Then open `http://localhost:8501`.

The app reads `mkb.csv` from the same directory as `app.py`. To use a different snapshot, replace the file or change `DATA_PATH` in `app.py`, then refresh the app. The Streamlit cache is keyed by the file path and modification time, so editing the CSV invalidates the cached preparation automatically.

### Verified runtime

The current environment was smoke-tested with:

| Package | Installed version |
| --- | --- |
| Python | 3.14.0 |
| Streamlit | 1.59.2 |
| pandas | 2.3.3 |
| NumPy | 2.5.1 |
| Plotly | 6.5.0 |
| scikit-learn | 1.7.x |
| networkx | 3.6.1 |
| wordcloud | 1.9.4 |
| SciPy | 1.16.3 |

`requirements.txt` contains minimum versions rather than exact pins.

## How to use the dashboard

### 1. Start with the sidebar filters

The left sidebar is global and applies to every page. It filters:

- Posted date range and unknown-date inclusion
- Industry, sub-niche, problem type, and skill
- Job type, hourly rate, fixed budget, and unknown rate/budget inclusion
- Proposal bucket, client country, payment verification, contract duration
- Source search query

Use **Reset filters** to return to the full snapshot. The sidebar always shows how many of the 93 unique jobs remain in scope.

### 2. Market Overview

Use this page to establish the size and shape of the opportunity:

- KPI cards for relevant jobs, 30-day demand, fresh/low-competition jobs, rates, and high-value share
- Weekly demand trend and 7/30/90-day demand windows
- Demand by niche and demand vs. economic strength
- Top individual job opportunities

**Interpretation:** most dates are estimated from relative strings such as “3 days ago.” The weekly trend is directional, not a precise time series.

### 3. Niche Intelligence

Use this page to compare specialisation candidates:

- Niche Opportunity Score and raw score
- Score composition by demand, economics, competition, client quality, and freshness
- Rate and budget comparisons by contract type
- Niche × problem heatmap
- Drill-down tables with sample-size warnings

The adjusted score shrinks small samples toward a prior. A one-job niche can look interesting but is not reliable.

### 4. Project & Service Discovery

Use this page to translate repeated problems into service hypotheses:

- Problem → service map
- Demand × budget bubbles for hourly and fixed-price work
- Skills and skill co-occurrence
- Recurring 2–3 word phrases
- Unsupervised text clusters
- Searchable job explorer

The service names are a rule-based mapping from the problem taxonomy. They are starting points for validation, not proven product-market fit.

### 5. Text Mining & Keyword Intelligence

Use this page to understand the actual language in briefs:

- Corpus switch: relevant jobs only or all filtered jobs
- Text source: title, description, skills, or combinations
- Phrase length and minimum job threshold
- Keyword table with occurrences, coverage, TF-IDF, NPMI, distinctiveness, and trend
- Word cloud by frequency, document coverage, or distinctiveness
- Phrase mining and niche comparison presets
- Keyword co-occurrence network and association metrics
- Weekly phrase trends and emerging terms
- Context explorer with highlighted sentences, co-occurring terms, related skills, and job links

**Interpretation:** frequency is not commercial value. Generic words such as `data`, `analyst`, or `report` can dominate counts. Read the context and evidence jobs before using a phrase in an offer.

### 6. Clients & Competition

Use this page to evaluate pursuit economics:

- Proposal pressure and contract mix
- Budget vs. competition
- Client spend bands and rating/review quality
- Client quality by niche
- Client country and contract duration
- A simple revenue scenario calculator
- Fresh, low-competition job list

**Interpretation:** proposal counts are a single collection-time snapshot. A low proposal count does not imply a high win probability.

### 7. Data Quality & Collection Coverage

Read this page before making decisions:

- Row and unique-job counts
- Exact vs. estimated dates
- Missing economics and critical fields
- Classification confidence
- Coverage by niche
- Duplicate titles and unclassified jobs
- Collection parameters
- A coverage checklist
- Filtered CSV export

The export is useful for offline analysis or for handing the enriched data to another tool.

## Analytical findings from the current snapshot

These findings are intentionally cautious because the sample is small and collection-biased.

### 1. The sample is a query result, not the market

All 93 rows come from one search label and one query: `data analyst`, with `location=United States`. The dataset therefore over-represents general data analysis and US clients.

Only 35 jobs match a target niche. The remaining 58 are treated as unclassified or general analytics. That does **not** mean the other jobs are irrelevant to a business; it means they do not match the current high-specificity taxonomy.

### 2. Data Engineering is the largest relevant cluster, but partly because of the query

Cross-industry Data Engineering has 16 relevant jobs, followed by Finance Financial Operations and Finance Investment Intelligence with 8 each. Banking, Compliance, and Payments have only one job each, and Crypto has none.

This distribution is consistent with a broad analytics query. It is not evidence that Payments, AML/KYC, or DeFi are weak markets. Those areas need targeted collection.

### 3. The strongest scoring niches have small samples

Current adjusted niche scores rank:

1. Cross-industry — Data Engineering: 62.7
2. Finance — Financial Operations: 50.4
3. Banking — Lending & Credit: 50.0 (one job)
4. Finance — Investment Intelligence: 49.7
5. FinTech — Payments: 42.5 (one job)
6. Compliance — AML / KYC: 41.8 (one job)

Only Data Engineering has a strong sample by the dashboard's threshold. Finance has moderate samples. The one-job niches should be treated as hypotheses to investigate, not conclusions.

### 4. General problems dominate, but specialised problems are present

The most common extracted problems are:

| Problem | Jobs |
| --- | --- |
| General Data Analysis | 33 |
| Dashboard & BI | 17 |
| Data Pipeline & Integration | 12 |
| Market & Investment Research | 10 |
| Data Quality & Validation | 9 |
| Reconciliation & Settlement | 3 |
| Credit & Lending Analytics | 1 |
| Compliance & AML Monitoring | 1 |
| Financial Reporting & Automation | 1 |
| Fraud & Risk Analytics | 1 |

The generic problems dominate because of the query and taxonomy breadth. The specialised problems are too thin to support a niche decision yet, but they are useful leads for targeted data collection.

### 5. Pricing signals are weak

The relevant hourly median is $17.5/h from only 19 known hourly rates. The relevant fixed median is $150 from only 9 known fixed budgets. These values are not reliable market rates:

- Seven relevant jobs have no usable economics.
- Fixed and hourly amounts are not comparable.
- The sample is small and search-biased.
- The “high-value” share is only 8.6%, but high-value thresholds are fixed rules, not market benchmarks.

Use the tables for relative comparison inside this sample, not as a pricing recommendation.

### 6. Competition is high in this snapshot

Only 5.4% of all jobs and 5.7% of relevant jobs are in the 0–10 proposal bands. Most jobs have 11–50 or 50+ proposals. This makes “low competition” a scarce filter, not a default opportunity.

One relevant job is both ≤ 7 days old and low competition. That is the practical size of the quick-application pool in this snapshot.

### 7. Time trends are directional

Ten dates are exact; 83 are estimated from relative strings. The 30-day demand count is 86 jobs versus 5 in the previous 30 days, which looks like enormous growth but mostly reflects when the snapshot was collected. Do not interpret it as market growth.

The correct use of the trend page is:

- Compare coverage shares, not raw counts.
- Use the 7-day or 14-day window with care.
- Treat “new” terms as candidates for manual review.
- Check the collection dates and sample size before drawing conclusions.

### 8. Date and client data are incomplete but usable with caveats

- 89% of rows lack an exact publication timestamp.
- Client spend is missing for 4 rows, and the displayed spend is available for only 10 rows as a structured field.
- 18% of jobs lack usable economics.
- 7 duplicate titles suggest possible reposts, but the app de-duplicates by `job_id`, not by title.

## Recommended workflow

### For a first exploration (30–45 minutes)

1. Open **Market Overview** with no filters.
2. Note the relevant-job count and sample reliability.
3. Open **Niche Intelligence** and compare the adjusted score with the reliability column.
4. Open **Data Quality & Collection Coverage** and read the search-bias warning and coverage checklist.
5. Do not choose a niche from this snapshot alone.

### For a service hypothesis (1–2 hours)

1. Use **Project & Service Discovery** to identify repeated problems with at least 8–10 supporting jobs.
2. Read the sample job briefs behind the problem, not only the aggregate table.
3. Use **Text Mining** to inspect the exact phrases clients use.
4. Compare the phrase context across niches.
5. Check **Clients & Competition** for budget and proposal pressure.
6. Write a one-page service hypothesis with supporting jobs and a falsification test.

### For a text-mining review (30–60 minutes)

1. Start with **Relevant jobs** and **Title + description**.
2. Set a minimum of 2–3 jobs per term.
3. Review the phrase table sorted by NPMI or mean TF-IDF.
4. Open the context explorer for any phrase that looks commercially meaningful.
5. Check whether the phrase is specific to a niche or common across all jobs.
6. Use the co-occurrence network to find skill/problem bundles.
7. Treat the word cloud as an orientation aid only.

## What to do next

The most valuable next step is **better data**, not a more complicated model.

### Priority 1 — Expand collection coverage

Add separate queries for the target areas that the current query missed:

- Payment analytics, payment reconciliation, settlement, payout, chargeback
- Financial reporting, month-end close, general ledger, reconciliation
- Risk analytics, fraud analytics, transaction monitoring, anomaly detection
- AML, KYC, sanctions, regulatory reporting
- On-chain analytics, wallet clustering, blockchain data, token flows
- DeFi, TVL, liquidity, protocol analytics
- Crypto exchanges, order book, trading volume
- Credit risk, lending, loan portfolio, underwriting

Also vary location and remote filters, and collect multiple pages per query. Keep the query label and filters on every row so the provenance is preserved.

### Priority 2 — Capture time

Store absolute publication timestamps wherever possible. If only relative strings are available, capture the scrape time and the raw relative string together. Re-scrape the same jobs to measure proposal velocity and repost behaviour.

### Priority 3 — Validate the classification

Sample at least 20 low-confidence jobs and 20 random relevant jobs. Check whether:

- The title/skills support the assigned niche.
- Description-only matches are genuine.
- “General Analytics” is hiding a specific opportunity.
- The taxonomy is missing important language.

Refine keywords only after reviewing real briefs. Do not tune the rules to make the current sample look better.

### Priority 4 — Strengthen the economics

Capture budget fields consistently. Distinguish:

- Unknown budget from zero budget
- Hourly rate from fixed budget
- Posted budget from final contract value
- Client lifetime spend from this job's value

Do not blend hourly and fixed-price amounts in one average.

### Priority 5 — Persist and precompute

For a larger dataset, store the enriched table and text-mining outputs instead of recomputing them on every filter change. Good targets:

- `jobs_enriched` — one row per unique job
- `job_keyword_stats` — phrase-level frequency and coverage
- `job_keyword_by_niche` — niche-level distinctiveness
- `job_keyword_trends` — period-level coverage
- `job_keyword_cooccurrence` — edge list with support, confidence, lift, Jaccard
- `job_keyword_context` — sentence-level evidence with job links

The specifications in `readme2.md` describe this target schema.

### Priority 6 — Add tests and snapshots

Before expanding the dashboard:

- Add unit tests for money parsing, relative-date parsing, proposal mapping, classification thresholds, scoring renormalisation, and co-occurrence metrics.
- Add a small fixture CSV with known expected outputs.
- Add a smoke test that loads every page with empty, partial, and full filters.
- Snapshot the raw data and derived outputs so changes are comparable over time.

### Priority 7 — Only then consider richer models

Topic modelling, embeddings, or LLM-based extraction can wait. With 93 rows and one query, they would produce statistically polished but commercially fragile output. First make collection, labels, and time provenance trustworthy.

## Project structure

```text
testproject/
├── app.py                  # Entrypoint, data cache, sidebar filters, navigation
├── analysis.py             # Cleaning, taxonomy, classification, scoring, aggregations
├── textmining.py           # Tokenisation, phrases, NPMI, comparisons, edges, word clouds
├── mkb.csv                 # Raw 93-row Upwork snapshot
├── requirements.txt        # Runtime dependencies
├── .streamlit/
│   └── config.toml         # Light theme and browser settings
├── views/
│   ├── common.py           # Shared colours, charts, KPI cards, tables, formatting
│   ├── overview.py         # Market Overview
│   ├── niches.py           # Niche Intelligence
│   ├── discovery.py        # Project & Service Discovery
│   ├── textmining.py       # Text Mining & Keyword Intelligence
│   ├── clients.py          # Clients & Competition
│   └── quality.py          # Data Quality & Collection Coverage
├── readme1.md              # Original main-dashboard specification
├── readme2.md              # Original text-mining specification
├── artitechur.md           # Technical architecture and maintenance guide
└── README.md               # This file
```

## Data model in brief

`analysis.prepare_data()` returns one row per unique `job_id` with 86 columns. The most important groups are:

- Raw identifiers and source metadata: `job_id`, `title`, `url`, `search_query`, `search_label`, `search_filters`, `scraped_at`
- Contract fields: `job_type`, `hourly_min`, `hourly_max`, `fixed_budget`, `duration`, `workload`, `proposals`
- Client fields: `client_country`, `client_payment_verified`, `client_total_spent`, `client_reviews`, `client_rating`
- Date fields: `posted_date`, `date_source`, `days_old`, `fresh_7d`, `fresh_30d`
- Derived economics: `hourly_mid`, `economic_value`, `rate_label`, `high_value`, `client_spent`
- Taxonomy: `industry`, `sub_niche`, `niche`, `niche_confidence`, `relevant`
- Problem/service: `problem_primary`, `problems`, `service`, `service_description`
- Scores: `job_score`, `score_coverage`, `score_fit`, `score_economic`, `score_client`, `score_competition`, `score_freshness`, `score_recurring`

The full schema and formulas are documented in `artitechur.md`.

## Limitations and responsible use

- The dataset is a small, single-query, US-biased snapshot. It is not representative of the whole Upwork market.
- Most dates are estimated. Time-series results are directional.
- The taxonomy is rule-based and intentionally explainable. It will miss paraphrases and can produce false positives.
- The “relevant” label is a filter for this analysis, not an objective measure of commercial quality.
- The individual job score is a triage aid. It renormalises over available fields and never scores non-relevant jobs.
- The Niche Opportunity Score is a relative ranking inside this sample. Bayesian shrinkage reduces small-sample overconfidence but cannot create evidence.
- Proposal counts are a single snapshot and do not measure competitor quality.
- Text-mining metrics are descriptive. Co-occurrence is not causation.
- Client spend and budget fields are incomplete and should not be treated as revenue or market rates.
- Do not use the tool to make automated application, pricing, or hiring decisions without human review.
- If the dataset is refreshed, follow the source platform's terms and applicable privacy rules; the repository does not include a scraper.

## Troubleshooting

### The app loads but the pages are empty

Widen or reset the sidebar filters. Some combinations leave zero relevant jobs.

### The word cloud is missing

The scope may have too few terms, or the minimum job count may be too high. Lower **Min jobs per term** or switch the corpus.

### A trend looks implausible

Check the date provenance and collection window. Estimated dates and a single snapshot make trend interpretation unreliable.

### A score looks too high for a tiny niche

Read the **Reliability** column. Scores are shrunk toward a prior, but one or two jobs still cannot support a decision.

### `ModuleNotFoundError`

Install the requirements in the active environment:

```bash
python -m pip install -r requirements.txt
```

The text-mining module imports SciPy directly; it is currently available transitively through scikit-learn. Add an explicit `scipy` pin if the environment is rebuilt without that transitive dependency.

## Verification commands

Compile all Python modules:

```bash
python -m py_compile analysis.py app.py textmining.py views/*.py
```

Run the dashboard:

```bash
streamlit run app.py
```

Quick data verification:

```bash
python - <<'PY'
import analysis as a
df = a.prepare_data('mkb.csv')
print(df.shape)
print(df[['job_id']].nunique().iloc[0])
print(df[df['relevant']].groupby('niche').size().sort_values(ascending=False))
PY
```
