# Churn Intelligence — Telco Customer Retention Analysis

Turning a customer churn dataset into a retention decision: who is leaving, what
that is associated with, who is likely to leave next, and — given limited
retention budget — who the company should actually contact first.

> **Status:** in progress. Stage 2 of 7 complete (data cleaned and documented).

---

## The business question

> How can a telecommunications company understand customer churn, identify
> customers at risk of leaving, and prioritize retention efforts?

Four questions structure the work:

1. **Who is churning?** — churn rates across contract type, tenure, services, payment method.
2. **Why are they churning?** — variables *associated* with churn, stated as association, not cause.
3. **Who is likely to churn?** — a predictive model producing customer-level churn probabilities.
4. **Who should be prioritized?** — churn risk combined with customer value, with the assumptions made explicit.

A churn model predicts *who leaves*. It does not prove that any retention offer
will change that. Interventions discussed in this project are treated as
**hypotheses to be tested by experiment**, not as established facts.

---

## Data

**IBM Telco Customer Churn** — the standard 21-column version.

- 7,043 customers, one row each, `customerID` unique
- Target: `Churn` (Yes/No)
- **Baseline churn rate: 26.54%** (1,869 churned / 5,174 retained)

The raw file is **not committed** (see `.gitignore`). To reproduce, place
`telco_customer_churn.csv` in `data/raw/` and run the validation script below.

### Data quality issues and how they were handled

Full evidence and reasoning for each decision is in
**[`docs/data_dictionary.md`](docs/data_dictionary.md)**.

Cleaning fixes what is **wrong** and changes nothing else. Of the four issues
investigated, only one was a data-quality problem; the rest are questions about how
to *represent* the data for a model, and are deferred to the Stage 5 preprocessing
pipeline where they can be fitted on training data and swapped freely.

| Issue | Detail | Resolution |
|---|---|---|
| `TotalCharges` is text | 11 rows contain a blank string, forcing the column to load as text | **Fixed** — set to `0.0`. These are exactly the `tenure = 0` rows, so zero is the true value, not an estimate |
| Those 11 rows | All `Churn = No`, and no customer with `tenure = 0` *can* be labelled churned — their label is definitional, not observed | Kept; flagged for possible exclusion from model training in Stage 5 |
| Redundant category labels | `"No internet service"` is bit-identical to `InternetService = "No"` across six columns — naive one-hot encoding yields 7 duplicate dummies | **Kept as-is.** An encoding decision, not a data-quality one — deferred to Stage 5 |
| `SeniorCitizen` coded `0/1` | Would be picked up by dtype-based column routing and scaled as a numeric feature | **Kept as-is.** Fixed in Stage 5 by naming columns explicitly instead of inferring from dtype |
| `TotalCharges` collinearity | `R² = 0.895` against `tenure + MonthlyCharges` (VIF 9.5); flips sign when fitted alongside them | **Kept as-is** — deferred to Stage 5 |
| Class imbalance | 26.5% positive class — accuracy is a misleading metric here | Stages 5–6 |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/validate_raw.py   # verifies the raw file matches the expected schema
```

---

## Repository structure

```
data/raw/          raw dataset (not committed)
data/processed/    cleaned outputs (not committed, rebuilt by script)
notebooks/         exploratory analysis
sql/               churn-rate, segmentation and revenue queries
src/               reusable scripts
dashboard/         Power BI / Tableau files
reports/           figures and written findings
```

---

## Roadmap

- [x] **1. Foundation & data validation** — environment, repo scaffold, schema checks
- [ ] **2. Cleaning & data dictionary**
- [ ] **3. SQL analysis layer** — churn rates, segmentation, revenue
- [ ] **4. EDA & statistical testing**
- [ ] **5. Feature engineering & modeling** — logistic regression, then a tree-based model
- [ ] **6. Evaluation, interpretation & threshold selection**
- [ ] **7. Retention prioritization, dashboard & findings**

---

## Findings

_To be written as the analysis produces them. Nothing is recorded here until it
has been computed from the data._
