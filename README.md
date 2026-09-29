# Churn Intelligence — Telco Customer Retention Analysis

Turning a customer churn dataset into a retention decision: who is leaving, what
that is associated with, who is likely to leave next, and — given limited
retention budget — who the company should actually contact first.

> **Status:** in progress. Stage 1 of 7 complete (data acquired and validated).

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

### Known data quality issues

| Issue | Detail | Handling |
|---|---|---|
| `TotalCharges` is text | 11 rows contain a blank string, forcing the whole column to load as text | Stage 2 |
| Those 11 rows | All have `tenure = 0` and all have `Churn = No` — new customers never billed. Structurally missing, not randomly missing | Stage 2 |
| Redundant category labels | `"No internet service"` / `"No phone service"` duplicate information already in `InternetService` / `PhoneService` | Stage 2 |
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
