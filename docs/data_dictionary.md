# Data Dictionary — `data/processed/telco_clean.csv`

Produced by `src/clean_data.py` from `data/raw/telco_customer_churn.csv`.
**7,043 rows × 21 columns. One row per customer. No missing values.**

All churn rates below are computed from the cleaned file. The overall baseline is
**26.54%** (1,869 churned / 5,174 retained).

---

## How to read this file

| Role | Columns | Note |
|---|---|---|
| **Identifier** | `customerID` | Never use as a model feature |
| **Target** | `Churn` | What we predict |
| **Features** | the other 19 | Available at prediction time |

---

## Identifier

### `customerID` — string
Unique customer key, e.g. `7590-VHVEG`. Unique across all 7,043 rows.

**Must be excluded from modeling features.** It is arbitrary and carries no
information about churn; including it invites the model to memorise individuals.
It is kept in the cleaned file because Stage 3 (SQL joins) and Stage 6
(customer-level risk scores) both need to identify customers.

---

## Target

### `Churn` — `"Yes"` / `"No"`
Whether the customer left within the last month.

| Value | n | Share |
|---|---|---|
| No | 5,174 | 73.46% |
| Yes | 1,869 | 26.54% |

**Class imbalance.** A model predicting "nobody churns" scores **73.46% accuracy**
and identifies zero at-risk customers. This is why Stage 6 evaluates on
precision, recall, F1, ROC-AUC and PR-AUC rather than accuracy.

---

## Demographics

| Column | Values | Churn rate | Notes |
|---|---|---|---|
| `gender` | Male (3,555) / Female (3,488) | 26.2% / 26.9% | **0.7pp gap — essentially no association.** A useful negative result: the obvious demographic split explains nothing. |
| `SeniorCitizen` | 0 (5,901) / 1 (1,142) | 23.6% / **41.7%** | Seniors churn at ~1.8× the rate. `1` = senior. Kept in its original `0`/`1` coding — see Cleaning Decision 3. |
| `Partner` | No (3,641) / Yes (3,402) | 33.0% / 19.7% | |
| `Dependents` | No (4,933) / Yes (2,110) | 31.3% / 15.5% | Household ties track with retention. `Partner` and `Dependents` are correlated with each other — a Stage 5 concern. |

---

## Account

### `tenure` — integer, months
How many months the customer has been with the company. **Range 0–72, median 29, mean 32.4.**

Strongly associated with churn. Two values deserve attention:

- **`tenure = 1`: 613 customers, 62.0% churn rate** — the highest-risk group in the
  dataset, more than double baseline.
- **`tenure = 0`: 11 customers.** These have no observed outcome. No customer with
  `tenure = 0` can be labelled `Churn = Yes`, because churn is measured over a
  completed billing period — the minimum tenure among all churners is 1. Their
  `"No"` is **definitional, not measured**. See Cleaning Decision 1.

### `Contract` — three levels
| Value | n | Churn rate |
|---|---|---|
| Month-to-month | 3,875 | **42.7%** |
| One year | 1,473 | 11.3% |
| Two year | 1,695 | **2.8%** |

**The strongest single association in the dataset — a 15× spread.** Interpret with
care: this is not evidence that moving a customer onto a two-year contract would
reduce their churn risk. Customers who expect to stay are the ones who *choose*
long contracts (self-selection), and a contract makes leaving contractually
harder *during* its term. The causal question requires an experiment (Stage 7).

### `PaperlessBilling` — Yes (4,171, 33.6%) / No (2,872, 16.3%)

### `PaymentMethod` — four levels
| Value | n | Churn rate |
|---|---|---|
| Electronic check | 2,365 | **45.3%** |
| Mailed check | 1,612 | 19.1% |
| Bank transfer (automatic) | 1,544 | 16.7% |
| Credit card (automatic) | 1,522 | 15.2% |

Electronic check stands out sharply. The two *automatic* methods behave almost
identically (15.2% / 16.7%), suggesting the real distinction is **automatic vs.
manual payment**, not the specific instrument — a candidate engineered feature
for Stage 5. Whether electronic check is a *cause* or simply a marker of a
less-committed customer segment is unresolved by this data.

---

## Charges

### `MonthlyCharges` — float, USD
Current monthly bill. **Range \$18.25–\$118.75, median \$70.35.**

### `TotalCharges` — float, USD
Total billed over the customer's lifetime. **Range \$0.00–\$8,684.80, median \$1,394.55.**

> **Known multicollinearity — see Cleaning Decision 4.**
> `corr(TotalCharges, tenure × MonthlyCharges) = 0.9996`.
> Regressing `TotalCharges ~ tenure + MonthlyCharges` gives **R² = 0.895, VIF = 9.5**.
>
> The relationship is near-exact but not exact (median absolute error 2.0%, max
> 45.1%), because `MonthlyCharges` is the *current* rate while `TotalCharges`
> accumulated across a history of plan and price changes. **That residual is
> itself potentially meaningful** — it measures how a customer's spending has
> drifted over their lifetime — and is a candidate engineered feature.
>
> All three columns are retained. The feature-selection decision belongs to Stage 5.

---

## Services

`PhoneService` — Yes (6,361, 26.7%) / No (682, 24.9%). Nearly no association.

`MultipleLines` — Yes (2,971, 28.6%) / No (3,390, 25.0%) / No phone service (682, 24.9%).
*Three levels, unchanged — see Cleaning Decision 2.*

### `InternetService` — three levels
| Value | n | Churn rate |
|---|---|---|
| Fiber optic | 3,096 | **41.9%** |
| DSL | 2,421 | 19.0% |
| No | 1,526 | **7.4%** |

**This column carries a large amount of the signal in the dataset**, including all
of the information formerly duplicated across the six add-on columns below.
Fiber-optic customers churn at 2.2× the rate of DSL customers despite fiber being
the premium product — a finding worth investigating in Stage 4 (price? service
quality? competitive market?).

### Add-on services — `"Yes"` / `"No"` / `"No internet service"`
*All six keep their original three levels — see Cleaning Decision 2.*

| Column | Yes (churn) | No (churn) | No internet service (churn) |
|---|---|---|---|
| `OnlineSecurity` | 2,019 (14.6%) | 3,498 (41.8%) | 1,526 (7.4%) |
| `TechSupport` | 2,044 (15.2%) | 3,473 (41.6%) | 1,526 (7.4%) |
| `OnlineBackup` | 2,429 (21.5%) | 3,088 (39.9%) | 1,526 (7.4%) |
| `DeviceProtection` | 2,422 (22.5%) | 3,095 (39.1%) | 1,526 (7.4%) |
| `StreamingTV` | 2,707 (30.1%) | 2,810 (33.5%) | 1,526 (7.4%) |
| `StreamingMovies` | 2,732 (29.9%) | 2,785 (33.7%) | 1,526 (7.4%) |

The `"No internet service"` column is the same 1,526 customers in all six — it is identical to
`InternetService = "No"`.

Note the split in direction: the four **protective/support** services are
associated with *lower* churn, while the two **streaming** services are associated
with *higher* churn. Streaming customers are also concentrated in fiber, so this
may be confounded — a question for Stage 4.

**Caveat for Stage 5.** Because `"No internet service"` is bit-identical to
`InternetService = "No"`, naive one-hot encoding of these six columns produces
**seven duplicate dummy variables** — perfect multicollinearity, which makes
logistic regression coefficients uninterpretable. This is recorded here, not
solved; the preprocessing pipeline must handle it.

---

# Cleaning Decisions

Four issues were investigated (see `notebooks/01_data_quality_investigation.ipynb`).
**Only one of them turned out to be a cleaning problem.** The other three are real,
but they are questions about how to *represent* the data for a model rather than
about whether the data is *correct* — so they are deferred to the Stage 5 pipeline,
where they can be fitted on training data and swapped without touching this file.

The guiding rule: **cleaning fixes what is wrong and changes nothing else.**

| # | Issue | Outcome |
|---|---|---|
| 1 | `TotalCharges` blanks | **Fixed** — set to `0.0` |
| 2 | `"No internet service"` / `"No phone service"` | **Kept** — Stage 5 |
| 3 | `SeniorCitizen` `0/1` | **Kept** — Stage 5 |
| 4 | `TotalCharges` collinearity | **Kept** — Stage 5 |

No rows were dropped, no columns were added or removed, and the churn rate is
identical to the raw file.

### 1. `TotalCharges` — 11 blank values set to `0.0`

**Evidence.** The 11 rows with a blank `TotalCharges` are *exactly* the 11 rows
with `tenure = 0` (verified as identical sets, asserted in code). All 11 have
`Churn = No`; 10 of 11 are on two-year contracts against 24.1% in the full data.

**Decision.** Set to `0.0`. A customer billed for zero months has been charged
zero — this is the true value, not an estimate.

**Rejected.** Median imputation (\$1,394.55) would assert that a never-billed
customer has paid \$1,394.55. That fabricates data.

**Risk.** These customers' `Churn = No` is structurally determined, so they carry
no churn signal. At 11 of 7,043 rows (0.16%) the statistical impact is nil, but
**Stage 5 should consider excluding them from training** — a model that learns
`tenure = 0 → no churn` has learned an artifact of the observation window.

### 2. `"No internet service"` / `"No phone service"` — **kept unchanged**

**Evidence.** `"No internet service"` is identical to `InternetService = "No"` in
all six add-on columns (n=1,526, verified). `"No phone service"` is identical to
`PhoneService = "No"` (n=682). One-hot encoding them naively produces **seven
bit-identical dummy variables** — perfect multicollinearity.

**Decision.** Keep all three levels. The redundancy is real, but *how to encode a
column for a model* is a modeling decision, not a data-quality one. Nothing about
these values is **incorrect** — they are merely inconvenient for one family of
models, and they are directly useful for SQL aggregation and EDA, which need no
encoding at all.

**Why not collapse here.** Collapsing would bake one modeling choice into the file
every later stage reads, and would discard a large distinction (7.4% vs 41.8%
churn) at the point where it is most visible. Deferring keeps the choice
reversible and testable.

**Deferred to Stage 5**, where these can be compared inside a pipeline fitted on
training data only:
1. collapse to `"No"` and let `InternetService` carry the effect
2. keep three levels and drop the redundant dummy after encoding
3. keep three levels and let regularisation or a tree absorb it

### 3. `SeniorCitizen` — **kept as `0`/`1`**

**Is `0/1` a problem?** No. It is already the format every `Yes`/`No` column must
be converted to before modeling, so it is not incorrect in any sense.

**The related hazard.** If Stage 5 routes columns by dtype, `SeniorCitizen` is
selected as numeric and passed through a `StandardScaler` alongside `tenure`,
becoming `−0.44`/`+2.27` and losing its interpretability.

**Decision.** Keep the original coding. The hazard belongs to **dtype-based column
routing**, not to the data, so the fix belongs there:

```python
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [c for c in X.columns if c not in numeric_features]
```

Explicit column lists are more robust than `select_dtypes` regardless — they fail
loudly when a column is added or renamed rather than silently re-routing it.

**Note for SQL (Stage 3):** senior customers are `SeniorCitizen = 1`, not `'Yes'`.

### 4. `TotalCharges` — retained, flagged

**Decision.** Keep it. It is a genuine, correctly-recorded field with no
data-quality defect. Dropping it would be a *modeling* decision, and modeling
decisions do not belong in a cleaning script.

**Flagged for Stage 5 (VIF = 9.5).** Three options to compare there: keep all
three columns; drop `TotalCharges`; or replace it with the residual
`TotalCharges − tenure × MonthlyCharges` as a "spending drift" feature.

**Why it matters more for logistic regression than for trees.** Logistic
regression fits all coefficients jointly; when one feature is 89% explained by
two others, many coefficient combinations fit nearly equally well. Standard
errors inflate by roughly √9.5 ≈ 3.1×, and coefficients become unstable across
splits — they can even flip sign. Predictions stay accurate; **interpretation
breaks**, which defeats the purpose of choosing an interpretable model. Trees
split on one variable at a time and are unaffected in accuracy, but their feature
importances get *diluted* across the correlated group.

---

# What this script deliberately does not do

Nothing in `clean_data.py` is *fitted* to the data. Every change is a fact about
the data, not a parameter estimated from it.

Encoding, **category collapsing**, scaling, imputation, feature selection, and row
exclusion are all deferred to the **Stage 5 pipeline, fitted on the training split
only**. Fitting any of them here — across all 7,043 rows — would leak test-set
information into the model and produce optimistically biased evaluation scores.

## Carried forward to Stage 5

| What | Why it matters |
|---|---|
| `"No internet service"` × 6 + `"No phone service"` | Naive one-hot encoding creates 7 duplicate dummies (perfect collinearity) |
| `SeniorCitizen` is `0/1` | Route columns by explicit name, not `select_dtypes`, or it gets scaled |
| `TotalCharges` VIF = 9.5 | Compare: keep all three / drop it / replace with the spending-drift residual |
| 11 customers with `tenure = 0` | Label is definitional, not observed — consider excluding from training |
