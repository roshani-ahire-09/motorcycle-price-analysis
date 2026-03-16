# 🏍️ Used Motorcycle Market Analysis
### Price Prediction & Market Insights — Indian Used Bike Listings

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?style=flat&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ECC71?style=flat)

---

## 📌 Project Overview

End-to-end data analysis on **7,857 used motorcycle listings** from India.  
Starting from a messy real-world dataset, this project covers the full pipeline:  
data cleaning → feature engineering → EDA → model building → model comparison → business insights.

**The core question:** *What drives the resale price of a used motorcycle in India?*

---

## 📊 Key Results at a Glance

| Metric | Linear Regression | Random Forest |
|--------|:-----------------:|:-------------:|
| R² Score | 0.559 | **0.666** |
| MAE | ₹36,854 | **₹18,095** |
| MAPE | 51.1% | **20.4%** |
| CV R² (5-fold) | — | **0.827 ± 0.072** |

> Switching to Random Forest **reduced prediction error by 51%** — from ₹36,854 to ₹18,095 per bike.

---

## 🔍 Top 6 Findings

| # | Finding | Data |
|---|---------|------|
| 1 | **Power is the #1 price driver** | 74.7% of Random Forest feature importance |
| 2 | **First-owner premium** | 19.5% higher median price vs second-owner (₹15,200 difference) |
| 3 | **Annual depreciation** | ₹10,933/year for bikes aged 1–20 years |
| 4 | **Market is right-skewed** | Median ₹75,000 vs Mean ₹1,06,791 — premium bikes pull the average up |
| 5 | **Economy segment dominates** | 63.8% of all 7,857 listings |
| 6 | **Premium bikes cost 9.5x more** | yet represent only 3.9% of the market |

---

## 🗂️ Dataset

- **Source:** Indian used motorcycle listings (Kaggle practice dataset)
- **Size:** 7,857 rows × 8 raw columns
- **Raw columns:** `model_name`, `model_year`, `kms_driven`, `owner`, `location`, `mileage`, `power`, `price`
- **Data issues found:** Mixed formats in kms_driven, 1,867 rows with fuel mileage in wrong column, 218 model names with no engine CC, 556 location variants

---

## 🛠️ Tech Stack

```
Python 3.10+
├── pandas          — data loading, cleaning, transformation
├── numpy           — numerical operations, polyfit for depreciation
├── matplotlib      — all chart creation
├── seaborn         — chart styling
├── scikit-learn    — train_test_split, LinearRegression,
│                     RandomForestRegressor, cross_val_score,
│                     LabelEncoder, evaluation metrics
└── re              — regex for extracting engine CC and brand from text
```

---

## 📁 Project Structure

```
motorcycle-price-analysis/
│
├── motorcycle_analysis_complete.py   ← main analysis file (all sections)
├── README.md                         ← this file
│
├── Section breakdown:
│   ├── Section 1  — Load Data
│   ├── Section 2  — Data Cleaning (6 sub-steps)
│   ├── Section 3  — Feature Engineering
│   ├── Section 4  — EDA (6 charts)
│   ├── Section 5  — Prepare Shared Feature Set
│   ├── Section 6  — Model 1: Linear Regression
│   ├── Section 7  — Model 2: Random Forest
│   ├── Section 8  — Model Comparison
│   ├── Section 9  — Feature Importance
│   ├── Section 10 — Comparison Visualisations
│   └── Section 11 — Key Findings Summary
```

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/motorcycle-price-analysis.git
cd motorcycle-price-analysis
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**3. Add the dataset**  
Download the dataset from Kaggle and place it in the same folder as the `.py` file.  
Rename it to: `bikes (1).csv`

**4. Run the analysis**
```bash
python motorcycle_analysis_complete.py
```

**Output:** 7 chart images saved to your folder + printed findings summary.

---

## 🧹 Data Cleaning Highlights

The raw data had significant quality issues that required careful handling:

**kms_driven column** — 4 different formats in one column:
- `17000 Km` · `28 Kms` · `Mileage 28 Kms` (fuel mileage in wrong column!) · plain numbers
- 1,867 rows had fuel efficiency (kmpl) stored as distance — detected via regex and replaced with NaN
- 129 rows had suspicious values under 100 km (Bajaj Dominar entries with 28 km — their fuel economy, not odometer)

**engine_cc** — extracted from free-text model names using regex `(\d+)`:
- 218 model names had no numbers at all (e.g. "Harley-Davidson Fat Boy", "UM Renegade Commando")
- Handled with a 28-entry manual lookup table of real engine specs

**mileage** — applied brand-specific realistic bounds:
- Harley-Davidson: 12–30 kmpl (not 85 kmpl)
- Hero/Bajaj/TVS commuters: 35–85 kmpl
- Anything outside brand bounds → set to NaN → filled with brand median

**Location** — 556 unique city name variants standardised  
(e.g. `Bangalore` → `Bengaluru`, `Bhatinda` → `Bathinda`)

---

## 🔬 Feature Engineering

Three new features created from existing columns:

| Feature | Formula | Purpose |
|---------|---------|---------|
| `bike_age` | `2021 - model_year` | Drives depreciation analysis |
| `price_per_cc` | `price / engine_cc` | Value-for-money comparison metric |
| `brand_category` | Rule-based (Premium/Mid-Range/Economy) | Market segmentation |

---

## 🤖 Modelling Approach

Both models were trained on the **same 8 features** and **same 80/20 split** for a fair comparison:

```python
FEATURES = ['bike_age', 'kms_driven', 'engine_cc', 'power',
            'mileage', 'owner_num', 'brand_encoded', 'category_encoded']
```

**Why Random Forest outperforms Linear Regression here:**
- Price-age relationship is non-linear (steep drop years 1–3, flatter after)
- Brand interactions with other features can't be captured linearly
- RF handles the right-skewed price distribution better

**Key Random Forest hyperparameters:**
```python
RandomForestRegressor(
    n_estimators=200,      # 200 trees for stable predictions
    max_depth=15,          # prevents memorising training noise
    min_samples_split=5,   # node needs 5+ samples to split
    min_samples_leaf=2,    # each leaf needs 2+ samples
    random_state=42        # reproducible results
)
```

---

## 📈 Feature Importance

```
power            ████████████████████████████████████  74.7%
engine_cc        █████  9.7%
bike_age         ███    5.2%
category_encoded ██     4.5%
kms_driven       █      3.1%
mileage          █      1.9%
brand_encoded         0.9%
owner_num             0.2%
```

Engine **power (bhp) dominates** — it captures the buyer's fundamental question:  
*"Is this a commuter or a performance bike?"* Better than engine size, brand, or age alone.

---

## ⚠️ Model Limitations

| Limitation | What to do about it |
|------------|---------------------|
| MAPE = 20.4% (not production-ready) | Log-transform price target to handle right skew |
| Location (556 cities) not used | Cluster into Metro / Tier-1 / Tier-2 tiers |
| engine_cc min = 1cc (extraction error) | Add lower bound filter: `engine_cc >= 50` |
| No hyperparameter tuning | Run GridSearchCV on max_depth, n_estimators |
| Premium segment under-represented (3.9%) | Train separate model for premium bikes |

---

## 💡 Business Recommendations

**For Buyers:**
- Buy bikes aged 3–5 years — past steepest depreciation, still significant life
- Second-owner bikes save ~19.5% without major quality loss
- Higher power bikes (>30 bhp) typically have lower mileage (20–30 kmpl vs 50+ kmpl)

**For Sellers:**
- Sell before year 3 to maximise returns (depreciation steepest in early years)
- First ownership adds ~₹15,200 to median resale value — worth documenting

**For Platforms (OLX, Bike24 etc.):**
- Use this model as a "Fair Price" baseline (MAE ₹18,095)
- Economy segment (63.8%) is where volume lies — prioritise inventory there

---

## 👤 Author

**Roshani Dadaji Ahire**  
Data Analyst — Python · SQL · Tableau  
📧 roshani@email.com  
🔗 [LinkedIn](https://linkedin.com/in/roshaniahire)

---

*This is a portfolio project built on a Kaggle practice dataset.*  
*Findings are for analytical demonstration and should not be used for real pricing decisions.*
