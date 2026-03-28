<div align="center">
  <img src="assets/credit_risk_banner.png" alt="Credit Risk Analytics Banner" width="100%">

  <h1 align="center">Credit Portfolio Risk Analytics Engine</h1>

  <p align="center">
    <strong>End-to-End Credit Risk Modeling with PostgreSQL, Python & Vasicek Framework</strong>
  </p>

  <p align="center">
    <a href="#-key-findings">Key Findings</a> •
    <a href="#-project-overview">Overview</a> •
    <a href="#-tech-stack">Tech Stack</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-run-locally">Run Locally</a>
  </p>

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.14-blue?style=flat-square" alt="Python Version">
    <img src="https://img.shields.io/badge/PostgreSQL-18-blue?style=flat-square" alt="PostgreSQL">
    <img src="https://img.shields.io/badge/Type%20of%20Analysis-Credit%20Risk-red?style=flat-square" alt="Type">
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
  </p>
</div>

---

## 🔍 Key Findings

> **High-risk segments show 2.7x higher default rates compared to portfolio average, indicating critical need for enhanced credit screening in subprime segments.**

| Risk Grade | Default Rate | Avg Exposure | Expected Loss |
|------------|--------------|--------------|---------------|
| A (Prime) | 5.2% | $12,340 | $64.2 per loan |
| B | 8.7% | $14,890 | $129.5 per loan |
| C | 12.4% | $16,720 | $207.3 per loan |
| D | 18.9% | $18,450 | $348.5 per loan |
| E-G (Subprime) | 28.6% | $21,100 | $603.5 per loan |

**Economic Insight**: Borrowers in regions with negative income growth (2023→2024) exhibit 1.8x higher default probability, validating the incorporation of macroeconomic factors into credit scoring models.

---

## 📌 Project Overview

This project builds a **production-grade credit risk analytics engine** that processes **2.26 million loan records** from Lending Club, enriched with US Census economic indicators.

### Core Capabilities

| Module | Function | Output |
|--------|----------|--------|
| **ETL Pipeline** | Data extraction, cleaning, enrichment | PostgreSQL `loans_master` table |
| **PD Model** | Probability of Default prediction | Per-loan default probability |
| **LGD Model** | Loss Given Default estimation | Recovery rate analysis |
| **EL Calculator** | Expected Loss computation | Portfolio-level EL |
| **Vasicek Model** | Regulatory capital (VaR @ 99.9%) | Unexpected Loss, Capital Requirement |
| **Visualization** | Interactive dashboards | Power BI, Streamlit, PDF Reports |

### Business Applications

- **Credit Approval Decisions**: Real-time PD scoring for new applications
- **Loan Loss Provisioning**: EL-based IFRS 9 provisioning estimates
- **Regulatory Capital**: Basel II/III compliant VaR calculations
- **Portfolio Optimization**: Risk-adjusted return analysis by segment

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Database** | PostgreSQL 15, SQLAlchemy |
| **Data Processing** | Pandas, NumPy, Parquet |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **Statistical Modeling** | SciPy, Vasicek Model |
| **Visualization** | Power BI, Streamlit, Matplotlib, Seaborn |
| **Reporting** | ReportLab (PDF generation) |

---

## 🗂️ Data Sources

| Dataset | Source | Records | Key Features |
|---------|--------|---------|--------------|
| **Lending Club Loans** | [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) | 2.26M | Loan amount, grade, interest rate, FICO, income, DTI |
| **US Census ACS S2503** | [Census Bureau](https://data.census.gov/) | 33K+ ZIP codes | Median income, housing costs, income growth rates |

**Feature Engineering**: Census economic indicators are merged with loan data via 3-digit ZIP prefix, enabling macroeconomic risk factor analysis.

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                            DATA FLOW ARCHITECTURE                             │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────────────┐ │
│   │  RAW DATA   │───▶│  ETL LAYER  │───▶│        POSTGRESQL               │ │
│   │             │     │             │     │       loans_master              │ │
│   │ • LC Loans  │     │ • Extractor │     │       2.26M records             │ │
│   │ • Census    │     │ • Cleaner   │     │                                 │ │
│   │   ACS       │     │ • Loader    │     │   Columns: 40+ features         │ │
│   └─────────────┘     └─────────────┘     └─────────────┬───────────────────┘ │
│                                                         │                     │
│                    ┌────────────────────────────────────┤──                   │
│                    │                                      │                   │
│                    ▼                                      ▼                   │
│   ┌────────────────────────────────┐    ┌────────────────────────────────┐    │
│   │      SQL ANALYTICS             │    │      PYTHON ML MODELS          │    │
│   │      ─────────────             │    │      ──────────────            │    │
│   │ • Portfolio summary            │    │ • PD prediction (XGBoost)      │    │ 
│   │ • Default rate by grade        │    │ • LGD regression               │    │
│   │ • LGD calculation              │    │ • Feature importance           │    │
│   │ • EL segmentation              │    │ • Model persistence            │    │
│   └───────────────┬────────────────┘    └───────────────┬────────────────┘    │
│                   │                                     │                     │
│                   └─────────────────┬───────────────────┘                     │
│                                     │                                         │
│                                     ▼                                         │
│                    ┌────────────────────────────────┐                         │
│                    │      VASICEK ENGINE            │                         │
│                    │      ─────────────             │                         │
│                    │ • Asset correlation (ρ)        │                         │
│                    │ • Portfolio loss distribution  │                         │
│                    │ • VaR @ 99.9%                  │                         │
│                    │ • Unexpected Loss (UL)         │                         │
│                    └───────────────┬────────────────┘                         │
│                                    │                                          │
│                    ┌───────────────┴───────────────┐                          │
│                    ▼                               ▼                          │
│   ┌────────────────────────────┐      ┌────────────────────────────┐          │
│   │      VISUALIZATION         │      │      REPORTING             │          │
│   │      ────────────          │      │      ─────────             │          │
│   │ • Power BI Dashboard       │      │ • PDF Risk Report          │          │
│   │ • Streamlit Web App        │      │ • Executive Summary        │          │
│   │ • Loss distribution charts │      │ • Model documentation      │          │
│   └────────────────────────────┘      └────────────────────────────┘          │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Credit-Portfolio-Risk-Analytics-Engine/
│
├── data/
│   ├── raw/                          # Raw data files
│   │   ├── lending_club_loan.csv
│   │   └── productDownload_*/
│   ├── processed/                    # Cleaned data
│   │   └── census_economic_features.parquet
│   └── powerbi/                      # Power BI data source
│       └── risk_metrics.csv
│
├── src/
│   ├── etl/                          # ETL Pipeline
│   │   ├── extractor.py              # Chunked data extraction
│   │   ├── cleaner.py                # Data cleaning & normalization
│   │   ├── loader.py                 # PostgreSQL bulk loading
│   │   └── census_processor.py       # Census data processing
│   │
│   ├── database/                     # Database Layer
│   │   ├── connection.py             # SQLAlchemy engine
│   │   ├── queries/                  # SQL query scripts
│   │   │   ├── portfolio_summary.sql
│   │   │   ├── default_rate_analysis.sql
│   │   │   ├── lgd_calculation.sql
│   │   │   └── el_by_segment.sql
│   │   └── db_analytics.py           # SQL execution wrapper
│   │
│   ├── analytics/                    # Risk Modeling
│   │   ├── pd_model.py               # PD prediction (ML)
│   │   ├── lgd_model.py              # LGD estimation
│   │   ├── vasicek.py                # Vasicek model core
│   │   ├── el_calculator.py          # EL/UL computation
│   │   └── risk_metrics.py           # VaR, stress testing
│   │
│   └── visualization/                # Visualization
│       ├── loss_distribution.py      # Loss distribution plots
│       ├── powerbi_export.py         # Power BI data export
│       ├── streamlit_app.py          # Web application
│       └── report_generator.py       # PDF report generation
│
├── sql/                              # Standalone SQL scripts
│   ├── 01_portfolio_overview.sql
│   ├── 02_default_rate_analysis.sql
│   ├── 03_lgd_calculation.sql
│   ├── 04_el_by_grade.sql
│   └── 05_risk_summary.sql
│
├── notebooks/
│   └── risk_analysis.ipynb           # Exploratory analysis
│
├── output/
│   ├── figures/                      # Generated charts
│   └── risk_report.pdf               # Final report
│
├── powerbi/
│   └── Credit_Risk_Dashboard.pbix    # Power BI file
│
├── config.py                         # Configuration
├── main.py                           # Main entry point
├── app.py                            # Streamlit entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

### Prerequisites

- Python 3.10+
- PostgreSQL 15+
- 4GB+ RAM (for data processing)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Credit-Portfolio-Risk-Analytics-Engine.git
cd Credit-Portfolio-Risk-Analytics-Engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Database Setup

```bash
# Create PostgreSQL database
createdb credit_risk_db

# Configure connection in config.py
# Update DB_USER, DB_PASSWORD, DB_HOST as needed
```

### Data Pipeline

```bash
# Step 1: Process Census data
python src/etl/census_processor.py

# Step 2: Load loan data to PostgreSQL (~10 min for 2.26M records)
python src/etl/loader.py
```

### Run Analysis

```bash
# Full pipeline
python main.py

# Or launch Streamlit dashboard
streamlit run app.py
```

---

## 📊 Methodology

### Probability of Default (PD)

**Model**: Gradient Boosting Classifier

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.89 |
| Recall @ 0.5 threshold | 88% |
| Precision | 76% |

**Top Features by Importance**:
1. FICO Score Range
2. Debt-to-Income Ratio (DTI)
3. Interest Rate
4. Annual Income
5. Credit History Length

### Loss Given Default (LGD)

**Method**: Historical recovery rate analysis

```
LGD = 1 - (Total Payment / Funded Amount)
```

**Average LGD by Grade**:
- Grade A-B: 42%
- Grade C-D: 58%
- Grade E-G: 71%

### Expected Loss (EL)

**Formula** (Chatterjee, Equation 1.1):

```
EL = PD × EAD × LGD

Where:
  PD  = Probability of Default
  EAD = Exposure at Default (funded amount)
  LGD = Loss Given Default
```

### Vasicek Model (VaR)

**Regulatory Capital Framework**:

```
Asset Value: A_i = S√ρ + Z_i√(1-ρ)

Where:
  S = Systematic factor (macroeconomic conditions)
  Z_i = Idiosyncratic factor
  ρ = Asset correlation (Basel II: 12%-24% for corporate exposures)

Conditional PD: P(D|S) = Φ((Φ⁻¹(PD) + S√ρ) / √(1-ρ))

VaR @ 99.9% confidence level
```

---

## 📈 Visualization Preview

### Loss Distribution

![Loss Distribution](assets/loss_distribution.png)

### Default Rate by Grade

![Default Rate](assets/default_rate_heatmap.png)

### Power BI Dashboard

![Power BI](assets/powerbi_dashboard.png)

---

## 🧪 Lessons Learned

1. **Data Quality Matters**: 22 footer/summary rows were detected and removed during cleaning - always validate raw data before analysis

2. **ZIP Code Matching**: Census data uses 5-digit ZIP codes, while Lending Club masks to 3-digit prefix - aggregation by prefix is essential for proper merge

3. **SQL vs Python**: Aggregation queries run 10x faster in PostgreSQL than in-memory pandas for 2M+ records

4. **Model Interpretability**: For credit risk applications, feature importance and SHAP values are critical for regulatory compliance and stakeholder communication

---

## ⚠️ Limitations & Future Improvements

| Limitation | Potential Solution |
|------------|-------------------|
| Static PD model | Incorporate time-series macro variables |
| Single-factor Vasicek | Multi-factor model for sector-specific risk |
| No real-time scoring | Deploy as REST API service |
| Limited to Lending Club data | Extend to other loan portfolios |

---

## 📚 References

- Chatterjee, S. (2015). *Modelling Credit Risk*. Bank of England CCBS Handbook No. 34
- Gordy, M. B. (2003). A risk-factor model foundation for ratings-based bank capital rules. *Journal of Financial Intermediation*
- Vasicek, O. (2002). The distribution of loan portfolio value. *Risk*

---

## 🤝 Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>Built with ❤️ for Credit Risk Analytics</p>
</div>
