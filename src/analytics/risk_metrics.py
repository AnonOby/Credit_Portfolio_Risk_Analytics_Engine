"""
Risk Metrics & Stress Testing Module

Provides a comprehensive set of risk metrics and stress testing scenarios
beyond the standard EL/VaR calculations. This module aggregates outputs
from PD, LGD, Vasicek, and EL models into high-level risk indicators.

Core capabilities:
    1. Concentration Risk Analysis (Herfindahl-Hirschman Index)
    2. Risk-Adjusted Return Metrics (RAROC, expected return vs EL)
    3. Stress Testing (GDP shock, interest rate shift, unemployment)
    4. Migration Analysis (grade transition matrix)
    5. Portfolio Quality Indicators (weighted avg PD/LGD, risk-weighted assets)

References:
    - Basel Committee on Banking Supervision (2006). "International
      Convergence of Capital Measurement and Capital Standards."
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.connection import get_engine
import config


# ==================================================================
# Configuration
# ==================================================================

OUTPUT_DIR = config.BASE_DIR / 'output'
STRESS_TEST_PATH = OUTPUT_DIR / 'stress_test_results.csv'
MIGRATION_PATH = OUTPUT_DIR / 'grade_migration_matrix.csv'
CONCENTRATION_PATH = OUTPUT_DIR / 'concentration_risk.csv'


class RiskMetrics:
    """
    Comprehensive risk metrics and stress testing engine.

    Aggregates data from the database and trained models to compute
    portfolio-level risk indicators and scenario analyses.
    """

    def __init__(self):
        self.df = None

    # ----------------------------------------------------------
    # Data Loading
    # ----------------------------------------------------------

    def load_portfolio(self):
        """
        Load portfolio with historical PD, LGD, and EAD.

        Returns:
            pd.DataFrame: Portfolio data with risk metrics.
        """
        print("Loading portfolio with risk metrics...")

        pd_query = """
            WITH mature_loans AS (
                SELECT * FROM loans_master
                WHERE loan_status IN (
                    'Fully Paid', 'Charged Off', 'Default',
                    'Does not meet the credit policy. Status:Fully Paid',
                    'Does not meet the credit policy. Status:Charged Off'
                )
            )
            SELECT
                grade,
                SUM(CASE WHEN loan_status IN ('Charged Off', 'Default',
                    'Does not meet the credit policy. Status:Charged Off')
                    THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS pd
            FROM mature_loans
            GROUP BY grade
        """

        lgd_query = """
            SELECT
                grade,
                AVG(1 - (total_pymnt / funded_amnt)) AS lgd
            FROM loans_master
            WHERE loan_status IN ('Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off')
            AND funded_amnt > 0
            GROUP BY grade
        """

        loan_query = """
            SELECT
                id, grade, sub_grade, funded_amnt, loan_amnt,
                term, int_rate, purpose, home_ownership,
                annual_inc, dti, issue_d, loan_status, zip_code,
                fico_range_low, fico_range_high
            FROM loans_master
        """

        df = pd.read_sql(loan_query, get_engine())
        pd_df = pd.read_sql(pd_query, get_engine())
        lgd_df = pd.read_sql(lgd_query, get_engine())

        pd_lookup = pd_df.set_index('grade')['pd'].to_dict()
        lgd_lookup = lgd_df.set_index('grade')['lgd'].to_dict()

        df['pd'] = df['grade'].map(pd_lookup).fillna(0.20)
        df['lgd'] = df['grade'].map(lgd_lookup).fillna(0.46)
        df['el'] = df['pd'] * df['lgd'] * df['funded_amnt']
        df['el_rate'] = df['pd'] * df['lgd']

        print("   -> Loaded {:,} loans with PD/LGD/EL.".format(len(df)))
        self.df = df
        return df

    # ----------------------------------------------------------
    # 1. Concentration Risk
    # ----------------------------------------------------------

    def compute_concentration_risk(self):
        """
        Measure portfolio concentration using Herfindahl-Hirschman Index (HHI).

        HHI = sum(weight_i^2), where weight_i = exposure_i / total_exposure.
        HHI ranges from 1/N (perfectly diversified) to 1.0 (fully concentrated).

        Computed across multiple dimensions: grade, purpose, term.

        Returns:
            pd.DataFrame: Concentration metrics per dimension.
        """
        print("\n--- Concentration Risk Analysis ---")

        if self.df is None:
            self.load_portfolio()

        total_ead = self.df['funded_amnt'].sum()

        results = []

        for dimension in ['grade', 'purpose', 'term', 'home_ownership']:
            groups = self.df.groupby(dimension)['funded_amnt'].sum()
            weights = groups / total_ead
            hhi = (weights ** 2).sum()
            hhi_normalized = (hhi - 1 / len(groups)) / (1 - 1 / len(groups))
            top3_share = weights.sort_values(ascending=False).head(3).sum()

            results.append({
                'dimension': dimension,
                'n_groups': len(groups),
                'hhi': round(hhi, 4),
                'hhi_normalized': round(hhi_normalized, 4),
                'effective_n': round(1 / hhi, 1),
                'top3_concentration': round(top3_share, 4),
                'max_single_share': round(weights.max(), 4)
            })

        result_df = pd.DataFrame(results)

        print("")
        print("   {:<18} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
            'Dimension', 'Groups', 'HHI', 'HHI Norm', 'Eff. N', 'Top3', 'Max'))
        print("   " + "-" * 72)
        for _, row in result_df.iterrows():
            print("   {:<18} {:>8} {:>8.4f} {:>8.4f} {:>8.1f} {:>7.2%} {:>7.2%}".format(
                row['dimension'],
                row['n_groups'],
                row['hhi'],
                row['hhi_normalized'],
                row['effective_n'],
                row['top3_concentration'],
                row['max_single_share']
            ))

        self.concentration = result_df
        return result_df

    # ----------------------------------------------------------
    # 2. Risk-Adjusted Return Metrics
    # ----------------------------------------------------------

    def compute_risk_adjusted_returns(self):
        """
        Compute Risk-Adjusted Return on Capital (RAROC) by grade.

        RAROC = (Interest Income - Expected Loss) / Economic Capital

        Returns:
            pd.DataFrame: RAROC by grade.
        """
        print("\n--- Risk-Adjusted Return Analysis ---")

        if self.df is None:
            self.load_portfolio()

        grade_agg = self.df.groupby('grade').agg(
            total_ead=('funded_amnt', 'sum'),
            avg_int_rate=('int_rate', 'mean'),
            avg_pd=('pd', 'mean'),
            avg_lgd=('lgd', 'mean'),
            n_loans=('id', 'count')
        ).reset_index()

        # Annual interest income (approximate)
        grade_agg['interest_income'] = grade_agg['avg_int_rate'] / 100 * grade_agg['total_ead']

        # Expected loss
        grade_agg['expected_loss'] = grade_agg['avg_pd'] * grade_agg['avg_lgd'] * grade_agg['total_ead']

        # Net income after EL
        grade_agg['net_income'] = grade_agg['interest_income'] - grade_agg['expected_loss']

        # UL factor by PD band (higher PD = proportionally more UL)
        ul_factors = []
        for pd_val in grade_agg['avg_pd']:
            if pd_val < 0.10:
                ul_factors.append(2.5)
            elif pd_val < 0.20:
                ul_factors.append(3.0)
            elif pd_val < 0.30:
                ul_factors.append(3.5)
            else:
                ul_factors.append(4.0)

        grade_agg['ul_factor'] = ul_factors
        grade_agg['unexpected_loss'] = grade_agg['expected_loss'] * grade_agg['ul_factor']
        grade_agg['economic_capital'] = grade_agg['unexpected_loss']

        # RAROC
        raroc_values = []
        for idx, row in grade_agg.iterrows():
            if row['economic_capital'] > 0:
                raroc_values.append(row['net_income'] / row['economic_capital'])
            else:
                raroc_values.append(0.0)
        grade_agg['raroc'] = raroc_values

        # Spread over EL (bps)
        spread_values = []
        for idx, row in grade_agg.iterrows():
            if row['total_ead'] > 0:
                spread_values.append((row['net_income'] / row['total_ead']) * 10000)
            else:
                spread_values.append(0.0)
        grade_agg['spread_over_el_bps'] = spread_values

        print("")
        print("   {:<6} {:>8} {:>8} {:>14} {:>14} {:>8} {:>12}".format(
            'Grade', 'Int Rate', 'EL Rate', 'Net Income', 'Econ Capital', 'RAROC', 'Spread (bps)'))
        print("   " + "-" * 75)
        for _, row in grade_agg.iterrows():
            el_rate = row['avg_pd'] * row['avg_lgd']
            print("   {:<6} {:>7.2f}% {:>7.2f}% ${:>12,.0f} ${:>12,.0f} {:>7.1f}% {:>11.0f}".format(
                row['grade'],
                row['avg_int_rate'],
                el_rate * 100,
                row['net_income'],
                row['economic_capital'],
                row['raroc'] * 100,
                row['spread_over_el_bps']
            ))

        self.risk_adjusted = grade_agg
        return grade_agg

    # ----------------------------------------------------------
    # 3. Stress Testing
    # ----------------------------------------------------------

    def run_stress_tests(self):
        """
        Run stress test scenarios on the portfolio.

        Scenarios:
            - Baseline: Current PD and LGD
            - Mild Recession: PD +30%, LGD +10%
            - Moderate Recession: PD +80%, LGD +20%
            - Severe Recession: PD +150%, LGD +35%
            - Stagflation: PD +100%, LGD +25%

        Returns:
            pd.DataFrame: EL under each scenario by grade.
        """
        print("\n--- Stress Testing ---")

        if self.df is None:
            self.load_portfolio()

        scenarios = {
            'Baseline':           {'pd_shock': 1.00, 'lgd_shock': 1.00},
            'Mild Recession':     {'pd_shock': 1.30, 'lgd_shock': 1.10},
            'Moderate Recession': {'pd_shock': 1.80, 'lgd_shock': 1.20},
            'Severe Recession':   {'pd_shock': 2.50, 'lgd_shock': 1.35},
            'Stagflation':        {'pd_shock': 2.00, 'lgd_shock': 1.25},
        }

        total_ead = self.df['funded_amnt'].sum()

        results = []
        for scenario_name, params in scenarios.items():
            pd_shocked = np.clip(self.df['pd'] * params['pd_shock'], 0, 1)
            lgd_shocked = np.clip(self.df['lgd'] * params['lgd_shock'], 0, 1)
            el_shocked = pd_shocked * lgd_shocked * self.df['funded_amnt']

            for grade in sorted(self.df['grade'].unique()):
                mask = self.df['grade'] == grade
                grade_el = el_shocked[mask].sum()
                grade_ead = self.df.loc[mask, 'funded_amnt'].sum()

                results.append({
                    'scenario': scenario_name,
                    'grade': grade,
                    'total_ead': grade_ead,
                    'pd_shocked': pd_shocked[mask].mean(),
                    'lgd_shocked': lgd_shocked[mask].mean(),
                    'el': grade_el,
                    'el_rate': grade_el / grade_ead if grade_ead > 0 else 0
                })

            results.append({
                'scenario': scenario_name,
                'grade': 'TOTAL',
                'total_ead': total_ead,
                'pd_shocked': pd_shocked.mean(),
                'lgd_shocked': lgd_shocked.mean(),
                'el': el_shocked.sum(),
                'el_rate': el_shocked.sum() / total_ead
            })

        result_df = pd.DataFrame(results)

        # Print scenario comparison at portfolio level
        print("")
        print("   Portfolio-Level Stress Test Results:")
        print("   {:<22} {:>8} {:>8} {:>16} {:>10} {:>12}".format(
            'Scenario', 'PD', 'LGD', 'Total EL', 'EL Rate', 'EL Increase'))
        print("   " + "-" * 80)

        baseline_el = 0
        for scenario in scenarios.keys():
            filtered = result_df[(result_df['scenario'] == scenario) & (result_df['grade'] == 'TOTAL')]
            if len(filtered) == 0:
                continue
            row = filtered.iloc[0]

            if scenario == 'Baseline':
                baseline_el = row['el']
                inc_str = '-'
            else:
                delta = row['el'] - baseline_el
                inc_str = "+${:,.0f}".format(delta)

            print("   {:<22} {:>7.2%} {:>7.2%} ${:>14,.0f} {:>9.2%} {:>12}".format(
                scenario,
                row['pd_shocked'],
                row['lgd_shocked'],
                row['el'],
                row['el_rate'],
                inc_str
            ))

        self.stress_results = result_df
        return result_df

    # ----------------------------------------------------------
    # 4. Grade Migration Analysis
    # ----------------------------------------------------------

    def compute_migration_matrix(self):
        """
        Build a simplified grade migration matrix using actual loan data.

        Returns:
            pd.DataFrame: Migration probability matrix.
        """
        print("\n--- Grade Migration Analysis ---")

        if self.df is None:
            self.load_portfolio()

        query = """
            SELECT
                grade,
                loan_status
            FROM loans_master
            WHERE loan_status IN (
                'Fully Paid', 'Charged Off', 'Default',
                'Does not meet the credit policy. Status:Fully Paid',
                'Does not meet the credit policy. Status:Charged Off',
                'Current', 'In Grace Period',
                'Late (16-30 days)', 'Late (31-120 days)'
            )
        """

        df_mig = pd.read_sql(query, get_engine())
        total = len(df_mig)
        print("   -> Loaded {:,} loans for migration analysis.".format(total))

        # Classify destination states
        def classify_status(status):
            if status in ('Fully Paid', 'Does not meet the credit policy. Status:Fully Paid'):
                return 'Fully Paid'
            elif status in ('Charged Off', 'Default',
                             'Does not meet the credit policy. Status:Charged Off'):
                return 'Default'
            elif status == 'Current':
                return 'Current'
            elif 'In Grace' in status:
                return 'Current'
            elif 'Late' in status:
                return 'Delinquent'
            else:
                return 'Other'

        df_mig['dest_state'] = df_mig['loan_status'].apply(classify_status)

        # Build migration matrix (%)
        crosstab = pd.crosstab(
            df_mig['grade'],
            df_mig['dest_state'],
            normalize='index'
        ).round(4) * 100

        col_order = ['Fully Paid', 'Current', 'Delinquent', 'Default', 'Other']
        col_order = [c for c in col_order if c in crosstab.columns]
        crosstab = crosstab[col_order]

        # Header row
        header = "   {:<12}".format("From \\ To")
        for col in crosstab.columns:
            header += " {:>12}".format(col)
        print(header)
        print("   " + "-" * (12 + 13 * len(crosstab.columns)))

        for grade, row in crosstab.iterrows():
            line = "   {:<12}".format(grade)
            for val in row:
                line += " {:>11.2f}%".format(val)
            print(line)

        self.migration_matrix = crosstab
        return crosstab

    # ----------------------------------------------------------
    # 5. Portfolio Quality Indicators
    # ----------------------------------------------------------

    def compute_portfolio_quality(self):
        """
        Compute portfolio quality indicators for reporting.

        Returns:
            dict: Portfolio quality indicators.
        """
        print("\n--- Portfolio Quality Indicators ---")

        if self.df is None:
            self.load_portfolio()

        df = self.df
        total_ead = df['funded_amnt'].sum()

        # Exposure-weighted metrics
        wavg_pd = np.average(df['pd'], weights=df['funded_amnt'])
        wavg_lgd = np.average(df['lgd'], weights=df['funded_amnt'])
        wavg_el = wavg_pd * wavg_lgd

        # Grade band shares
        subprime_mask = df['grade'].isin(['E', 'F', 'G'])
        subprime_share = df.loc[subprime_mask, 'funded_amnt'].sum() / total_ead

        near_prime_mask = df['grade'].isin(['C', 'D'])
        near_prime_share = df.loc[near_prime_mask, 'funded_amnt'].sum() / total_ead

        prime_mask = df['grade'].isin(['A', 'B'])
        prime_share = df.loc[prime_mask, 'funded_amnt'].sum() / total_ead

        # High-risk purpose share
        high_risk_purposes = ['small_business', 'medical', 'moving']
        high_risk_share = df.loc[df['purpose'].isin(high_risk_purposes), 'funded_amnt'].sum() / total_ead

        # FICO statistics
        fico_mid = (df['fico_range_low'] + df['fico_range_high']) / 2.0
        fico_below_660 = (fico_mid < 660).sum() / len(df)

        # DTI risk
        high_dti = (df['dti'] > 25).sum() / len(df)

        # Loan status distribution
        status_query = """
            SELECT loan_status, COUNT(*) as cnt
            FROM loans_master
            GROUP BY loan_status
            ORDER BY cnt DESC
        """
        status_df = pd.read_sql(status_query, get_engine())
        total_status = status_df['cnt'].sum()
        status_df['pct'] = status_df['cnt'] / total_status

        indicators = {
            'total_exposure': total_ead,
            'total_loans': len(df),
            'wavg_pd': round(wavg_pd, 4),
            'wavg_lgd': round(wavg_lgd, 4),
            'wavg_el_rate': round(wavg_el, 4),
            'prime_share': round(prime_share, 4),
            'near_prime_share': round(near_prime_share, 4),
            'subprime_share': round(subprime_share, 4),
            'high_risk_purpose_share': round(high_risk_share, 4),
            'fico_below_660_pct': round(fico_below_660, 4),
            'high_dti_pct': round(high_dti, 4),
            'status_distribution': status_df.to_dict('records')
        }

        print("")
        print("   Portfolio Quality Scorecard:")
        print("   " + "=" * 50)
        print("   {:30} ${:>15,.0f}".format("Total Exposure:", total_ead))
        print("   {:30} {:>14.2%}".format("Weighted Avg PD:", wavg_pd))
        print("   {:30} {:>14.2%}".format("Weighted Avg LGD:", wavg_lgd))
        print("   {:30} {:>14.2%}".format("Weighted Avg EL Rate:", wavg_el))
        print("   " + "-" * 50)
        print("   {:30} {:>14.2%}".format("Prime (A-B) Share:", prime_share))
        print("   {:30} {:>14.2%}".format("Near-Prime (C-D) Share:", near_prime_share))
        print("   {:30} {:>14.2%}".format("Subprime (E-G) Share:", subprime_share))
        print("   {:30} {:>14.2%}".format("High-Risk Purpose Share:", high_risk_share))
        print("   " + "-" * 50)
        print("   {:30} {:>14.2%}".format("FICO < 660 Share:", fico_below_660))
        print("   {:30} {:>14.2%}".format("DTI > 25 Share:", high_dti))
        print("   " + "=" * 50)

        print("\n   Loan Status Distribution:")
        print("   {:<50} {:>10} {:>8}".format("Status", "Count", "Share"))
        print("   " + "-" * 70)
        for _, row in status_df.head(10).iterrows():
            print("   {:<50} {:>10,} {:>7.2%}".format(
                row['loan_status'], row['cnt'], row['pct']))

        self.quality_indicators = indicators
        return indicators

    # ----------------------------------------------------------
    # Export
    # ----------------------------------------------------------

    def export_results(self):
        """Export all computed metrics to CSV."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if hasattr(self, 'stress_results'):
            self.stress_results.to_csv(STRESS_TEST_PATH, index=False)
            print("   -> Stress test results saved to: {}".format(STRESS_TEST_PATH))

        if hasattr(self, 'migration_matrix'):
            self.migration_matrix.to_csv(MIGRATION_PATH)
            print("   -> Migration matrix saved to: {}".format(MIGRATION_PATH))

        if hasattr(self, 'concentration'):
            self.concentration.to_csv(CONCENTRATION_PATH, index=False)
            print("   -> Concentration risk saved to: {}".format(CONCENTRATION_PATH))

    # ----------------------------------------------------------
    # Full Pipeline
    # ----------------------------------------------------------

    def run_all(self):
        """Execute all risk metric analyses."""
        print("=" * 60)
        print("RISK METRICS & STRESS TESTING")
        print("=" * 60)

        self.compute_concentration_risk()
        self.compute_risk_adjusted_returns()
        self.run_stress_tests()
        self.compute_migration_matrix()
        self.compute_portfolio_quality()

        print("\nExporting results...")
        self.export_results()

        print("\n" + "=" * 60)
        print("RISK METRICS COMPLETE")
        print("=" * 60)


# ==================================================================
# Standalone Execution
# ==================================================================
if __name__ == "__main__":
    try:
        rm = RiskMetrics()
        rm.run_all()
    except Exception as e:
        print("ERROR: {}".format(e))
        import traceback
        traceback.print_exc()