"""
Vasicek (ASRF) Model - Portfolio Credit Risk

Implements the Asymptotic Single Risk Factor (ASRF) model,
the foundation of Basel II/III capital requirements.

Core formula:
    Conditional PD:  P(D|S) = Phi( (Phi_inv(PD) + S*sqrt(rho)) / sqrt(1-rho) )

Where:
    Phi      = Standard normal CDF
    Phi_inv  = Standard normal inverse CDF (quantile function)
    S        = Systematic risk factor (macroeconomic state)
    rho      = Asset correlation
    PD       = Unconditional probability of default

Outputs:
    - Portfolio loss distribution (via Monte Carlo)
    - VaR at 99.9% confidence level
    - Expected Loss (EL)
    - Unexpected Loss (UL = VaR - EL)
    - Economic capital requirement

Monte Carlo uses grade-grouped Binomial approximation (7 groups instead
of 2.26M individual loans), reducing inner-loop cost from O(n_loans) to
O(n_grades) per scenario.

References:
    - Vasicek, O. (2002). "The distribution of loan portfolio value." Risk.
    - Gordy, M. (2003). "A risk-factor model foundation for ratings-based
      bank capital rules." Journal of Financial Intermediation.
"""

import numpy as np
import pandas as pd
import json
import sys
import os

from scipy.stats import norm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.database.connection import get_engine
import config


# ==================================================================
# Configuration
# ==================================================================

# Monte Carlo simulation settings
N_SIMULATIONS = 100_000       # Number of systematic scenarios
SEED = 42                     # Reproducibility

# Regulatory confidence level (Basel II/III)
CONFIDENCE_LEVEL = 0.999      # 99.9% VaR

# Asset correlation by risk segment (Basel II corporate exposure)
# Can be fine-tuned or loaded from external calibration
ASSET_CORRELATION = {
    'A-B': 0.15,      # Prime: lower correlation, more idiosyncratic risk
    'C-D': 0.20,      # Near-Prime
    'E-G': 0.25,      # Subprime: higher systemic sensitivity
    'default': 0.20   # Fallback
}

# Output paths
OUTPUT_DIR = config.BASE_DIR / 'output'
DISTRIBUTION_PATH = OUTPUT_DIR / 'loss_distribution.csv'
SUMMARY_PATH = OUTPUT_DIR / 'vasicek_summary.csv'

# Dashboard export paths
MODEL_DIR = config.BASE_DIR / 'output' / 'models'
DASHBOARD_PATH = MODEL_DIR / 'vasicek_results.json'


class VasicekModel:
    """
    Vasicek Asymptotic Single Risk Factor model.

    Computes portfolio credit risk metrics using both
    analytical formulas and Monte Carlo simulation.
    """

    def __init__(self, n_simulations=N_SIMULATIONS, seed=SEED):
        self.n_simulations = n_simulations
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Results storage
        self.loss_distribution = None
        self.summary = None
        self._portfolio_df = None   # Stored for reuse in export

    # ----------------------------------------------------------
    # Core Mathematical Functions
    # ----------------------------------------------------------

    @staticmethod
    def conditional_pd(pd_unconditional, rho, s):
        """
        Compute conditional default probability given systematic factor.

        P(D|S) = Phi( (Phi_inv(PD) + S * sqrt(rho)) / sqrt(1 - rho) )

        Args:
            pd_unconditional: Unconditional PD (scalar or array).
            rho: Asset correlation (scalar).
            s: Systematic risk factor (scalar or array).

        Returns:
            Conditional PD values, clipped to [0, 1].
        """
        # Inverse normal of unconditional PD
        # Clip PD to avoid inf at boundaries
        pd_safe = np.clip(pd_unconditional, 1e-10, 1 - 1e-10)
        inv_pd = norm.ppf(pd_safe)

        sqrt_rho = np.sqrt(rho)
        sqrt_one_minus_rho = np.sqrt(1 - rho)

        # Conditional default threshold
        cond_default_threshold = (inv_pd - s * sqrt_rho) / sqrt_one_minus_rho

        # Conditional PD = probability that idiosyncratic factor Z < threshold
        cond_pd = norm.cdf(cond_default_threshold)
        return np.clip(cond_pd, 0.0, 1.0)

    @staticmethod
    def analytical_vasicek_loss_percentile(pd, lgd, rho, alpha=0.999):
        """
        Analytical (closed-form) Vasicek loss quantile.

        Loss quantile at confidence level alpha:
            L_alpha = LGD * Phi( (Phi_inv(PD) + sqrt(rho) * Phi_inv(alpha)) / sqrt(1 - rho) )

        Args:
            pd: Unconditional probability of default.
            lgd: Loss given default.
            rho: Asset correlation.
            alpha: Confidence level (e.g., 0.999 for 99.9%).

        Returns:
            float: Loss rate at the given confidence level.
        """
        pd_safe = max(min(pd, 1 - 1e-10), 1e-10)
        inv_pd = norm.ppf(pd_safe)
        inv_alpha = norm.ppf(alpha)

        sqrt_rho = np.sqrt(rho)
        sqrt_one_minus_rho = np.sqrt(1 - rho)

        loss_rate = lgd * norm.cdf(
            (inv_pd + sqrt_rho * inv_alpha) / sqrt_one_minus_rho
        )
        return loss_rate

    # ----------------------------------------------------------
    # Data Loading
    # ----------------------------------------------------------

    def load_portfolio_data(self):
        """
        Load loan-level data with PD, LGD, and EAD from the database.
        Uses actual historical PD and LGD from the SQL analytics.

        Returns:
            pd.DataFrame: Portfolio data with pd, lgd, ead, grade columns.
        """
        print("Loading portfolio data from database...")

        query = """
            SELECT
                id,
                grade,
                funded_amnt AS ead,
                loan_amnt,
                int_rate,
                CASE WHEN grade IN ('A', 'B') THEN 'A-B'
                     WHEN grade IN ('C', 'D') THEN 'C-D'
                     WHEN grade IN ('E', 'F', 'G') THEN 'E-G'
                END AS risk_segment
            FROM loans_master
        """

        df = pd.read_sql(query, get_engine())
        print("   -> Loaded {:,} loans.".format(len(df)))

        # Map asset correlation per segment
        df['rho'] = df['risk_segment'].map(ASSET_CORRELATION).fillna(ASSET_CORRELATION['default'])

        # Use historical PD per grade (from SQL analytics)
        print("   -> Attaching historical PD by grade...")
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
                    THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS historical_pd
            FROM mature_loans
            GROUP BY grade
        """
        pd_df = pd.read_sql(pd_query, get_engine())
        pd_lookup = pd_df.set_index('grade')['historical_pd'].to_dict()

        df['pd'] = df['grade'].map(pd_lookup).fillna(0.20)
        print("   -> PD range: {:.4f} (Grade A) to {:.4f} (Grade G)".format(
            df['pd'].min(), df['pd'].max()))

        # Use historical LGD per grade (from SQL analytics)
        print("   -> Attaching historical LGD by grade...")
        lgd_query = """
            SELECT
                grade,
                AVG(1 - (total_pymnt / funded_amnt)) AS historical_lgd
            FROM loans_master
            WHERE loan_status IN ('Charged Off', 'Default',
                'Does not meet the credit policy. Status:Charged Off')
            AND funded_amnt > 0
            GROUP BY grade
        """
        lgd_df = pd.read_sql(lgd_query, get_engine())
        lgd_lookup = lgd_df.set_index('grade')['historical_lgd'].to_dict()

        df['lgd'] = df['grade'].map(lgd_lookup).fillna(0.46)
        print("   -> LGD range: {:.4f} to {:.4f}".format(
            df['lgd'].min(), df['lgd'].max()))

        # Store as instance variable so other methods can reference it
        self._portfolio_df = df

        return df

    # ----------------------------------------------------------
    # Monte Carlo Simulation
    # ----------------------------------------------------------

    def simulate_loss_distribution(self, df):
        """
        Monte Carlo simulation using grouped Binomial approximation.

        Instead of drawing Bernoulli for each of 2.26M loans per scenario,
        we group loans by grade (7 groups) and use Binomial distribution.
        This reduces the inner loop from 2.26M to 7 per scenario.

        Args:
            df: DataFrame with pd, lgd, ead, rho columns.

        Returns:
            np.ndarray: Array of simulated portfolio losses.
        """
        print("\nRunning Monte Carlo simulation ({:,} scenarios)...".format(
            self.n_simulations))

        total_exposure = df['ead'].sum()
        print("   -> Portfolio: {:,} loans, ${:,.0f} total exposure".format(
            len(df), total_exposure))

        # Group loans by grade to get unique (pd, lgd, rho) combinations
        groups = df.groupby('grade').agg(
            pd=('pd', 'first'),
            lgd=('lgd', 'first'),
            rho=('rho', 'first'),
            n_loans=('ead', 'count'),
            total_ead=('ead', 'sum')
        ).reset_index()

        n_groups = len(groups)
        print("   -> Aggregated to {} grade groups".format(n_groups))

        # Pre-compute per-group constants
        pd_safe = np.clip(groups['pd'].values, 1e-10, 1 - 1e-10)
        inv_pd = norm.ppf(pd_safe)
        sqrt_rho = np.sqrt(groups['rho'].values)
        sqrt_one_minus_rho = np.sqrt(1 - groups['rho'].values)
        lgd_values = groups['lgd'].values
        n_loans_arr = groups['n_loans'].values.astype(np.int64)
        ead_per_loan = groups['total_ead'].values / n_loans_arr  # avg EAD per loan in group

        losses = np.empty(self.n_simulations, dtype=np.float64)
        log_interval = max(self.n_simulations // 10, 1)

        for i in range(self.n_simulations):
            # Draw systematic factor
            s = self.rng.standard_normal()

            # Conditional PD per group (7 values only)
            threshold = (inv_pd - s * sqrt_rho) / sqrt_one_minus_rho
            cond_pd = norm.cdf(threshold)

            # Binomial draws: how many defaults in each group
            n_defaults = self.rng.binomial(n_loans_arr, cond_pd)

            # Portfolio loss = sum(defaults_k * avg_EAD_k * LGD_k)
            losses[i] = np.sum(n_defaults * ead_per_loan * lgd_values)

            if (i + 1) % log_interval == 0:
                print("   -> Progress: {:,} / {:,} scenarios".format(
                    i + 1, self.n_simulations))

        self.loss_distribution = losses
        print("   -> Simulation complete.")
        return losses

    # ----------------------------------------------------------
    # Risk Metrics
    # ----------------------------------------------------------

    def compute_risk_metrics(self, losses, df):
        """
        Compute key risk metrics from the simulated loss distribution.

        Metrics:
            - EL (Expected Loss): Mean of loss distribution
            - VaR (Value at Risk): Quantile at 99.9% confidence
            - ES (Expected Shortfall / CVaR): Mean loss beyond VaR
            - UL (Unexpected Loss): VaR - EL
            - Economic Capital: VaR @ 99.9%
            - Loss rates as percentage of total exposure

        Args:
            losses: Simulated loss array.
            df: Portfolio DataFrame with ead.

        Returns:
            pd.DataFrame: Summary of risk metrics.
        """
        print("\nComputing risk metrics...")

        total_exposure = df['ead'].sum()

        # Core metrics
        el = np.mean(losses)
        el_rate = el / total_exposure

        var_999 = np.percentile(losses, CONFIDENCE_LEVEL * 100)
        var_999_rate = var_999 / total_exposure

        # Expected Shortfall (CVaR) - average of losses beyond VaR
        tail_losses = losses[losses >= var_999]
        es_999 = np.mean(tail_losses) if len(tail_losses) > 0 else var_999
        es_999_rate = es_999 / total_exposure

        # Unexpected Loss
        ul = var_999 - el
        ul_rate = ul / total_exposure

        # Additional percentiles for distribution shape
        var_990 = np.percentile(losses, 99.0)
        var_995 = np.percentile(losses, 99.5)
        var_950 = np.percentile(losses, 95.0)

        # Standard deviation of losses
        loss_std = np.std(losses)

        # Analytical comparison (using portfolio-weighted average PD/LGD/rho)
        avg_pd = np.average(df['pd'], weights=df['ead'])
        avg_lgd = np.average(df['lgd'], weights=df['ead'])
        avg_rho = np.average(df['rho'], weights=df['ead'])

        analytical_var_rate = self.analytical_vasicek_loss_percentile(
            avg_pd, avg_lgd, avg_rho, alpha=CONFIDENCE_LEVEL
        )
        analytical_var = analytical_var_rate * total_exposure

        self.summary = {
            'total_exposure': float(total_exposure),
            'total_loans': len(df),
            'avg_pd': float(avg_pd),
            'avg_lgd': float(avg_lgd),
            'avg_rho': float(avg_rho),
            'el': float(el),
            'el_rate': float(el_rate),
            'var_99.9%': float(var_999),
            'var_99.9%_rate': float(var_999_rate),
            'var_99.5%': float(var_995),
            'var_99.5%_rate': float(var_995 / total_exposure),
            'var_99.0%': float(var_990),
            'var_99.0%_rate': float(var_990 / total_exposure),
            'var_95.0%': float(var_950),
            'var_95.0%_rate': float(var_950 / total_exposure),
            'expected_shortfall_99.9%': float(es_999),
            'expected_shortfall_99.9%_rate': float(es_999_rate),
            'unexpected_loss': float(ul),
            'ul_rate': float(ul_rate),
            'economic_capital': float(var_999),
            'economic_capital_rate': float(var_999_rate),
            'loss_std': float(loss_std),
            'loss_std_rate': float(loss_std / total_exposure),
            'analytical_var_99.9%': float(analytical_var),
            'analytical_var_99.9%_rate': float(analytical_var_rate),
            'n_simulations': self.n_simulations
        }

        # Print summary
        print("\n   {}".format("=" * 55))
        print("   VASICEK MODEL RESULTS")
        print("   {}".format("=" * 55))
        print("   Portfolio Exposure:      ${:>15,.0f}".format(total_exposure))
        print("   Number of Loans:         {:>15,}".format(len(df)))
        print("   Avg PD (exposure-wtd):   {:>15.4%}".format(avg_pd))
        print("   Avg LGD (exposure-wtd):  {:>15.4%}".format(avg_lgd))
        print("   Avg Asset Correlation:   {:>15.4f}".format(avg_rho))
        print("   {}".format("-" * 55))
        print("   Expected Loss (EL):      ${:>15,.0f}  ({:.4%})".format(el, el_rate))
        print("   VaR @ 99.9%:             ${:>15,.0f}  ({:.4%})".format(var_999, var_999_rate))
        print("   VaR @ 99.5%:             ${:>15,.0f}  ({:.4%})".format(var_995, var_995 / total_exposure))
        print("   VaR @ 99.0%:             ${:>15,.0f}  ({:.4%})".format(var_990, var_990 / total_exposure))
        print("   VaR @ 95.0%:             ${:>15,.0f}  ({:.4%})".format(var_950, var_950 / total_exposure))
        print("   {}".format("-" * 55))
        print("   Unexpected Loss (UL):    ${:>15,.0f}  ({:.4%})".format(ul, ul_rate))
        print("   Exp. Shortfall @99.9%:   ${:>15,.0f}  ({:.4%})".format(es_999, es_999_rate))
        print("   Loss Std Dev:            ${:>15,.0f}  ({:.4%})".format(loss_std, loss_std / total_exposure))
        print("   {}".format("-" * 55))
        print("   Economic Capital:        ${:>15,.0f}  ({:.4%})".format(var_999, var_999_rate))
        print("   Analytical VaR @99.9%:   ${:>15,.0f}  ({:.4%})".format(analytical_var, analytical_var_rate))
        print("   {}".format("=" * 55))

        return self.summary

    # ----------------------------------------------------------
    # Per-Segment Analysis
    # ----------------------------------------------------------

    def compute_segment_metrics(self, df):
        """
        Compute Vasicek metrics per risk segment (A-B, C-D, E-G).

        Uses analytical Vasicek formula for each segment.

        Args:
            df: Portfolio DataFrame.

        Returns:
            pd.DataFrame: Per-segment risk metrics.
        """
        print("\nComputing per-segment Vasicek metrics...")

        results = []
        for segment, group in df.groupby('risk_segment'):
            rho = ASSET_CORRELATION.get(segment, ASSET_CORRELATION['default'])
            pd_seg = np.average(group['pd'], weights=group['ead'])
            lgd_seg = np.average(group['lgd'], weights=group['ead'])
            ead_seg = group['ead'].sum()
            n_loans = len(group)

            # Expected loss
            el = pd_seg * lgd_seg * ead_seg

            # Analytical VaR @ 99.9%
            var_rate = self.analytical_vasicek_loss_percentile(pd_seg, lgd_seg, rho)
            var = var_rate * ead_seg

            ul = var - el

            results.append({
                'risk_segment': segment,
                'n_loans': n_loans,
                'total_exposure': ead_seg,
                'avg_pd': pd_seg,
                'avg_lgd': lgd_seg,
                'asset_correlation': rho,
                'expected_loss': el,
                'el_rate': pd_seg * lgd_seg,
                'var_99.9%': var,
                'var_99.9%_rate': var_rate,
                'unexpected_loss': ul,
                'ul_rate': var_rate - pd_seg * lgd_seg
            })

        segment_df = pd.DataFrame(results).sort_values('risk_segment')

        print("\n   {:<12} {:>10} {:>16} {:>8} {:>8} {:>6} {:>14} {:>14} {:>14}".format(
            "Segment", "Loans", "Exposure", "PD", "LGD", "rho", "EL", "VaR 99.9%", "UL"))
        print("   {}".format("-" * 105))
        for _, row in segment_df.iterrows():
            print("   {:<12} {:>10,} ${:>14,.0f} {:>7.4f} {:>7.4f} {:>6.2f} "
                  "${:>12,.0f} ${:>12,.0f} ${:>12,.0f}".format(
                row['risk_segment'], row['n_loans'], row['total_exposure'],
                row['avg_pd'], row['avg_lgd'], row['asset_correlation'],
                row['expected_loss'], row['var_99.9%'], row['unexpected_loss']))

        return segment_df

    # ----------------------------------------------------------
    # Export
    # ----------------------------------------------------------

    def export_results(self, losses, df):
        """
        Save loss distribution and summary to CSV.

        Args:
            losses: Simulated loss array.
            df: Portfolio DataFrame (used to compute total exposure).
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        total_exposure = df['ead'].sum()

        # Loss distribution with percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
        loss_percentiles = np.percentile(losses, percentiles)

        dist_df = pd.DataFrame({
            'percentile': percentiles,
            'loss_amount': loss_percentiles,
            'loss_rate': loss_percentiles / total_exposure
        })

        dist_df.to_csv(DISTRIBUTION_PATH, index=False)
        print("\n   Loss distribution saved to: {}".format(DISTRIBUTION_PATH))

        # Summary
        if self.summary:
            sum_df = pd.DataFrame([self.summary])
            sum_df.to_csv(SUMMARY_PATH, index=False)
            print("   Vasicek summary saved to: {}".format(SUMMARY_PATH))

    def _export_dashboard(self, df, segment_df):
        """
        Export results for the Streamlit visualization dashboard.

        Files produced:
            output/models/vasicek_results.json
        """
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Build per-grade result list
        grade_results = []
        for grade in sorted(df['grade'].unique()):
            grade_mask = df['grade'] == grade
            grade_results.append({
                "grade": str(grade),
                "pd": float(df.loc[grade_mask, 'pd'].iloc[0]),
                "ead": float(df.loc[grade_mask, 'ead'].sum()),
                "lgd": float(df.loc[grade_mask, 'lgd'].iloc[0]),
                "correlation": float(df.loc[grade_mask, 'rho'].iloc[0]),
                "n_loans": int(grade_mask.sum()),
            })

        # Include a sample of the loss distribution (every 10th scenario)
        if self.loss_distribution is not None:
            loss_samples = self.loss_distribution[::10].tolist()
        else:
            loss_samples = []

        # Map summary keys to the names expected by the dashboard
        s = self.summary if self.summary else {}

        results_json = {
            "expected_loss": float(s.get("el", 0)),
            "var_99": float(s.get("var_99.0%", 0)),
            "var_99.9": float(s.get("var_99.9%", 0)),
            "economic_capital": float(s.get("economic_capital", 0)),
            "n_simulations": int(s.get("n_simulations", self.n_simulations)),
            "total_exposure": float(s.get("total_exposure", 0)),
            "total_loans": int(s.get("total_loans", 0)),
            "avg_pd": float(s.get("avg_pd", 0)),
            "avg_lgd": float(s.get("avg_lgd", 0)),
            "unexpected_loss": float(s.get("unexpected_loss", 0)),
            "expected_shortfall_99.9": float(s.get("expected_shortfall_99.9%", 0)),
            "loss_std": float(s.get("loss_std", 0)),
            "loss_samples": loss_samples,
            "grade_results": grade_results,
            "segment_results": segment_df.to_dict(orient="records") if segment_df is not None else [],
        }

        with open(DASHBOARD_PATH, "w") as f:
            json.dump(results_json, f, indent=2)
        print("\n   Dashboard results saved to: {}".format(DASHBOARD_PATH))

    # ----------------------------------------------------------
    # Full Pipeline
    # ----------------------------------------------------------

    def run(self):
        """
        Execute the full Vasicek pipeline:
        1. Load portfolio data with PD, LGD, EAD
        2. Run Monte Carlo simulation
        3. Compute risk metrics (EL, VaR, UL, capital)
        4. Compute per-segment analysis
        5. Export results (CSV + dashboard JSON)
        """
        print("=" * 60)
        print("VASICEK ASRF MODEL")
        print("=" * 60)

        # Step 1: Load data
        df = self.load_portfolio_data()

        # Step 2: Monte Carlo
        losses = self.simulate_loss_distribution(df)

        # Step 3: Risk metrics
        self.compute_risk_metrics(losses, df)

        # Step 4: Per-segment
        segment_metrics = self.compute_segment_metrics(df)

        # Step 5: Export CSV
        self.export_results(losses, df)

        # Step 6: Export dashboard JSON
        self._export_dashboard(df, segment_metrics)

        print("\n" + "=" * 60)
        print("VASICEK MODEL COMPLETE")
        print("=" * 60)

        return self.summary, segment_metrics


# ==================================================================
# Standalone Execution
# ==================================================================
if __name__ == "__main__":
    try:
        model = VasicekModel()
        summary, segments = model.run()
    except Exception as e:
        print("ERROR: {}".format(e))
        import traceback
        traceback.print_exc()