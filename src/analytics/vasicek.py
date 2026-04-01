"""
Vasicek (ASRF) Model — Portfolio Credit Risk

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

References:
    - Vasicek, O. (2002). "The distribution of loan portfolio value." Risk.
    - Gordy, M. (2003). "A risk-factor model foundation for ratings-based
      bank capital rules." Journal of Financial Intermediation.
"""

import numpy as np
import pandas as pd
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
        print(f"   -> Loaded {len(df):,} loans.")

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
        print(f"   -> PD range: {df['pd'].min():.4f} (Grade A) to {df['pd'].max():.4f} (Grade G)")

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
        print(f"   -> LGD range: {df['lgd'].min():.4f} to {df['lgd'].max():.4f}")

        return df

    # ----------------------------------------------------------
    # Monte Carlo Simulation
    # ----------------------------------------------------------

    def simulate_loss_distribution(self, df):
        """
        Run Monte Carlo simulation of the portfolio loss distribution.

        For each simulation:
            1. Draw systematic factor S ~ N(0,1)
            2. Compute conditional PD for each loan: P(D|S)
            3. Draw default events: Bernoulli(P(D|S)) for each loan
            4. Compute portfolio loss: sum(LGD_i * EAD_i * default_i)

        Args:
            df: DataFrame with pd, lgd, ead, rho columns.

        Returns:
            np.ndarray: Array of simulated portfolio losses.
        """
        print(f"\nRunning Monte Carlo simulation ({self.n_simulations:,} scenarios)...")

        # Extract arrays for vectorized computation
        pd_arr = df['pd'].values
        lgd_arr = df['lgd'].values
        ead_arr = df['ead'].values
        rho_arr = df['rho'].values
        n_loans = len(df)

        total_exposure = ead_arr.sum()
        print(f"   -> Portfolio: {n_loans:,} loans, ${total_exposure:,.0f} total exposure")

        # Draw all systematic factors at once
        s_factors = self.rng.standard_normal(self.n_simulations)

        # Pre-compute portfolio-level EL (constant across simulations)
        portfolio_el = np.sum(pd_arr * lgd_arr * ead_arr)

        # Storage for losses
        losses = np.empty(self.n_simulations)

        # --- Simulation loop ---
        # Process in batches to manage memory for 2.26M loans
        batch_size = 50_000
        n_batches = int(np.ceil(self.n_simulations / batch_size))

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, self.n_simulations)
            s_batch = s_factors[start:end]
            batch_n = end - start

            # Conditional PD for each (loan, scenario) pair
            # Shape: (batch_n, n_loans)
            pd_safe = np.clip(pd_arr, 1e-10, 1 - 1e-10)
            inv_pd = norm.ppf(pd_safe)

            sqrt_rho = np.sqrt(rho_arr)
            sqrt_one_minus_rho = np.sqrt(1 - rho_arr)

            # S is (batch_n, 1), inv_pd is (n_loans,)
            # threshold shape: (batch_n, n_loans)
            threshold = (inv_pd[np.newaxis, :] - s_batch[:, np.newaxis] * sqrt_rho[np.newaxis, :]) / \
                        sqrt_one_minus_rho[np.newaxis, :]

            cond_pd = norm.cdf(threshold)

            # Draw defaults: Bernoulli(cond_pd)
            defaults = self.rng.uniform(size=(batch_n, n_loans)) < cond_pd

            # Loss for each loan: LGD * EAD * default_indicator
            # Sum across loans for each scenario
            loan_losses = lgd_arr[np.newaxis, :] * ead_arr[np.newaxis, :] * defaults
            losses[start:end] = loan_losses.sum(axis=1)

            if (batch_idx + 1) % 2 == 0 or batch_idx == n_batches - 1:
                print(f"   -> Progress: {end:,} / {self.n_simulations:,} scenarios")

        self.loss_distribution = losses

        print(f"   -> Simulation complete.")

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
            'total_exposure': total_exposure,
            'total_loans': len(df),
            'avg_pd': avg_pd,
            'avg_lgd': avg_lgd,
            'avg_rho': avg_rho,
            'el': el,
            'el_rate': el_rate,
            'var_99.9%': var_999,
            'var_99.9%_rate': var_999_rate,
            'var_99.5%': var_995,
            'var_99.5%_rate': var_995 / total_exposure,
            'var_99.0%': var_990,
            'var_99.0%_rate': var_990 / total_exposure,
            'var_95.0%': var_950,
            'var_95.0%_rate': var_950 / total_exposure,
            'expected_shortfall_99.9%': es_999,
            'expected_shortfall_99.9%_rate': es_999_rate,
            'unexpected_loss': ul,
            'ul_rate': ul_rate,
            'economic_capital': var_999,
            'economic_capital_rate': var_999_rate,
            'loss_std': loss_std,
            'loss_std_rate': loss_std / total_exposure,
            'analytical_var_99.9%': analytical_var,
            'analytical_var_99.9%_rate': analytical_var_rate,
            'n_simulations': self.n_simulations
        }

        # Print summary
        print(f"\n   {'=' * 55}")
        print(f"   VASICEK MODEL RESULTS")
        print(f"   {'=' * 55}")
        print(f"   Portfolio Exposure:      ${total_exposure:>15,.0f}")
        print(f"   Number of Loans:         {len(df):>15,}")
        print(f"   Avg PD (exposure-wtd):   {avg_pd:>15.4%}")
        print(f"   Avg LGD (exposure-wtd):  {avg_lgd:>15.4%}")
        print(f"   Avg Asset Correlation:   {avg_rho:>15.4f}")
        print(f"   {'-' * 55}")
        print(f"   Expected Loss (EL):      ${el:>15,.0f}  ({el_rate:.4%})")
        print(f"   VaR @ 99.9%:             ${var_999:>15,.0f}  ({var_999_rate:.4%})")
        print(f"   VaR @ 99.5%:             ${var_995:>15,.0f}  ({var_995 / total_exposure:.4%})")
        print(f"   VaR @ 99.0%:             ${var_990:>15,.0f}  ({var_990 / total_exposure:.4%})")
        print(f"   VaR @ 95.0%:             ${var_950:>15,.0f}  ({var_950 / total_exposure:.4%})")
        print(f"   {'-' * 55}")
        print(f"   Unexpected Loss (UL):    ${ul:>15,.0f}  ({ul_rate:.4%})")
        print(f"   Exp. Shortfall @99.9%:   ${es_999:>15,.0f}  ({es_999_rate:.4%})")
        print(f"   Loss Std Dev:            ${loss_std:>15,.0f}  ({loss_std / total_exposure:.4%})")
        print(f"   {'-' * 55}")
        print(f"   Economic Capital:        ${var_999:>15,.0f}  ({var_999_rate:.4%})")
        print(f"   Analytical VaR @99.9%:   ${analytical_var:>15,.0f}  ({analytical_var_rate:.4%})")
        print(f"   {'=' * 55}")

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

        print(f"\n   {'Segment':<12} {'Loans':>10} {'Exposure':>16} {'PD':>8} {'LGD':>8} {'rho':>6} {'EL':>14} {'VaR 99.9%':>14} {'UL':>14}")
        print(f"   {'-' * 105}")
        for _, row in segment_df.iterrows():
            print(f"   {row['risk_segment']:<12} {row['n_loans']:>10,} ${row['total_exposure']:>14,.0f} "
                  f"{row['avg_pd']:>7.4f} {row['avg_lgd']:>7.4f} {row['asset_correlation']:>6.2f} "
                  f"${row['expected_loss']:>12,.0f} ${row['var_99.9%']:>12,.0f} ${row['unexpected_loss']:>12,.0f}")

        return segment_df

    # ----------------------------------------------------------
    # Export
    # ----------------------------------------------------------

    def export_results(self, losses):
        """
        Save loss distribution and summary to CSV.

        Args:
            losses: Simulated loss array.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Loss distribution with percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
        loss_percentiles = np.percentile(losses, percentiles)

        dist_df = pd.DataFrame({
            'percentile': percentiles,
            'loss_amount': loss_percentiles,
            'loss_rate': loss_percentiles / df['ead'].sum() if self.summary else loss_percentiles
        })

        dist_df.to_csv(DISTRIBUTION_PATH, index=False)
        print(f"\n   Loss distribution saved to: {DISTRIBUTION_PATH}")

        # Summary
        if self.summary:
            sum_df = pd.DataFrame([self.summary])
            sum_df.to_csv(SUMMARY_PATH, index=False)
            print(f"   Vasicek summary saved to: {SUMMARY_PATH}")

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
        5. Export results
        """
        global df
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

        # Step 5: Export
        self.export_results(losses)

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
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

