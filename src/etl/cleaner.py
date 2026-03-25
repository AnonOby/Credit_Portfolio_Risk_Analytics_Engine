import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class DataCleaner:
    """
    Handles data cleaning logic for credit portfolio data.
    Focuses on type conversion, text normalization, and missing value handling.
    """

    def __init__(self):
        # Define a mapping dictionary to normalize common job titles
        # This reduces cardinality of the 'emp_title' feature significantly
        self.job_title_mapping = {
            'manager': 'Manager',
            'driver': 'Driver',
            'teacher': 'Teacher',
            'owner': 'Owner',
            'nurse': 'Nurse',
            'rn': 'Nurse',
            'engineer': 'Engineer',
            'analyst': 'Analyst',
            'director': 'Director',
            'supervisor': 'Supervisor',
            'admin': 'Administrator',
            'assistant': 'Assistant',
            'sales': 'Sales',
            'accountant': 'Accountant',
            'police': 'Police Officer',
            'developer': 'Developer',
            'programmer': 'Developer'
        }

    def execute_pipeline(self, df):
        """
        Apply the full cleaning pipeline to a DataFrame chunk.
        """
        # 1. Clean Percentage Columns
        df = self._clean_percentage_cols(df)

        # 2. Clean Employment Length
        df = self._clean_emp_length(df)

        # 3. Normalize Job Titles
        df = self._normalize_emp_titles(df)

        # 4. Convert Dates and Calculate History
        df = self._process_dates(df)

        # 5. Handle Missing Values
        df = self._handle_missing_data(df)

        # 6. Select and Rename Key Columns (Reducing dimensionality for efficiency)
        # We keep only the most relevant columns for risk modeling
        df = self._select_relevant_columns(df)

        return df

    def _clean_percentage_cols(self, df):
        """
        Remove '%' signs and convert to float.
        Affects columns: 'int_rate', 'revol_util'
        """
        percent_cols = ['int_rate', 'revol_util']
        for col in percent_cols:
            if col in df.columns:
                # Remove non-numeric chars (except .) and convert
                df[col] = df[col].astype(str).str.replace('%', '', regex=False).replace('nan', np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _clean_emp_length(self, df):
        """
        Convert employment length from string (e.g., '10+ years', '< 1 year') to float.
        Mapping:
        - '10+ years' -> 10.0
        - '< 1 year'  -> 0.0
        - 'n/a'       -> NaN
        """
        if 'emp_length' not in df.columns:
            return df

        # Extract numeric part using regex, handle special cases first
        df['emp_length'] = df['emp_length'].astype(str)

        # Handle specific strings
        df['emp_length'] = df['emp_length'].str.replace('< 1 year', '0', regex=False)
        df['emp_length'] = df['emp_length'].str.replace('10+ years', '10', regex=False)

        # Extract numbers (e.g., "3 years" -> "3")
        df['emp_length'] = df['emp_length'].str.extract(r'(\d+)')[0]

        # Convert to float
        df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce')

        return df

    def _normalize_emp_titles(self, df):
        """
        Reduce the high cardinality of 'emp_title' by mapping keywords.
        Titles not in the mapping are labeled 'Other'.
        """
        if 'emp_title' not in df.columns:
            return df

        # Fill missing with 'Unknown'
        df['emp_title'] = df['emp_title'].fillna('Unknown').astype(str)

        # Lowercase for matching
        df['emp_title_clean'] = df['emp_title'].str.lower()

        # Apply mapping based on keywords
        # This vectorized approach is faster than iterating rows
        for keyword, standard_title in self.job_title_mapping.items():
            # Use regex to match whole word for better accuracy
            mask = df['emp_title_clean'].str.contains(rf'\b{keyword}\b', regex=True, na=False)
            df.loc[mask, 'emp_title'] = standard_title

        # Drop helper column
        df.drop('emp_title_clean', axis=1, inplace=True)

        return df

    def _process_dates(self, df):
        """
        Convert 'issue_d' and 'earliest_cr_line' to datetime objects.
        Calculate 'credit_history_months' as a derived feature.
        """
        date_cols = ['issue_d', 'earliest_cr_line']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%b-%Y', errors='coerce')

        # Calculate Credit History Length in Months
        if 'issue_d' in df.columns and 'earliest_cr_line' in df.columns:
            # Calculate difference in months (approximate)
            df['credit_history_months'] = (
                    (df['issue_d'].dt.year - df['earliest_cr_line'].dt.year) * 12 +
                    (df['issue_d'].dt.month - df['earliest_cr_line'].dt.month)
            )
            # Handle negative values (data error) by setting to NaN
            df.loc[df['credit_history_months'] < 0, 'credit_history_months'] = np.nan

        return df

    def _handle_missing_data(self, df):
        """
        Strategy:
        - Numerical: Fill with Median (robust against outliers).
        - Categorical: Fill with 'Unknown'.
        - Drop columns with > 50% missing rate to save space.
        """
        # Drop columns with excessive missing data (>50%)
        missing_ratio = df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

        # Fill Numerical with Median
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                # FIX: Reassign instead of inplace to avoid ChainedAssignmentError
                df[col] = df[col].fillna(median_val)

        # Fill Categorical with 'Unknown'
        # FIX: Include 'string' to handle future Pandas versions and silence warning
        cat_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in cat_cols:
            if df[col].isnull().any():
                # FIX: Reassign instead of inplace
                df[col] = df[col].fillna('Unknown')

        return df

    def _select_relevant_columns(self, df):
        """
        Keep only essential columns for the risk model to reduce memory footprint.
        """
        # Core features for EL (PD/LGD/EAD) calculation
        essential_cols = [
            'id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
            'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'emp_title',
            'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
            'issue_d', 'loan_status', 'purpose', 'dti', 'delinq_2yrs',
            'earliest_cr_line', 'fico_range_low', 'fico_range_high',
            'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
            'total_acc', 'total_pymnt', 'total_rec_prncp', 'total_rec_int',
            'last_pymnt_d', 'last_pymnt_amnt', 'credit_history_months'
        ]

        # Filter: only keep columns that exist in the current chunk
        existing_cols = [col for col in essential_cols if col in df.columns]

        # Also keep zip_code for later joining with Census data
        if 'zip_code' in df.columns:
            existing_cols.append('zip_code')

        return df[existing_cols]


# --- Test Block ---
if __name__ == "__main__":
    print("--- Testing DataCleaner ---")

    # Create a dummy DataFrame for testing
    data = {
        'int_rate': ['12.5%', '10.0%', np.nan],
        'emp_length': ['10+ years', '< 1 year', '5 years'],
        'emp_title': ['Senior Software Engineer', 'Registered Nurse', 'Manager'],
        'issue_d': ['Dec-2018', 'Jan-2019', 'Feb-2019'],
        'earliest_cr_line': ['Jan-2000', 'Feb-2010', 'Mar-2015']
    }
    df_test = pd.DataFrame(data)

    cleaner = DataCleaner()
    df_cleaned = cleaner.execute_pipeline(df_test)

    print("\n🧪 Original Data:")
    print(df_test[['int_rate', 'emp_length', 'emp_title']].head())

    print("\n✨ Cleaned Data:")
    print(df_cleaned[['int_rate', 'emp_length', 'emp_title', 'credit_history_months']].head())

    print("\n✅ Data types:")
    print(df_cleaned.dtypes)