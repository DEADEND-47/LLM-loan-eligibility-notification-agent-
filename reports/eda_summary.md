# EDA & Feature Engineering Summary

- Merged dataset: 51,336 rows x 87 columns (inner join of Internal_Bank_Dataset and External_Cibil_Dataset on PROSPECTID).
- Target `Approved_Flag` distribution: {'P2': 32199, 'P3': 7452, 'P4': 5882, 'P1': 5803}.
- Dropped 6 columns for >30.0% missing (mostly utilization and delinquency-recency fields that don't apply to customers with no such history): ['CC_utilization', 'PL_utilization', 'time_since_recent_deliquency', 'max_delinquency_level', 'time_since_first_deliquency', 'max_unsec_exposure_inPct'].
- Dropped 13 further columns as redundant duplicates (|correlation| > 0.9 with another retained column): ['PL_enq_L12m', 'Secured_TL', 'Tot_Closed_TL', 'enq_L6m', 'num_dbt_12mts', 'num_deliq_6_12mts', 'num_lss_12mts', 'num_std_12mts', 'num_times_60p_dpd', 'pct_CC_enq_L6m_of_ever', 'pct_PL_enq_L6m_of_ever', 'pct_closed_tl', 'pct_of_active_TLs_ever'].
- Total columns removed: 19 (from 87 down to 72 pre-encoding feature columns).
- `NETMONTHLYINCOME` contains data-entry outliers (values as low as single digits and as high as 25 lakh/month); capped at the 1st/99th percentile as `NETMONTHLYINCOME_Capped` for use in business calculations.
- Engineered features: `Income_TL_Ratio`, `Credit_Health_Score` (model features), plus `Age_Group`, `Income_Segment`, `Credit_Score_Band` (reporting-only bins).
- All 5 categorical columns show a statistically significant association with `Approved_Flag` (chi-square p < 0.05).