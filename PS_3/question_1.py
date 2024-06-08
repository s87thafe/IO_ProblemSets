import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from config import PS3

# Load data
data_path = PS3.joinpath("data").resolve()
data = pd.read_csv(data_path/'GMdata.csv')

# Task 1: Sample Statistics

# Add missing observations to full sample
mapp = data.set_index('index')['yr'].drop_duplicates()
data_unbalanced = data.set_index(['index', 'yr']).reindex(
        pd.MultiIndex.from_product([data['index'].unique(),
                                data['yr'].unique()],
                                names=['index', 'yr'])
        ).reset_index()
data[['index', 'yr']].drop_duplicates().merge(data_unbalanced[['index', 'yr']], on='index')

# Limit summary statistics to variables of interest:
data_interest = data_unbalanced[['ldsal', 'lemp', 'ldnpt', 'ldrst', 'ldrnd', 'ldinv']]

# Calculate descriptive statistics for all
all_sample_stat = data_interest.describe().loc[['count', 'mean', '25%', '50%', '75%', 'std', 'min', 'max']]
print("Descriptive Statistics all:")
print(all_sample_stat)

# Group by 'index' and filter out groups with any missing values
subsample_data = data_unbalanced.groupby('index').filter(lambda x: not x.isnull().any().any())

# Limit summary statistics to variables of interest:
subdata_interest = subsample_data[['ldsal', 'lemp', 'ldnpt', 'ldrst', 'ldrnd', 'ldinv']]

# Calculate descriptive statistics for sub-sample
subsample_stat = subdata_interest.describe().loc[['count', 'mean', '25%', '50%', '75%', 'std', 'min', 'max']]

# Display the descriptive statistics and the formatted DataFrame
print("Descriptive Statistics Subsample:")
print(subsample_stat)

# Define Function for Task 2 and 3

def run_regression(data):
    # Combine 'sic3' and 'yr' to create unique identifiers
    data['sic3_yr'] = data['sic3'].astype(str) + '_yr' + data['yr'].astype(str)
    
    # Create dummy variables for the combined 'sic3_yr' column
    sic_dummies = pd.get_dummies(data['sic3_yr'], prefix='d357', drop_first=True).astype(int)
    
    # Drop the temporary 'sic3_yr' column
    data.drop(columns=['sic3_yr'], inplace=True)
    
    # Year Fixed Effects
    # Generate dummy variables for the year fixed effects
    year_dummies = pd.get_dummies(data['yr'], prefix='year', drop_first=True).astype(int)
    
    # Define the variables, excluding the original
    X = data[['lemp', 'ldnpt', 'ldrst']]
    
    # Add the year and sic dummies to the X_YEAR dataframe
    X_YEAR = pd.concat([X, year_dummies, sic_dummies], axis=1)
    
    # Add a constant for the intercept
    X_YEAR = sm.add_constant(X_YEAR)
    
    # Define the dependent variable
    Y = data['ldsal']
    
    # Fit the OLS regression model for year fixed effects
    year_model = sm.OLS(Y, X_YEAR)
    year_results = year_model.fit()
    
    # Get the summary of the results
    year_summary = year_results.summary2().tables[1]
    
    # Filter out the dummy coefficients
    non_dummy_coefficients_year = year_summary.loc[~year_summary.index.str.startswith('d357_')]
    non_dummy_coefficients_year = non_dummy_coefficients_year.loc[~non_dummy_coefficients_year.index.str.startswith('year_')]
    print("Year Fixed Effects Results:")
    print(non_dummy_coefficients_year)
    
    # Firm Fixed Effects
    # Generate dummy variables for the firm fixed effects
    firm_dummies = pd.get_dummies(data['index'], prefix='firm', drop_first=True).astype(int)
    
    # Add the year, firm, and sic dummies to the X_FIRM dataframe
    X_FIRM = pd.concat([X, year_dummies, firm_dummies, sic_dummies], axis=1)
    
    # Add a constant for the intercept
    X_FIRM = sm.add_constant(X_FIRM)
    
    # Fit the OLS regression model for firm fixed effects
    firm_model = sm.OLS(Y, X_FIRM)
    firm_results = firm_model.fit()
    
    # Get the summary of the results
    firm_summary = firm_results.summary2().tables[1]
    
    # Filter out the dummy coefficients
    non_dummy_coefficients_firm = firm_summary.loc[~firm_summary.index.str.startswith('d357_')]
    non_dummy_coefficients_firm = non_dummy_coefficients_firm.loc[~non_dummy_coefficients_firm.index.str.startswith('firm_')]
    non_dummy_coefficients_firm = non_dummy_coefficients_firm.loc[~non_dummy_coefficients_firm.index.str.startswith('year_')]
    print("Firm Fixed Effects Results:")
    print(non_dummy_coefficients_firm)

# Task 2: Year Fixed Effects
run_regression(subsample_data)

# Task 3: Firm Fixed Effects
run_regression(data)