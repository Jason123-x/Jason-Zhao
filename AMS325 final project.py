import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# ------------------ Step 1: Data Preparation ------------------ #
# Load the uploaded Excel files
csi800_df = pd.read_excel('/Users/jason/Downloads/CSI800.xlsx')
shibor_df = pd.read_excel('/Users/jason/Downloads/shibor(day).xlsx')

# Convert dates and drop rows with invalid dates
csi800_df['Month'] = pd.to_datetime(csi800_df['Month'], format='%Y-%m-%d', errors='coerce')
shibor_df['SgnDate'] = pd.to_datetime(shibor_df['SgnDate'], format='%Y-%m-%d', errors='coerce')

# Drop rows with NaT values in date columns
csi800_df = csi800_df.dropna(subset=['Month'])
shibor_df = shibor_df.dropna(subset=['SgnDate'])

# Ensure the 'Month' and 'SgnDate' columns are in ascending order
csi800_df = csi800_df.sort_values(by='Month')
shibor_df = shibor_df.sort_values(by='SgnDate')

# Set DatetimeIndex before resampling
csi800_df = csi800_df.set_index('Month')
shibor_df = shibor_df.set_index('SgnDate')

# Resample Shibor rates to weekly average
shibor_weekly = shibor_df[shibor_df['Term'] == '1天'].resample('W-FRI').mean().reset_index()

# Resample CSI800 prices to weekly frequency (Friday closing)
csi800_prices_weekly = csi800_df.resample('W-FRI').last().reset_index()

# Calculate weekly returns
csi800_prices_weekly['Weekly Return'] = csi800_prices_weekly['Idxrtn'].pct_change()

# Merge weekly Shibor rates with CSI800 weekly returns
merged_data = pd.merge(csi800_prices_weekly, shibor_weekly, left_on='Month', right_on='SgnDate', how='inner')

# Drop redundant columns and rows with NaN
merged_data = merged_data.drop(columns=['SgnDate'])
merged_data = merged_data.dropna(subset=['Weekly Return', 'Shibor'])

# Save the merged data to a CSV file
output_file_path = '/Users/jason/Downloads/merged_data.csv'
merged_data.to_csv(output_file_path, index=False)

# Display the first few rows of the merged data
merged_data.head()


# ------------------ Step 2: Time-Series Regression (Period 1) ------------------ #
# Define Periods
period_1 = merged_data[(merged_data['Month'] >= '2017-01-06') & (merged_data['Month'] <= '2018-05-04')]
period_2 = merged_data[(merged_data['Month'] >= '2018-05-11') & (merged_data['Month'] <= '2019-09-06')]
period_3 = merged_data[(merged_data['Month'] >= '2019-09-13') & (merged_data['Month'] <= '2020-12-25')]

# Excess Return Calculation
period_1['Excess Return'] = period_1['Weekly Return'] - period_1['Shibor']
period_1['Market Excess Return'] = period_1['Idxrtn'] - period_1['Shibor']

# Simulate beta estimation for stocks (assuming multiple stocks)
def time_series_regression(data, market_col, rf_col, stock_col):
    X = (data[market_col] - data[rf_col]).values.reshape(-1, 1)
    y = (data[stock_col] - data[rf_col]).values
    model = LinearRegression().fit(X, y)
    return model.intercept_, model.coef_[0]

betas = np.random.normal(1, 0.5, 100)  # Simulated betas for 100 stocks
portfolio_thresholds = np.percentile(betas, [25, 50, 75])

# ------------------ Step 3: Portfolio Construction (Period 2) ------------------ #
# Assign stocks to portfolios based on beta
portfolios = np.digitize(betas, bins=portfolio_thresholds, right=True)
portfolio_avg_returns = np.random.normal(0.02, 0.01, 4)  # Simulated average portfolio returns
portfolio_betas = [np.mean(betas[portfolios == i]) for i in range(4)]

# ------------------ Step 4: Cross-Sectional Regression (Period 3) ------------------ #
# Run regression: Rp,t - Rf,t = λ0 + λ1 * Bp
X_cross = sm.add_constant(portfolio_betas)
y_cross = portfolio_avg_returns
cross_section_model = sm.OLS(y_cross, X_cross).fit()

lambda_0 = cross_section_model.params[0]
lambda_1 = cross_section_model.params[1]
r_squared = cross_section_model.rsquared


# ------------------ Step 5: Summary Statistics ------------------ #
beta_stats = {
    'mean': np.mean(betas),
    'sd': np.std(betas),
    'min': np.min(betas),
    'max': np.max(betas),
    'p1': np.percentile(betas, 1),
    'p25': np.percentile(betas, 25),
    'p50': np.percentile(betas, 50),
    'p75': np.percentile(betas, 75),
    'p99': np.percentile(betas, 99)
}

# ------------------ Step 5: Output Results ------------------ #
print("### Cross-Sectional Regression Results ###")
print(f"Lambda_0 (Intercept): {lambda_0:.4f}")
print(f"Lambda_1 (Slope): {lambda_1:.4f}")
print(f"R-squared: {r_squared:.4f}")
print("\nPortfolio Betas and Average Returns:")
for i, (beta, ret) in enumerate(zip(portfolio_betas, portfolio_avg_returns)):
    print(f"Portfolio {i+1}: Beta = {beta:.4f}, Average Return = {ret:.4f}")

results_df = pd.DataFrame({
    'Variable': ['_b_rmr', 'cons'],
    'Coef.': [lambda_1, lambda_0],
    'Std. Err.': [cross_section_model.bse[1], cross_section_model.bse[0]],
    't': [cross_section_model.tvalues[1], cross_section_model.tvalues[0]],
    'P>|t|': [cross_section_model.pvalues[1], cross_section_model.pvalues[0]],
    '[95% Conf. Interval]': [f"{cross_section_model.conf_int()[1][0]:.4f} to {cross_section_model.conf_int()[1][1]:.4f}",
                             f"{cross_section_model.conf_int()[0][0]:.4f} to {cross_section_model.conf_int()[0][1]:.4f}"]
})

print("\n### Regression Coefficients Table ###")
print(results_df.to_string(index=False))
