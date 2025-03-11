import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# For reproducibility
np.random.seed(42)

# Load the datasets
# Note: In a real scenario, replace with your actual file path
excel_file = "Practice Question.xlsx"
current_centres = pd.read_excel(excel_file, sheet_name="Current Centres")
potential_centres = pd.read_excel(excel_file, sheet_name="Potential Centres")

# ------ Exploratory Data Analysis ------

print("Current Centres Dataset Overview:")
print(f"Number of centers: {current_centres.shape[0]}")
print(f"Number of features: {current_centres.shape[1]}")

print("\nPotential Centres Dataset Overview:")
print(f"Number of centers: {potential_centres.shape[0]}")
print(f"Number of features: {potential_centres.shape[1]}")

# Check for missing values
print("\nMissing values in Current Centres:")
print(current_centres.isnull().sum())

print("\nMissing values in Potential Centres:")
print(potential_centres.isnull().sum())

# Statistical summary for current centers
print("\nStatistical Summary for Current Centres:")
print(current_centres.describe())

# ------ Data Preprocessing ------

# Handle missing values by imputing with median
imputer = SimpleImputer(strategy='median')

# For current centers
current_centres['AREA_EV_PERC'] = imputer.fit_transform(current_centres[['AREA_EV_PERC']])
current_centres['AREA_POPULATION_DENSITY_PPSKM'] = imputer.fit_transform(current_centres[['AREA_POPULATION_DENSITY_PPSKM']])

# For potential centers
potential_centres['AREA_EV_PERC'] = imputer.fit_transform(potential_centres[['AREA_EV_PERC']])
potential_centres['AREA_POPULATION_DENSITY_PPSKM'] = imputer.fit_transform(potential_centres[['AREA_POPULATION_DENSITY_PPSKM']])

# Convert the categorical variable to numeric
affluence_mapping = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1}
current_centres['AFFLUENCE_NUMERIC'] = current_centres['AREA_AFFLUENCE_GRADE'].map(affluence_mapping)
potential_centres['AFFLUENCE_NUMERIC'] = potential_centres['AREA_AFFLUENCE_GRADE'].map(affluence_mapping)

# ------ Feature Engineering and Selection ------

# Create feature for revenue per staff member
current_centres['REVENUE_PER_STAFF'] = current_centres['ANNUAL_REVENUE'] / current_centres['TOTAL_STAFF']

# Create feature for revenue per bay (total bays)
current_centres['TOTAL_BAYS'] = (
    current_centres['TYRE_BAYS'] + 
    current_centres['MOT_BAYS'] + 
    current_centres['SERVICE_BAYS']
)
current_centres['REVENUE_PER_BAY'] = current_centres['ANNUAL_REVENUE'] / current_centres['TOTAL_BAYS']

# Create similar features for potential centers
potential_centres['TOTAL_BAYS'] = (
    potential_centres['TYRE_BAYS'] + 
    potential_centres['MOT_BAYS'] + 
    potential_centres['SERVICE_BAYS']
)

# Create a "staff efficiency" feature (avg daily staff / total staff)
current_centres['STAFF_EFFICIENCY'] = current_centres['AVG_DAILY_STAFF'] / current_centres['TOTAL_STAFF']
potential_centres['STAFF_EFFICIENCY'] = potential_centres['AVG_DAILY_STAFF'] / potential_centres['TOTAL_STAFF']

# Create a "rent per bay" feature
current_centres['RENT_PER_BAY'] = current_centres['ANNUAL_RENT'] / current_centres['TOTAL_BAYS']
potential_centres['RENT_PER_BAY'] = potential_centres['ANNUAL_RENT'] / potential_centres['TOTAL_BAYS']

# Create feature for 'weekly revenue potential' (considering open hours)
current_centres['REVENUE_PER_HOUR_OPEN'] = current_centres['ANNUAL_REVENUE'] / (current_centres['HOURS_OPEN_PER_WEEK'] * 52)
# Can't calculate this for potential centers as they don't have revenue data yet

# Analyze correlation with annual revenue
correlation_with_revenue = current_centres.corr(numeric_only=True)['ANNUAL_REVENUE'].sort_values(ascending=False)
print("\nCorrelation with ANNUAL_REVENUE:")
print(correlation_with_revenue)

# Visualize the top correlations
plt.figure(figsize=(12, 6))
top_corrs = correlation_with_revenue[:10]  # Top 10 correlations
sns.barplot(x=top_corrs.values, y=top_corrs.index)
plt.title('Top 10 Features Correlated with Annual Revenue')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('revenue_correlations.png')

# ------ Modeling ------

# Define the features to use in the model based on correlation analysis
features = [
    'TYRE_BAYS', 'MOT_BAYS', 'SERVICE_BAYS', 'TOTAL_STAFF', 
    'AVG_DAILY_STAFF', 'HOURS_OPEN_PER_WEEK', 
    'AREA_EV_PERC', 'AREA_POPULATION_DENSITY_PPSKM', 'ANNUAL_RENT', 
    'AFFLUENCE_NUMERIC', 'TOTAL_BAYS', 'STAFF_EFFICIENCY', 'RENT_PER_BAY'
]

X = current_centres[features]
y = current_centres['ANNUAL_REVENUE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models to find the best one
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100, max_depth=15),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1)
}

# Dictionary to store model performances
model_performances = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Store the performance metrics
    model_performances[name] = {'R2': r2, 'RMSE': rmse, 'model': model}
    
    print(f"\n{name}:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")

# Identify the best model based on R² score
best_model_name = max(model_performances, key=lambda k: model_performances[k]['R2'])
best_model = model_performances[best_model_name]['model']
best_r2 = model_performances[best_model_name]['R2']
best_rmse = model_performances[best_model_name]['RMSE']

print(f"\nBest Model: {best_model_name} with R² of {best_r2:.4f} and RMSE of {best_rmse:.2f}")

# Extract feature importances if using a tree-based model
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    importances = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(feature_importance)
    
    # Visualize feature importances
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(f'Feature Importances from {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importances.png')

# ------ Predict Revenue for Potential Centres ------

# Prepare potential centers data
X_potential = potential_centres[features]
X_potential_scaled = scaler.transform(X_potential)

# Make predictions
potential_centres['PREDICTED_REVENUE'] = best_model.predict(X_potential_scaled)

# Add profitability metrics
potential_centres['ESTIMATED_PROFIT'] = potential_centres['PREDICTED_REVENUE'] - potential_centres['ANNUAL_RENT']
potential_centres['ESTIMATED_ROI'] = potential_centres['ESTIMATED_PROFIT'] / potential_centres['ANNUAL_RENT']
potential_centres['REVENUE_PER_BAY'] = potential_centres['PREDICTED_REVENUE'] / potential_centres['TOTAL_BAYS']
potential_centres['REVENUE_PER_STAFF'] = potential_centres['PREDICTED_REVENUE'] / potential_centres['TOTAL_STAFF']

# Sort potential centers by predicted revenue
top_centers_by_revenue = potential_centres.sort_values(by='PREDICTED_REVENUE', ascending=False)
print("\nTop 5 Potential Centres by Predicted Revenue:")
print(top_centers_by_revenue[['CENTRE_NO', 'PREDICTED_REVENUE', 'ESTIMATED_PROFIT', 'ESTIMATED_ROI']].head(5))

# Sort potential centers by estimated profit
top_centers_by_profit = potential_centres.sort_values(by='ESTIMATED_PROFIT', ascending=False)
print("\nTop 5 Potential Centres by Estimated Profit:")
print(top_centers_by_profit[['CENTRE_NO', 'PREDICTED_REVENUE', 'ESTIMATED_PROFIT', 'ESTIMATED_ROI']].head(5))

# Sort potential centers by ROI
top_centers_by_roi = potential_centres.sort_values(by='ESTIMATED_ROI', ascending=False)
print("\nTop 5 Potential Centres by Estimated ROI:")
print(top_centers_by_roi[['CENTRE_NO', 'PREDICTED_REVENUE', 'ESTIMATED_PROFIT', 'ESTIMATED_ROI']].head(5))

# ------ Comprehensive Analysis of Top Centers ------

print("\n=== COMPREHENSIVE ANALYSIS OF TOP CENTERS ===")

# Function to analyze a center in detail
def analyze_center(center_no, metric_name):
    center_data = potential_centres[potential_centres['CENTRE_NO'] == center_no].iloc[0]
    print(f"\nCentre #{center_no} - Top by {metric_name}")
    print(f"Predicted Annual Revenue: £{center_data['PREDICTED_REVENUE']:,.2f}")
    print(f"Annual Rent: £{center_data['ANNUAL_RENT']:,.2f}")
    print(f"Estimated Profit: £{center_data['ESTIMATED_PROFIT']:,.2f}")
    print(f"Estimated ROI: {center_data['ESTIMATED_ROI']:.2f}")
    
    print("\nKey Center Characteristics:")
    print(f"- Area Affluence Grade: {center_data['AREA_AFFLUENCE_GRADE']}")
    print(f"- Tyre Bays: {center_data['TYRE_BAYS']}")
    print(f"- MOT Bays: {center_data['MOT_BAYS']}")
    print(f"- Service Bays: {center_data['SERVICE_BAYS']}")
    print(f"- Total Bays: {center_data['TOTAL_BAYS']}")
    print(f"- Total Staff: {center_data['TOTAL_STAFF']}")
    print(f"- Avg Daily Staff: {center_data['AVG_DAILY_STAFF']:.2f}")
    print(f"- Hours Open Per Week: {center_data['HOURS_OPEN_PER_WEEK']}")
    print(f"- Area EV Percentage: {center_data['AREA_EV_PERC']:.2f}%")
    print(f"- Population Density: {center_data['AREA_POPULATION_DENSITY_PPSKM']:.2f} people/km²")
    
    print("\nPerformance Metrics:")
    print(f"- Revenue Per Bay: £{center_data['REVENUE_PER_BAY']:,.2f}")
    print(f"- Revenue Per Staff: £{center_data['REVENUE_PER_STAFF']:,.2f}")
    print(f"- Rent Per Bay: £{center_data['RENT_PER_BAY']:,.2f}")
    print(f"- Staff Efficiency: {center_data['STAFF_EFFICIENCY']:.2f}")
    
    # Compare to average of current centers
    print("\nComparison to Current Centers (% above/below average):")
    current_avg_revenue = current_centres['ANNUAL_REVENUE'].mean()
    print(f"- Predicted Revenue: {(center_data['PREDICTED_REVENUE']/current_avg_revenue - 1)*100:.1f}%")
    
    current_avg_revenue_per_bay = current_centres['REVENUE_PER_BAY'].mean()
    print(f"- Revenue Per Bay: {(center_data['REVENUE_PER_BAY']/current_avg_revenue_per_bay - 1)*100:.1f}%")
    
    current_avg_revenue_per_staff = current_centres['REVENUE_PER_STAFF'].mean()
    print(f"- Revenue Per Staff: {(center_data['REVENUE_PER_STAFF']/current_avg_revenue_per_staff - 1)*100:.1f}%")

# Get top centers from each metric
top_revenue_center = top_centers_by_revenue.iloc[0]['CENTRE_NO']
top_profit_center = top_centers_by_profit.iloc[0]['CENTRE_NO']
top_roi_center = top_centers_by_roi.iloc[0]['CENTRE_NO']

# Analyze each top center
analyze_center(top_revenue_center, "Revenue")
analyze_center(top_profit_center, "Profit")
analyze_center(top_roi_center, "ROI")

# ------ Final Recommendation ------

print("\n=== FINAL RECOMMENDATION ===")
# We'll base our recommendation on balanced consideration of revenue, profit, and ROI
# Create a combined score
potential_centres['REV_PERCENTILE'] = potential_centres['PREDICTED_REVENUE'].rank(pct=True)
potential_centres['PROFIT_PERCENTILE'] = potential_centres['ESTIMATED_PROFIT'].rank(pct=True)
potential_centres['ROI_PERCENTILE'] = potential_centres['ESTIMATED_ROI'].rank(pct=True)

# Calculate a balanced score (give more weight to ROI since it's a relative measure)
potential_centres['COMBINED_SCORE'] = (
    potential_centres['REV_PERCENTILE'] * 0.3 + 
    potential_centres['PROFIT_PERCENTILE'] * 0.3 + 
    potential_centres['ROI_PERCENTILE'] * 0.4
)

# Find the center with the best combined score
best_center_overall = potential_centres.sort_values(by='COMBINED_SCORE', ascending=False).iloc[0]
print(f"The recommended center to open is Centre #{best_center_overall['CENTRE_NO']}.")
print(f"Predicted Annual Revenue: £{best_center_overall['PREDICTED_REVENUE']:,.2f}")
print(f"Estimated Annual Profit: £{best_center_overall['ESTIMATED_PROFIT']:,.2f}")
print(f"Estimated ROI: {best_center_overall['ESTIMATED_ROI']:.2f}")

print("\nThis center ranks:")
print(f"- #{potential_centres.sort_values('PREDICTED_REVENUE', ascending=False)['CENTRE_NO'].tolist().index(best_center_overall['CENTRE_NO'])+1} in Revenue")
print(f"- #{potential_centres.sort_values('ESTIMATED_PROFIT', ascending=False)['CENTRE_NO'].tolist().index(best_center_overall['CENTRE_NO'])+1} in Profit")
print(f"- #{potential_centres.sort_values('ESTIMATED_ROI', ascending=False)['CENTRE_NO'].tolist().index(best_center_overall['CENTRE_NO'])+1} in ROI")

print("\nKey Factors Contributing to This Recommendation:")
for feature in feature_importance['Feature'].values[:5]:  # Top 5 important features
    print(f"- {feature}: {best_center_overall[feature]}")

print("\nThis recommendation is based on a {best_model_name} model with an R² score of {best_r2:.4f}.")
print("The center was chosen using a balanced approach considering revenue potential, profit, and return on investment.")
print("This provides the best overall opportunity considering both short-term profitability and long-term revenue growth.")

# Save final recommendation to file
top_5_overall = potential_centres.sort_values(by='COMBINED_SCORE', ascending=False).head(5)
top_5_overall[['CENTRE_NO', 'PREDICTED_REVENUE', 'ESTIMATED_PROFIT', 'ESTIMATED_ROI', 'COMBINED_SCORE']].to_csv('top_recommendations.csv', index=False)