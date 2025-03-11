import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

class OptimalCenterAnalyzer:
    def __init__(self):
        # Initialize models for ensemble
        self.models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=20,
                min_samples_split=5,
                random_state=42
            ),
            'XGBoost': XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42
            ),
            'Ridge': Ridge(alpha=1.0)
        }
        self.scaler = StandardScaler()
        
    def load_data(self, excel_file):
        """Load and perform initial data checks"""
        self.current_centres = pd.read_excel(excel_file, sheet_name="Current Centres")
        self.potential_centres = pd.read_excel(excel_file, sheet_name="Potential Centres")
        
        print(f"Current centers: {self.current_centres.shape[0]}")
        print(f"Potential centers: {self.potential_centres.shape[0]}")
        
        # Check for missing values
        print("\nMissing values check:")
        print(self.current_centres.isnull().sum()[self.current_centres.isnull().sum() > 0])
        print(self.potential_centres.isnull().sum()[self.potential_centres.isnull().sum() > 0])
        
    def handle_missing_values(self):
        """Advanced missing value handling with KNN for similar areas"""
        # Create affluence numeric for grouping
        affluence_mapping = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1}
        for df in [self.current_centres, self.potential_centres]:
            df['AFFLUENCE_NUMERIC'] = df['AREA_AFFLUENCE_GRADE'].map(affluence_mapping)
        
        # Use KNN imputation for geographic features
        imputer = KNNImputer(n_neighbors=5)
        
        # For current centers
        cols_to_impute = ['AREA_EV_PERC', 'AREA_POPULATION_DENSITY_PPSKM']
        self.current_centres[cols_to_impute] = imputer.fit_transform(
            self.current_centres[['AFFLUENCE_NUMERIC'] + cols_to_impute]
        )[:, 1:]
        
        # For potential centers
        self.potential_centres[cols_to_impute] = imputer.transform(
            self.potential_centres[['AFFLUENCE_NUMERIC'] + cols_to_impute]
        )[:, 1:]

    def engineer_features(self):
        """Strategic feature engineering based on domain knowledge"""
        for df in [self.current_centres, self.potential_centres]:
            # Basic features - core operational metrics
            df['TOTAL_BAYS'] = df['TYRE_BAYS'] + df['MOT_BAYS'] + df['SERVICE_BAYS']
            df['STAFF_EFFICIENCY'] = df['AVG_DAILY_STAFF'] / df['TOTAL_STAFF']
            df['RENT_PER_BAY'] = df['ANNUAL_RENT'] / df['TOTAL_BAYS']
            
            # Advanced operational features
            df['HOURS_PER_STAFF_WEEKLY'] = df['HOURS_OPEN_PER_WEEK'] * df['AVG_DAILY_STAFF'] / 5
            df['RENT_PER_HOUR_OPEN'] = df['ANNUAL_RENT'] / (df['HOURS_OPEN_PER_WEEK'] * 52)
            
            # Location-based features
            df['MARKET_POTENTIAL'] = df['AFFLUENCE_NUMERIC'] * df['AREA_POPULATION_DENSITY_PPSKM'] * (1 + df['AREA_EV_PERC']/100)
        
        # Revenue metrics for current centers
        self.current_centres['REVENUE_PER_BAY'] = self.current_centres['ANNUAL_REVENUE'] / self.current_centres['TOTAL_BAYS']
        self.current_centres['REVENUE_PER_STAFF'] = self.current_centres['ANNUAL_REVENUE'] / self.current_centres['TOTAL_STAFF']
        self.current_centres['REVENUE_PER_HOUR'] = self.current_centres['ANNUAL_REVENUE'] / (self.current_centres['HOURS_OPEN_PER_WEEK'] * 52)
        
    def analyze_feature_importance(self):
        """Analyze and visualize feature correlations"""
        # Select only numeric columns for correlation
        numeric_cols = self.current_centres.select_dtypes(include=np.number).columns.tolist()
        
        # Make sure ANNUAL_REVENUE is in numeric_cols
        if 'ANNUAL_REVENUE' not in numeric_cols:
            print("Warning: ANNUAL_REVENUE not found in numeric columns")
            return None
            
        # Calculate correlations using only numeric columns
        correlation = self.current_centres[numeric_cols].corr()['ANNUAL_REVENUE'].sort_values(ascending=False)
        print("\nCorrelation with ANNUAL_REVENUE:")
        print(correlation)
        
        # Visualize top correlations
        plt.figure(figsize=(12, 8))
        top_features = correlation.iloc[:10].index
        sns.heatmap(
            self.current_centres[top_features].corr(), 
            annot=True, 
            cmap='coolwarm'
        )
        plt.title('Correlation Between Top Features')
        plt.tight_layout()
        plt.savefig('feature_correlation.png')
        
        return correlation
        
    def prepare_training_data(self):
        """Prepare features for modeling"""
        # Choose features based on correlation and domain knowledge
        self.features = [
            'TOTAL_BAYS', 
            'TYRE_BAYS',
            'MOT_BAYS',
            'SERVICE_BAYS',
            'TOTAL_STAFF',
            'AVG_DAILY_STAFF',
            'STAFF_EFFICIENCY',
            'HOURS_OPEN_PER_WEEK',
            'HOURS_PER_STAFF_WEEKLY',
            'RENT_PER_BAY',
            'RENT_PER_HOUR_OPEN',
            'AFFLUENCE_NUMERIC',
            'AREA_EV_PERC',
            'AREA_POPULATION_DENSITY_PPSKM',
            'MARKET_POTENTIAL'
        ]
        
        X = self.current_centres[self.features]
        y = self.current_centres['ANNUAL_REVENUE']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and evaluate performance"""
        print("\n=== Model Performance ===")
        best_model = None
        best_score = 0
        best_model_name = ""
        
        for name, model in self.models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # Fit on training data
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"{name}:")
            print(f"  CV R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"  Test R² Score: {r2:.4f}")
            print(f"  RMSE: {rmse:.2f}")
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_model_name = name
        
        print(f"\nBest Model: {best_model_name} with R² of {best_score:.4f}")
        
        # Feature importance for Random Forest
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_importance = pd.DataFrame({
                'Feature': self.features,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nFeature Importances:")
            print(feature_importance.head(10))
            
            self.feature_importance = feature_importance
        
        self.best_model = best_model
        self.best_model_name = best_model_name
        self.best_r2 = best_score
        
        return best_model
        
    def evaluate_potential_centers(self):
        """Predict performance and score potential centers"""
        X_potential = self.potential_centres[self.features]
        X_potential_scaled = self.scaler.transform(X_potential)
        
        # Predict revenue
        self.potential_centres['PREDICTED_REVENUE'] = self.best_model.predict(X_potential_scaled)
        
        # Calculate profit and ROI
        self.potential_centres['ESTIMATED_PROFIT'] = (
            self.potential_centres['PREDICTED_REVENUE'] - 
            self.potential_centres['ANNUAL_RENT']
        )
        
        self.potential_centres['ESTIMATED_ROI'] = (
            self.potential_centres['ESTIMATED_PROFIT'] / 
            self.potential_centres['ANNUAL_RENT']
        )
        
        # Create percentile ranks
        self.potential_centres['REV_PERCENTILE'] = self.potential_centres['PREDICTED_REVENUE'].rank(pct=True)
        self.potential_centres['PROFIT_PERCENTILE'] = self.potential_centres['ESTIMATED_PROFIT'].rank(pct=True)
        self.potential_centres['ROI_PERCENTILE'] = self.potential_centres['ESTIMATED_ROI'].rank(pct=True)
        
        # Calculate weighted score - REVENUE BIAS
        self.potential_centres['COMBINED_SCORE'] = (
            self.potential_centres['REV_PERCENTILE'] * 0.35 + 
            self.potential_centres['PROFIT_PERCENTILE'] * 0.35 + 
            self.potential_centres['ROI_PERCENTILE'] * 0.30
        )
        
        # Get top centers by different metrics
        print("\n=== Top 5 Potential Centres by Predicted Revenue ===")
        revenue_top = self.potential_centres.sort_values('PREDICTED_REVENUE', ascending=False).head(5)
        print(revenue_top[['CENTRE_NO', 'PREDICTED_REVENUE', 'ESTIMATED_PROFIT', 'ESTIMATED_ROI']])
        
        print("\n=== Top 5 Potential Centres by Estimated Profit ===")
        profit_top = self.potential_centres.sort_values('ESTIMATED_PROFIT', ascending=False).head(5)
        print(profit_top[['CENTRE_NO', 'PREDICTED_REVENUE', 'ESTIMATED_PROFIT', 'ESTIMATED_ROI']])
        
        print("\n=== Top 5 Potential Centres by Estimated ROI ===")
        roi_top = self.potential_centres.sort_values('ESTIMATED_ROI', ascending=False).head(5)
        print(roi_top[['CENTRE_NO', 'PREDICTED_REVENUE', 'ESTIMATED_PROFIT', 'ESTIMATED_ROI']])
        
        print("\n=== Top 5 Potential Centres by Combined Score ===")
        combined_top = self.potential_centres.sort_values('COMBINED_SCORE', ascending=False).head(5)
        print(combined_top[['CENTRE_NO', 'PREDICTED_REVENUE', 'ESTIMATED_PROFIT', 'ESTIMATED_ROI', 'COMBINED_SCORE']])
        
        # Save recommendations to file
        top_recommendations = self.potential_centres.sort_values('COMBINED_SCORE', ascending=False).head(5)
        top_recommendations[['CENTRE_NO', 'PREDICTED_REVENUE', 'ESTIMATED_PROFIT', 'ESTIMATED_ROI', 'COMBINED_SCORE']].to_csv(
            'optimal_recommendations.csv', index=False
        )
        
        return combined_top.iloc[0]  # Return best center
        
    def generate_final_recommendation(self, best_center):
        """Generate detailed recommendation report"""
        print("\n=== FINAL RECOMMENDATION ===")
        print(f"The recommended center to open is Centre #{best_center['CENTRE_NO']}.")
        print(f"Predicted Annual Revenue: £{best_center['PREDICTED_REVENUE']:,.2f}")
        print(f"Estimated Annual Profit: £{best_center['ESTIMATED_PROFIT']:,.2f}")
        print(f"Estimated ROI: {best_center['ESTIMATED_ROI']:.2f}")
        
        print("\nThis center ranks:")
        rev_rank = self.potential_centres.sort_values('PREDICTED_REVENUE', ascending=False)['CENTRE_NO'].tolist().index(
            best_center['CENTRE_NO'])+1
        profit_rank = self.potential_centres.sort_values('ESTIMATED_PROFIT', ascending=False)['CENTRE_NO'].tolist().index(
            best_center['CENTRE_NO'])+1
        roi_rank = self.potential_centres.sort_values('ESTIMATED_ROI', ascending=False)['CENTRE_NO'].tolist().index(
            best_center['CENTRE_NO'])+1
            
        print(f"- #{rev_rank} in Revenue")
        print(f"- #{profit_rank} in Profit")
        print(f"- #{roi_rank} in ROI")
        
        print("\nKey Factors Contributing to This Recommendation:")
        for feature in self.feature_importance['Feature'].values[:5]:  # Top 5 important features
            print(f"- {feature}: {best_center[feature]}")
        
        print(f"\nThis recommendation is based on a {self.best_model_name} model with an R² score of {self.best_r2:.4f}.")
        print("The center was chosen using a balanced approach considering revenue potential, profit, and return on investment.")
        print("This provides the best overall opportunity considering both short-term profitability and long-term revenue growth.")
        
    def run_full_analysis(self, excel_file):
        """Run complete analysis pipeline"""
        # Step 1: Load and prep data
        self.load_data(excel_file)
        
        # Step 2: Handle missing values
        self.handle_missing_values()
        
        # Step 3: Engineer features
        self.engineer_features()
        
        # Step 4: Analyze features
        self.analyze_feature_importance()
        
        # Step 5: Prepare training data
        X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = self.prepare_training_data()
        
        # Step 6: Train and evaluate models
        best_model = self.train_models(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Step 7: Evaluate potential centers
        best_center = self.evaluate_potential_centers()
        
        # Step 8: Generate final recommendation
        self.generate_final_recommendation(best_center)
        
        return best_center

# Execute analysis
if __name__ == "__main__":
    analyzer = OptimalCenterAnalyzer()
    analyzer.run_full_analysis("Practice Question.xlsx")