import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel, RFE
import warnings
warnings.filterwarnings('ignore')

class CenterAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()  # Changed from RobustScaler
        self.models = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }
        self.best_model = None
        self.feature_importance = None
        
    def load_data(self, excel_file):
        """Load and prepare the datasets"""
        self.current_centres = pd.read_excel(excel_file, sheet_name="Current Centres")
        self.potential_centres = pd.read_excel(excel_file, sheet_name="Potential Centres")
        
    def engineer_features(self):
        """Simplified feature engineering matching original success"""
        for df in [self.current_centres, self.potential_centres]:
            # Basic features (matching original)
            df['TOTAL_BAYS'] = df['TYRE_BAYS'] + df['MOT_BAYS'] + df['SERVICE_BAYS']
            df['STAFF_EFFICIENCY'] = df['AVG_DAILY_STAFF'] / df['TOTAL_STAFF']
            df['RENT_PER_BAY'] = df['ANNUAL_RENT'] / df['TOTAL_BAYS']
            
            # Simpler affluence mapping (matching original)
            affluence_mapping = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1}
            df['AFFLUENCE_NUMERIC'] = df['AREA_AFFLUENCE_GRADE'].map(affluence_mapping)
            
        # Revenue-based features for current centres only
        self.current_centres['REVENUE_PER_BAY'] = self.current_centres['ANNUAL_REVENUE'] / self.current_centres['TOTAL_BAYS']
        self.current_centres['REVENUE_PER_STAFF'] = self.current_centres['ANNUAL_REVENUE'] / self.current_centres['TOTAL_STAFF']
        self.current_centres['REVENUE_PER_HOUR'] = self.current_centres['ANNUAL_REVENUE'] / (self.current_centres['HOURS_OPEN_PER_WEEK'] * 52)

    def prepare_features(self):
        """Simplified feature selection matching original success"""
        self.features = [
            'TOTAL_BAYS',
            'STAFF_EFFICIENCY',
            'RENT_PER_BAY',
            'AFFLUENCE_NUMERIC',
            'HOURS_OPEN_PER_WEEK',
            'AREA_EV_PERC',
            'AREA_POPULATION_DENSITY_PPSKM',
            'TOTAL_STAFF',
            'AVG_DAILY_STAFF'
        ]
        
        X = self.current_centres[self.features]
        y = self.current_centres['ANNUAL_REVENUE']
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_models(self, X_train, y_train):
        """Enhanced model training with better parameters"""
        best_score = 0
        
        for name, model in self.models.items():
            if name == 'Random Forest':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [15, 20, 25],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt', 'log2', None]
                }
            elif name == 'XGBoost':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 7, 9],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            else:  # Gradient Boosting
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 7, 9],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 0.9, 1.0]
                }
                
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
                self.best_model_name = name

    def analyze_centers(self):
        """Modified scoring system"""
        # Prepare potential centers data
        X_potential = self.potential_centres[self.features]
        X_potential_scaled = self.scaler.transform(X_potential)
        
        # Make predictions
        self.potential_centres['PREDICTED_REVENUE'] = self.best_model.predict(X_potential_scaled)
        
        # Calculate metrics
        self.potential_centres['ESTIMATED_PROFIT'] = (
            self.potential_centres['PREDICTED_REVENUE'] - 
            self.potential_centres['ANNUAL_RENT']
        )
        self.potential_centres['ESTIMATED_ROI'] = (
            self.potential_centres['ESTIMATED_PROFIT'] / 
            self.potential_centres['ANNUAL_RENT']
        )
        
        # Adjusted weights to match original success
        weights = {
            'REVENUE_WEIGHT': 0.35,  # Increased revenue weight
            'PROFIT_WEIGHT': 0.35,   # Increased profit weight
            'ROI_WEIGHT': 0.30       # Decreased ROI weight
        }
        
        # Enhanced scoring system
        self.potential_centres['COMBINED_SCORE'] = (
            weights['REVENUE_WEIGHT'] * self.potential_centres['PREDICTED_REVENUE'].rank(pct=True) +
            weights['PROFIT_WEIGHT'] * self.potential_centres['ESTIMATED_PROFIT'].rank(pct=True) +
            weights['ROI_WEIGHT'] * self.potential_centres['ESTIMATED_ROI'].rank(pct=True)
        )
        
        # Add validation step
        top_centers = self.potential_centres.sort_values('COMBINED_SCORE', ascending=False).head()
        if not validate_predictions(top_centers):
            print("Warning: Predictions may be outside expected ranges")
            
        return top_centers

# Add to your model evaluation
from sklearn.metrics import mean_absolute_percentage_error

def validate_predictions(predictions):
    """Validate predictions against industry benchmarks"""
    min_revenue = 1_000_000  # £1M minimum
    max_revenue = 3_000_000  # £3M maximum
    min_roi = 20  # 20% minimum ROI
    
    return (predictions['PREDICTED_REVENUE'].between(min_revenue, max_revenue) & 
            predictions['ESTIMATED_ROI'].gt(min_roi)).all()

# Main execution
if __name__ == "__main__":
    analyzer = CenterAnalyzer()
    analyzer.load_data("Practice Question.xlsx")
    analyzer.engineer_features()
    X_train, X_test, y_train, y_test = analyzer.prepare_features()
    analyzer.train_models(X_train, y_train)
    
    # Get recommendations
    top_recommendations = analyzer.analyze_centers()
    
    # Save results
    top_recommendations[['CENTRE_NO', 'PREDICTED_REVENUE', 'ESTIMATED_PROFIT', 
                        'ESTIMATED_ROI', 'COMBINED_SCORE']].to_csv('top_recommendations_improved.csv', 
                                                                 index=False)