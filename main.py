import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
import ast
import warnings
warnings.filterwarnings('ignore')

class WhiteSpaceIdentificationSystem:
    """
    A comprehensive system for identifying white space opportunities in CRM data.
    
    This system uses multiple machine learning approaches:
    1. Random Forest Classification for predictive modeling
    2. K-Nearest Neighbors for similarity-based recommendations
    3. Association Rules for cross-selling opportunities
    4. Ensemble method combining all approaches
    """
    
    def __init__(self):
        self.rf_models = {}  # Random Forest models for each product
        self.knn_model = None  # KNN model for similarity
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.data = None
        self.product_matrix = None
        self.feature_matrix = None
        self.all_products = None
        self.association_rules = {}
        
    def load_and_preprocess_data(self, csv_file='crm_data.csv'):
        """Load and preprocess the CRM data"""
        print("Loading CRM data...")
        
        # Load data
        self.data = pd.read_csv(csv_file)
        print(f"Loaded {len(self.data)} accounts")
        
        # Display basic info
        print(f"Industries: {list(self.data['Industry'].unique())}")
        
        # Parse products and create matrices
        self._parse_products()
        self._create_features()
        self._create_product_matrix()
        
        print("Data preprocessing completed\n")
        
    def _parse_products(self):
        """Parse products from the Products_Sold_List column"""
        print("Parsing product information...")
        
        all_products_set = set()
        
        # Extract all unique products
        for products_str in self.data['Products_Sold_List']:
            try:
                # Convert string representation of list to actual list
                products = ast.literal_eval(products_str)
                all_products_set.update(products)
            except:
                # Fallback: split by comma if ast.literal_eval fails
                products = [p.strip().strip("'\"") for p in products_str.replace('[', '').replace(']', '').split(',')]
                all_products_set.update(products)
        
        self.all_products = list(all_products_set)
        print(f"Found products: {self.all_products}")
        
    def _create_features(self):
        """Create feature matrix for machine learning"""
        print("Creating features...")
        
        # Select numerical features
        numerical_features = [
            'Contacts', 'Active_Opps', 'Won_Opps', 'Lost_Opps',
            'Calls', 'Meetings', 'Tasks', 'Emails'
        ]
        
        # Create derived features
        feature_data = self.data.copy()
        
        # Calculate business metrics
        feature_data['Win_Rate'] = feature_data['Won_Opps'] / (feature_data['Won_Opps'] + feature_data['Lost_Opps'] + 0.001)
        feature_data['Total_Opps'] = feature_data['Active_Opps'] + feature_data['Won_Opps'] + feature_data['Lost_Opps']
        feature_data['Activity_Score'] = feature_data['Calls'] + feature_data['Meetings'] + feature_data['Tasks'] + feature_data['Emails']
        feature_data['Engagement_Ratio'] = feature_data['Activity_Score'] / (feature_data['Contacts'] + 0.001)
        feature_data['Products_Count'] = feature_data['Products_Sold'].apply(lambda x: len(x.split(',')) if pd.notnull(x) else 0)
        
        # Encode categorical features
        feature_data['Industry_Encoded'] = self.label_encoder.fit_transform(feature_data['Industry'])
        
        # Define final features for modeling
        self.model_features = [
            'Contacts', 'Active_Opps', 'Won_Opps', 'Lost_Opps',
            'Calls', 'Meetings', 'Tasks', 'Emails',
            'Win_Rate', 'Total_Opps', 'Activity_Score', 'Engagement_Ratio',
            'Products_Count', 'Industry_Encoded'
        ]
        
        # Create and scale feature matrix
        self.feature_matrix = feature_data[['Account'] + self.model_features].copy()
        
        # Scale numerical features (exclude categorical Industry_Encoded)
        features_to_scale = [col for col in self.model_features if col != 'Industry_Encoded']
        self.feature_matrix[features_to_scale] = self.scaler.fit_transform(
            self.feature_matrix[features_to_scale]
        )
        
        print(f"Created {len(self.model_features)} features")
        
    def _create_product_matrix(self):
        """Create binary product-account matrix"""
        print("Creating product-account matrix...")
        
        # Initialize product matrix
        product_data = []
        
        for _, row in self.data.iterrows():
            account_products = {'Account': row['Account']}
            
            # Parse products for this account
            try:
                owned_products = ast.literal_eval(row['Products_Sold_List'])
            except:
                owned_products = [p.strip().strip("'\"") for p in row['Products_Sold_List'].replace('[', '').replace(']', '').split(',')]
            
            # Create binary columns for each product
            for product in self.all_products:
                account_products[product] = 1 if product in owned_products else 0
                
            product_data.append(account_products)
        
        self.product_matrix = pd.DataFrame(product_data)
        print(f"Product matrix created: {self.product_matrix.shape}")
        
    def train_random_forest_models(self):
        """Train Random Forest models for each product"""
        print("Training Random Forest models...")
        
        # Merge feature matrix with product matrix
        training_data = self.feature_matrix.merge(self.product_matrix, on='Account')
        
        for product in self.all_products:
            X = training_data[self.model_features]
            y = training_data[product]
            
            # Check if we have enough positive examples
            if y.sum() < 2:
                print(f"Skipping {product} - insufficient positive examples ({y.sum()})")
                continue
            
            # Train Random Forest
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=2,
                random_state=42,
                class_weight='balanced'
            )
            
            rf.fit(X, y)
            self.rf_models[product] = rf
            
            # Calculate cross-validation score
            try:
                cv_scores = cross_val_score(rf, X, y, cv=3, scoring='roc_auc')
                print(f"  {product}: CV AUC = {cv_scores.mean():.3f}")
            except:
                print(f"  {product}: Model trained")
        
        print(f"Trained {len(self.rf_models)} Random Forest models")
        
    def train_knn_model(self):
        """Train KNN model for similarity-based recommendations"""
        print("Training KNN similarity model...")
        
        X = self.feature_matrix[self.model_features].values
        
        self.knn_model = NearestNeighbors(
            n_neighbors=min(5, len(self.data)),
            metric='cosine',
            algorithm='brute'
        )
        
        self.knn_model.fit(X)
        print("KNN model trained")
        
    def mine_association_rules(self):
        """Mine product association patterns for cross-selling"""
        print("Mining product association patterns...")
        
        self.association_rules = {}
        
        # For each product, find which other products are commonly bought together
        for product in self.all_products:
            product_accounts = self.product_matrix[self.product_matrix[product] == 1]
            
            if len(product_accounts) == 0:
                continue
                
            # Find co-occurring products
            co_occurrence = {}
            for other_product in self.all_products:
                if other_product != product:
                    co_count = (product_accounts[other_product] == 1).sum()
                    if co_count > 0:
                        confidence = co_count / len(product_accounts)
                        co_occurrence[other_product] = confidence
            
            self.association_rules[product] = co_occurrence
            
        print("Association rules mined")
        
    def predict_whitespace_rf(self, account_name, top_n=5):
        """Predict using Random Forest models"""
        predictions = []
        
        # Get account features
        account_data = self.feature_matrix[self.feature_matrix['Account'] == account_name]
        if account_data.empty:
            return predictions
            
        X = account_data[self.model_features].values
        
        # Get current products
        current_products = self._get_current_products(account_name)
        
        # Predict for each product
        for product, model in self.rf_models.items():
            if product in current_products:
                continue  # Skip products already owned
                
            prob = model.predict_proba(X)[0]
            confidence = prob[1] if len(prob) > 1 else prob[0]
            
            predictions.append({
                'product': product,
                'confidence': confidence,
                'method': 'Random Forest'
            })
        
        return sorted(predictions, key=lambda x: x['confidence'], reverse=True)[:top_n]
        
    def predict_whitespace_knn(self, account_name, top_n=5):
        """Predict using KNN similarity"""
        predictions = []
        
        # Get account features
        account_data = self.feature_matrix[self.feature_matrix['Account'] == account_name]
        if account_data.empty:
            return predictions
            
        X = account_data[self.model_features].values
        
        # Find similar accounts
        distances, indices = self.knn_model.kneighbors(X)
        similar_accounts = self.feature_matrix.iloc[indices[0]]['Account'].tolist()
        
        # Get current products
        current_products = self._get_current_products(account_name)
        
        # Calculate recommendation scores based on similar accounts
        for product in self.all_products:
            if product in current_products:
                continue
                
            # Count how many similar accounts have this product
            similar_with_product = 0
            for sim_account in similar_accounts:
                sim_products = self.product_matrix[self.product_matrix['Account'] == sim_account]
                if not sim_products.empty and sim_products[product].iloc[0] == 1:
                    similar_with_product += 1
            
            if similar_with_product > 0:
                confidence = similar_with_product / len(similar_accounts)
                predictions.append({
                    'product': product,
                    'confidence': confidence,
                    'similar_accounts': similar_with_product,
                    'method': 'KNN Similarity'
                })
        
        return sorted(predictions, key=lambda x: x['confidence'], reverse=True)[:top_n]
        
    def predict_whitespace_rules(self, account_name, top_n=5):
        """Predict using association rules"""
        predictions = []
        
        # Get current products
        current_products = self._get_current_products(account_name)
        
        if not current_products:
            return predictions
            
        # Apply association rules
        recommended_products = {}
        
        for owned_product in current_products:
            if owned_product in self.association_rules:
                for recommended, confidence in self.association_rules[owned_product].items():
                    if recommended not in current_products:
                        if recommended not in recommended_products:
                            recommended_products[recommended] = confidence
                        else:
                            recommended_products[recommended] = max(
                                recommended_products[recommended], confidence
                            )
        
        for product, confidence in recommended_products.items():
            predictions.append({
                'product': product,
                'confidence': confidence,
                'method': 'Association Rules'
            })
        
        return sorted(predictions, key=lambda x: x['confidence'], reverse=True)[:top_n]
        
    def _get_current_products(self, account_name):
        """Helper method to get current products for an account"""
        current_products = set()
        account_products = self.product_matrix[self.product_matrix['Account'] == account_name]
        if not account_products.empty:
            for product in self.all_products:
                if account_products[product].iloc[0] == 1:
                    current_products.add(product)
        return current_products
        
    def ensemble_predict(self, account_name, top_n=5):
        """Combine all prediction methods using ensemble approach"""
        # Get predictions from all methods
        rf_predictions = self.predict_whitespace_rf(account_name)
        knn_predictions = self.predict_whitespace_knn(account_name)
        rule_predictions = self.predict_whitespace_rules(account_name)
        
        # Combine predictions with weights
        weights = {'Random Forest': 0.4, 'KNN Similarity': 0.4, 'Association Rules': 0.2}
        combined_scores = {}
        
        # Process all predictions
        for pred_list in [rf_predictions, knn_predictions, rule_predictions]:
            for pred in pred_list:
                product = pred['product']
                method = pred['method']
                
                if product not in combined_scores:
                    combined_scores[product] = {
                        'score': 0,
                        'methods': [],
                        'details': {}
                    }
                
                combined_scores[product]['score'] += pred['confidence'] * weights[method]
                combined_scores[product]['methods'].append(method)
                combined_scores[product]['details'][method] = pred
        
        # Create final recommendations
        final_recommendations = []
        for product, data in combined_scores.items():
            final_recommendations.append({
                'product': product,
                'ensemble_score': data['score'],
                'confidence_level': 'High' if data['score'] > 0.6 else 'Medium' if data['score'] > 0.3 else 'Low',
                'methods_used': ', '.join(set(data['methods'])),
                'num_methods': len(set(data['methods'])),
                'details': data['details']
            })
        
        return sorted(final_recommendations, key=lambda x: x['ensemble_score'], reverse=True)[:top_n]
        
    def explain_recommendations(self, recommendations, account_name):
        """Provide detailed explanations for recommendations"""
        print(f"WHITE SPACE ANALYSIS FOR: {account_name}")
        print("=" * 60)
        
        # Get account info
        account_info = self.data[self.data['Account'] == account_name].iloc[0]
        print(f"Industry: {account_info['Industry']}")
        print(f"Contacts: {account_info['Contacts']}")
        print(f"Won Opportunities: {account_info['Won_Opps']}")
        print(f"Current Products: {account_info['Products_Sold']}")
        
        print(f"\nTOP RECOMMENDATIONS:")
        print("-" * 40)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['product']}")
            print(f"   Ensemble Score: {rec['ensemble_score']:.3f}")
            print(f"   Confidence: {rec['confidence_level']}")
            print(f"   Methods: {rec['methods_used']}")
            
            # Detailed reasoning
            details = rec['details']
            print("   Reasoning:")
            
            if 'Random Forest' in details:
                rf_conf = details['Random Forest']['confidence']
                print(f"   - ML Model predicts {rf_conf:.3f} probability based on account characteristics")
            
            if 'KNN Similarity' in details:
                knn_detail = details['KNN Similarity']
                similar_count = knn_detail.get('similar_accounts', 0)
                print(f"   - {similar_count} similar accounts already use this product")
            
            if 'Association Rules' in details:
                rule_conf = details['Association Rules']['confidence']
                print(f"   - Product association patterns suggest {rule_conf:.3f} likelihood")
        
        return recommendations
        
    def train_all_models(self):
        """Train all models in the system"""
        print("TRAINING ALL MODELS")
        print("=" * 40)
        
        self.train_random_forest_models()
        self.train_knn_model()
        self.mine_association_rules()
        
        print("All models trained successfully!\n")
        
    def analyze_account(self, account_name):
        """Complete analysis for an account"""
        if account_name not in self.data['Account'].values:
            print(f"Account '{account_name}' not found!")
            available_accounts = list(self.data['Account'])
            print(f"Available accounts: {available_accounts}")
            return None
            
        recommendations = self.ensemble_predict(account_name)
        return self.explain_recommendations(recommendations, account_name)
        
    def run_demo(self):
        """Run complete demonstration"""
        print("WHITE SPACE IDENTIFICATION SYSTEM")
        print("=" * 50)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Train all models
        self.train_all_models()
        
        # Analyze all accounts
        print(f"ANALYZING ALL ACCOUNTS")
        print("=" * 50)
        
        for account in self.data['Account']:
            self.analyze_account(account)
            print("\n" + "-" * 60 + "\n")
        
        print("DEMO COMPLETED SUCCESSFULLY!")

# Usage
if __name__ == "__main__":
    # Create and run the system
    ws_system = WhiteSpaceIdentificationSystem()
    ws_system.run_demo()
    
    # Interactive analysis
    print("\nINTERACTIVE MODE")
    print("Available accounts:", list(ws_system.data['Account']))
    
    # Uncomment to analyze specific account:
    # ws_system.analyze_account('Apple')