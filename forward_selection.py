import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from copy import deepcopy

def forward_selection_cv(X, y, K=5, early_stop=True, min_improvement=1.0):
    p = X.shape[1]
    remaining = list(X.columns)
    included = []
    all_models = []
    kf = KFold(n_splits=K, shuffle=True, random_state=1)
    
    # 1. Null model (intercept only)
    null_cv_mse = np.mean([np.mean((y.iloc[test_index] - np.mean(y.iloc[train_index]))**2) 
                          for train_index, test_index in kf.split(X)])
    all_models.append({
        'features': [],
        'CV_MSE': null_cv_mse,
        'model': None
    })
    
    # 2. Forward selection loop
    for k in range(p):
        cv_mse_scores = []
        
        # 2a. Test each remaining feature
        for feature in remaining:
            temp_features = included + [feature]
            X_temp = X[temp_features]
            
            # Calculate cross-validated MSE
            mse = -np.mean(cross_val_score(LinearRegression(), X_temp, y, 
                                         scoring='neg_mean_squared_error', cv=kf))
            model = LinearRegression().fit(X_temp, y)  # Full model for storage
            cv_mse_scores.append((feature, mse, deepcopy(model)))
        
        # 2b. Choose the best feature
        if cv_mse_scores:
            best_feature, best_cv_mse, best_model = min(cv_mse_scores, key=lambda x: x[1])
            prev_best_mse = min([m['CV_MSE'] for m in all_models])
            improvement = prev_best_mse - best_cv_mse
            
            # Early stopping check BEFORE adding feature
            if early_stop and k > 0 and improvement < min_improvement:
                print(f"Stopping: Adding '{best_feature}' would only improve MSE by {improvement:.2f} (< {min_improvement})")
                break
                
            # Only add if improvement is sufficient
            included.append(best_feature)
            remaining.remove(best_feature)
            
            model_info = {
                'features': deepcopy(included),
                'CV_MSE': best_cv_mse,
                'model': deepcopy(best_model)
            }
            all_models.append(model_info)
            print(f"Step {k+1}: Added '{best_feature}' | CV MSE = {best_cv_mse:.2f} (Δ = {improvement:.2f})")
    
    # 3. Select the single best model from all models
    best_model_idx = np.argmin([m['CV_MSE'] for m in all_models])
    best_model = all_models[best_model_idx]
    
    return best_model, all_models

# ----------------------------
# Load and Preprocess Data
# ----------------------------
df = pd.read_csv('Data/Podcast.csv')
df = pd.get_dummies(df, drop_first=True)
X = df.drop(columns=['Listening_Time_minutes'])
y = df['Listening_Time_minutes']

# ----------------------------
# Run Improved Forward Selection
# ----------------------------
print("=== Improved Forward Selection with CV ===")
best_model, all_models = forward_selection_cv(X, y, K=5, early_stop=True, min_improvement = 0.001)

# ----------------------------
# Final Results
# ----------------------------
print("\n=== Best Model ===")
print(f"Selected Features ({len(best_model['features'])}): {best_model['features']}")
print(f"CV MSE: {best_model['CV_MSE']:.2f}")
print(f"\nModel progression (CV MSE):")
for i, model in enumerate(all_models):
    print(f"Step {i}: {model['features']} | CV MSE = {model['CV_MSE']:.2f}")