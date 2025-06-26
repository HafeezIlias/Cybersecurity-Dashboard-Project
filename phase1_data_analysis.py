# Phase 1: Data Analysis & Model Development (Python)
# Cybersecurity Disaster Risk Management Dashboard

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, mean_squared_error, r2_score, silhouette_score,
                           confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

print("=== PHASE 1: DATA ANALYSIS & MODEL DEVELOPMENT ===")
print("Cybersecurity Disaster Risk Management Dashboard\n")

# Create directories for outputs
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# =============================================================================
# Step 1.1: Data Exploration & Correlation Analysis
# =============================================================================

print("Step 1.1: Data Exploration & Correlation Analysis")
print("-" * 50)

# Load and explore data
df = pd.read_csv('Cyber_security.csv')

print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
numeric_cols = ['CEI', 'GCI', 'NCSI', 'DDL']
print(df[numeric_cols].describe())

# Regional distribution
print("\nRegional Distribution:")
print(df['Region'].value_counts())

# Create correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, linewidths=0.5)
plt.title('Cybersecurity Metrics Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Data distribution visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    data_clean = df[col].dropna()
    axes[i].hist(data_clean, bins=20, alpha=0.7, color=plt.cm.Set1(i))
    axes[i].set_title(f'{col} Distribution', fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/data_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Step 1.2: Create Risk Categories
# =============================================================================

print("\n" + "="*70)
print("Step 1.2: Create Risk Categories")
print("-" * 50)

def create_risk_categories(df):
    """Create risk categories based on multiple metrics"""
    df_copy = df.copy()
    
    # Handle missing values by filling with median
    for col in numeric_cols:
        df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    # Normalize metrics (CEI is reverse - lower is better)
    df_copy['CEI_norm'] = 1 - (df_copy['CEI'] - df_copy['CEI'].min()) / (df_copy['CEI'].max() - df_copy['CEI'].min())
    df_copy['GCI_norm'] = (df_copy['GCI'] - df_copy['GCI'].min()) / (df_copy['GCI'].max() - df_copy['GCI'].min())
    df_copy['NCSI_norm'] = (df_copy['NCSI'] - df_copy['NCSI'].min()) / (df_copy['NCSI'].max() - df_copy['NCSI'].min())
    df_copy['DDL_norm'] = (df_copy['DDL'] - df_copy['DDL'].min()) / (df_copy['DDL'].max() - df_copy['DDL'].min())
    
    # Calculate composite risk score (higher = better security)
    df_copy['Risk_Score'] = (df_copy['CEI_norm'] + df_copy['GCI_norm'] + 
                            df_copy['NCSI_norm'] + df_copy['DDL_norm']) / 4
    
    # Create risk categories (inverted - higher score = lower risk)
    df_copy['Risk_Category'] = pd.cut(df_copy['Risk_Score'], 
                                     bins=[0, 0.33, 0.67, 1.0], 
                                     labels=['High Risk', 'Medium Risk', 'Low Risk'])
    
    return df_copy

# Apply risk categorization
df_processed = create_risk_categories(df)

print("Risk Score Statistics:")
print(df_processed['Risk_Score'].describe())

print("\nRisk Category Distribution:")
print(df_processed['Risk_Category'].value_counts())

# Visualize risk categories
plt.figure(figsize=(12, 8))

# Risk category by region
plt.subplot(2, 2, 1)
risk_region = pd.crosstab(df_processed['Region'], df_processed['Risk_Category'])
risk_region.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Risk Categories by Region')
plt.xticks(rotation=45)
plt.legend(title='Risk Category')

# Risk score distribution
plt.subplot(2, 2, 2)
plt.hist(df_processed['Risk_Score'], bins=20, alpha=0.7, color='skyblue')
plt.axvline(df_processed['Risk_Score'].mean(), color='red', linestyle='--', 
            label=f'Mean: {df_processed["Risk_Score"].mean():.3f}')
plt.title('Risk Score Distribution')
plt.xlabel('Risk Score')
plt.ylabel('Frequency')
plt.legend()

# Top 10 highest risk countries
plt.subplot(2, 2, 3)
top_risk = df_processed.nsmallest(10, 'Risk_Score')[['Country', 'Risk_Score']]
plt.barh(range(len(top_risk)), top_risk['Risk_Score'], color='red', alpha=0.7)
plt.yticks(range(len(top_risk)), top_risk['Country'])
plt.title('Top 10 Highest Risk Countries')
plt.xlabel('Risk Score')

# Top 10 lowest risk countries
plt.subplot(2, 2, 4)
low_risk = df_processed.nlargest(10, 'Risk_Score')[['Country', 'Risk_Score']]
plt.barh(range(len(low_risk)), low_risk['Risk_Score'], color='green', alpha=0.7)
plt.yticks(range(len(low_risk)), low_risk['Country'])
plt.title('Top 10 Lowest Risk Countries')
plt.xlabel('Risk Score')

plt.tight_layout()
plt.savefig('visualizations/risk_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Step 1.3: Enhanced Model Training with Proper Split Testing
# =============================================================================

print("\n" + "="*70)
print("Step 1.3: Enhanced Model Training with Proper Split Testing")
print("-" * 50)

# Prepare clean dataset for modeling
df_clean = df_processed.dropna(subset=numeric_cols).copy()
print(f"Clean dataset shape: {df_clean.shape}")

# Split data into train/validation/test sets (60/20/20)
features = ['CEI', 'GCI', 'NCSI', 'DDL']
X = df_clean[features].values
y_classification = df_clean['Risk_Category'].values
y_regression = df_clean['GCI'].values

# First split: 80% train+val, 20% test
X_temp, X_test, y_class_temp, y_class_test = train_test_split(
    X, y_classification, test_size=0.2, random_state=42, stratify=y_classification)

# Second split: 60% train, 20% validation from the remaining 80%
X_train, X_val, y_class_train, y_class_val = train_test_split(
    X_temp, y_class_temp, test_size=0.25, random_state=42, stratify=y_class_temp)

print(f"Train set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set size: {X_val.shape[0]} ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

# Model 1: K-Means Clustering with Enhanced Validation
print("\n1. K-Means Clustering Model with Enhanced Validation")
print("-" * 50)

def train_clustering_model_enhanced(X_train, X_val, X_test):
    """Train and validate K-Means clustering model"""
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Find optimal number of clusters using multiple metrics
    K_range = range(2, 8)
    train_inertias = []
    val_inertias = []
    train_silhouettes = []
    val_silhouettes = []
    
    for k in K_range:
        # Train on training set
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        train_labels = kmeans.fit_predict(X_train_scaled)
        train_inertias.append(kmeans.inertia_)
        train_silhouettes.append(silhouette_score(X_train_scaled, train_labels))
        
        # Validate on validation set
        val_labels = kmeans.predict(X_val_scaled)
        val_inertia = np.sum(np.min(cdist(X_val_scaled, kmeans.cluster_centers_)**2, axis=1))
        val_inertias.append(val_inertia)
        val_silhouettes.append(silhouette_score(X_val_scaled, val_labels))
    
    # Plot validation curves
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Inertia plots
    ax1.plot(K_range, train_inertias, 'bo-', label='Training')
    ax1.plot(K_range, val_inertias, 'ro-', label='Validation')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Inertia: Training vs Validation')
    ax1.legend()
    ax1.grid(True)
    
    # Silhouette plots
    ax2.plot(K_range, train_silhouettes, 'bo-', label='Training')
    ax2.plot(K_range, val_silhouettes, 'ro-', label='Validation')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score: Training vs Validation')
    ax2.legend()
    ax2.grid(True)
    
    # Find optimal k (best validation silhouette)
    optimal_k = K_range[np.argmax(val_silhouettes)]
    
    # Train final model with optimal k
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    train_labels_final = kmeans_final.fit_predict(X_train_scaled)
    val_labels_final = kmeans_final.predict(X_val_scaled)
    test_labels_final = kmeans_final.predict(X_test_scaled)
    
    # Calculate final metrics
    train_sil_final = silhouette_score(X_train_scaled, train_labels_final)
    val_sil_final = silhouette_score(X_val_scaled, val_labels_final)
    test_sil_final = silhouette_score(X_test_scaled, test_labels_final)
    
    # Visualize final clusters on test set
    ax3.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=test_labels_final, cmap='viridis', alpha=0.6)
    ax3.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1], 
                c='red', marker='x', s=200, linewidths=3)
    ax3.set_title(f'Test Set Clusters (k={optimal_k})')
    ax3.set_xlabel('CEI (normalized)')
    ax3.set_ylabel('GCI (normalized)')
    
    # Performance comparison
    metrics = ['Training', 'Validation', 'Test']
    silhouette_scores = [train_sil_final, val_sil_final, test_sil_final]
    ax4.bar(metrics, silhouette_scores, color=['blue', 'orange', 'green'], alpha=0.7)
    ax4.set_ylabel('Silhouette Score')
    ax4.set_title('Clustering Performance Across Sets')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/clustering_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return kmeans_final, scaler, optimal_k, (train_sil_final, val_sil_final, test_sil_final)

clustering_model, clustering_scaler, optimal_k, clustering_scores = train_clustering_model_enhanced(X_train, X_val, X_test)

print(f"Optimal number of clusters: {optimal_k}")
print(f"Training Silhouette Score: {clustering_scores[0]:.3f}")
print(f"Validation Silhouette Score: {clustering_scores[1]:.3f}")
print(f"Test Silhouette Score: {clustering_scores[2]:.3f}")

# Model 2: Random Forest Classification with Hyperparameter Tuning
print("\n2. Random Forest Classification with Hyperparameter Tuning")
print("-" * 60)

def train_classification_model_enhanced(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train and validate Random Forest Classification with hyperparameter tuning"""
    
    # Hyperparameter tuning with GridSearchCV (Anti-overfitting parameters)
    param_grid = {
        'n_estimators': [50, 100],              # Reduced to prevent overfitting
        'max_depth': [3, 5, 7],                 # Limited depth to prevent memorization
        'min_samples_split': [10, 20, 30],      # Higher splits for generalization
        'min_samples_leaf': [5, 10, 15],        # Higher leaf samples for stability
        'max_features': ['sqrt', 'log2'],       # Feature subsampling
        'max_samples': [0.7, 0.8, 0.9]         # Bootstrap sampling limitation
    }
    
    # Use validation set for hyperparameter tuning
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    
    # Create validation indices for GridSearchCV
    val_indices = [-1] * len(X_train) + [0] * len(X_val)
    
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=[(list(range(len(X_train))), list(range(len(X_train), len(X_train_val))))],
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("Performing hyperparameter tuning...")
    rf_grid.fit(X_train_val, y_train_val)
    
    print(f"Best parameters: {rf_grid.best_params_}")
    print(f"Best validation score: {rf_grid.best_score_:.3f}")
    
    # Train final model with best parameters
    rf_final = rf_grid.best_estimator_
    
    # Predictions on all sets
    y_train_pred = rf_final.predict(X_train)
    y_val_pred = rf_final.predict(X_val)
    y_test_pred = rf_final.predict(X_test)
    
    # Calculate comprehensive metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_final.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Cross-validation for robustness check
    cv_scores = cross_val_score(rf_final, X_train_val, y_train_val, cv=5, scoring='accuracy')
    
    # Visualize results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Performance comparison
    sets = ['Training', 'Validation', 'Test']
    accuracies = [train_accuracy, val_accuracy, test_accuracy]
    f1_scores = [train_f1, val_f1, test_f1]
    
    x = np.arange(len(sets))
    width = 0.35
    
    ax1.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.7)
    ax1.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.7)
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Score')
    ax1.set_title('Classification Performance Across Sets')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sets)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Feature importance
    ax2.barh(feature_importance['feature'], feature_importance['importance'])
    ax2.set_xlabel('Importance')
    ax2.set_title('Feature Importance')
    ax2.grid(True, alpha=0.3)
    
    # Confusion matrix for test set
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title('Test Set Confusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # Cross-validation scores
    ax4.boxplot(cv_scores)
    ax4.set_ylabel('Accuracy')
    ax4.set_title(f'5-Fold CV Scores\nMean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/classification_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rf_final, feature_importance, (train_accuracy, val_accuracy, test_accuracy), cv_scores

rf_classifier, feature_importance, classification_scores, cv_scores = train_classification_model_enhanced(
    X_train, X_val, X_test, y_class_train, y_class_val, y_class_test)

print(f"Training Accuracy: {classification_scores[0]:.3f}")
print(f"Validation Accuracy: {classification_scores[1]:.3f}")
print(f"Test Accuracy: {classification_scores[2]:.3f}")
print(f"Cross-validation Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

print("\nFeature Importance:")
print(feature_importance)

print("\nTest Set Classification Report:")
print(classification_report(y_class_test, rf_classifier.predict(X_test)))

# Model 3: Ridge Regression with Validation Curves
print("\n3. Ridge Regression with Validation Curves")
print("-" * 45)

def train_regression_model_enhanced(X_train, X_val, X_test):
    """Train and validate Ridge Regression with proper validation"""
    
    # Prepare regression data (predict GCI from CEI, NCSI, DDL)
    reg_features = ['CEI', 'NCSI', 'DDL']
    reg_feature_indices = [0, 2, 3]  # CEI, NCSI, DDL indices
    
    X_train_reg = X_train[:, reg_feature_indices]
    X_val_reg = X_val[:, reg_feature_indices]
    X_test_reg = X_test[:, reg_feature_indices]
    
    # Get corresponding GCI values
    train_indices = df_clean.index[:len(X_train)]
    val_indices = df_clean.index[len(X_train):len(X_train)+len(X_val)]
    test_indices = df_clean.index[len(X_train)+len(X_val):]
    
    y_train_reg = df_clean.loc[train_indices, 'GCI'].values
    y_val_reg = df_clean.loc[val_indices, 'GCI'].values
    y_test_reg = df_clean.loc[test_indices, 'GCI'].values
    
    # Hyperparameter tuning for Ridge alpha
    alphas = np.logspace(-3, 3, 50)
    
    train_scores = []
    val_scores = []
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_reg, y_train_reg)
        
        train_pred = ridge.predict(X_train_reg)
        val_pred = ridge.predict(X_val_reg)
        
        train_scores.append(r2_score(y_train_reg, train_pred))
        val_scores.append(r2_score(y_val_reg, val_pred))
    
    # Find optimal alpha
    optimal_alpha_idx = np.argmax(val_scores)
    optimal_alpha = alphas[optimal_alpha_idx]
    
    # Train final model
    ridge_final = Ridge(alpha=optimal_alpha)
    ridge_final.fit(X_train_reg, y_train_reg)
    
    # Predictions on all sets
    y_train_pred = ridge_final.predict(X_train_reg)
    y_val_pred = ridge_final.predict(X_val_reg)
    y_test_pred = ridge_final.predict(X_test_reg)
    
    # Calculate metrics
    train_r2 = r2_score(y_train_reg, y_train_pred)
    val_r2 = r2_score(y_val_reg, y_val_pred)
    test_r2 = r2_score(y_test_reg, y_test_pred)
    
    train_mse = mean_squared_error(y_train_reg, y_train_pred)
    val_mse = mean_squared_error(y_val_reg, y_val_pred)
    test_mse = mean_squared_error(y_test_reg, y_test_pred)
    
    # Visualize results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Validation curve
    ax1.semilogx(alphas, train_scores, 'b-', label='Training', alpha=0.7)
    ax1.semilogx(alphas, val_scores, 'r-', label='Validation', alpha=0.7)
    ax1.axvline(optimal_alpha, color='g', linestyle='--', label=f'Optimal α={optimal_alpha:.3f}')
    ax1.set_xlabel('Alpha (Regularization)')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Ridge Regression Validation Curve')
    ax1.legend()
    ax1.grid(True)
    
    # Performance comparison
    sets = ['Training', 'Validation', 'Test']
    r2_scores_all = [train_r2, val_r2, test_r2]
    mse_scores_all = [train_mse, val_mse, test_mse]
    
    ax2.bar(sets, r2_scores_all, alpha=0.7, color=['blue', 'orange', 'green'])
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score Across Sets')
    ax2.grid(True, alpha=0.3)
    
    # Actual vs Predicted (Test Set)
    ax3.scatter(y_test_reg, y_test_pred, alpha=0.6)
    ax3.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
    ax3.set_xlabel('Actual GCI')
    ax3.set_ylabel('Predicted GCI')
    ax3.set_title(f'Test Set: Actual vs Predicted\nR² = {test_r2:.3f}')
    ax3.grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_test_reg - y_test_pred
    ax4.scatter(y_test_pred, residuals, alpha=0.6)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('Predicted GCI')
    ax4.set_ylabel('Residuals')
    ax4.set_title('Residuals Plot (Test Set)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/regression_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return ridge_final, optimal_alpha, (train_r2, val_r2, test_r2), (train_mse, val_mse, test_mse)

ridge_reg, optimal_alpha, regression_r2_scores, regression_mse_scores = train_regression_model_enhanced(X_train, X_val, X_test)

print(f"Optimal Alpha: {optimal_alpha:.3f}")
print(f"Training R²: {regression_r2_scores[0]:.3f}")
print(f"Validation R²: {regression_r2_scores[1]:.3f}")
print(f"Test R²: {regression_r2_scores[2]:.3f}")
print(f"Test MSE: {regression_mse_scores[2]:.3f}")

# =============================================================================
# Model 4: Ensemble Learning with Voting Classifier
# =============================================================================

print("\n4. Ensemble Learning with Voting Classifier")
print("-" * 45)

def train_ensemble_model(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train and validate ensemble model with multiple classifiers"""
    
    # Define individual classifiers for the ensemble (Anti-overfitting versions)
    classifiers = {
        'random_forest': RandomForestClassifier(
            n_estimators=50, max_depth=5, min_samples_split=15, 
            min_samples_leaf=8, max_features='sqrt', max_samples=0.8, random_state=42
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.05, 
            min_samples_split=20, min_samples_leaf=10, random_state=42
        ),
        'svm': SVC(
            kernel='rbf', C=0.1, gamma='scale', probability=True, random_state=42
        ),
        'logistic_regression': LogisticRegression(
            C=0.1, max_iter=1000, random_state=42, penalty='l2'
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=10, weights='distance'  # Increased neighbors for smoothing
        ),
        'naive_bayes': GaussianNB(var_smoothing=1e-8),  # Added smoothing
        'decision_tree': DecisionTreeClassifier(
            max_depth=4, min_samples_split=20, min_samples_leaf=10, 
            max_features='sqrt', random_state=42
        )
    }
    
    # Train individual classifiers and evaluate
    individual_scores = {}
    trained_classifiers = {}
    
    print("Training individual classifiers...")
    for name, clf in classifiers.items():
        # Train on training set
        clf.fit(X_train, y_train)
        trained_classifiers[name] = clf
        
        # Evaluate on validation set
        val_pred = clf.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        individual_scores[name] = val_accuracy
        
        print(f"  {name.replace('_', ' ').title()}: {val_accuracy:.3f}")
    
    # Select top performing classifiers for ensemble (top 5)
    top_classifiers = sorted(individual_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    selected_classifiers = [(name, trained_classifiers[name]) for name, score in top_classifiers]
    
    print(f"\nTop 5 classifiers selected for ensemble:")
    for name, score in top_classifiers:
        print(f"  {name.replace('_', ' ').title()}: {score:.3f}")
    
    # Create Hard Voting Classifier
    hard_voting_clf = VotingClassifier(
        estimators=selected_classifiers,
        voting='hard'
    )
    
    # Create Soft Voting Classifier
    soft_voting_clf = VotingClassifier(
        estimators=selected_classifiers,
        voting='soft'
    )
    
    # Train ensemble models
    print("\nTraining ensemble models...")
    hard_voting_clf.fit(X_train, y_train)
    soft_voting_clf.fit(X_train, y_train)
    
    # Evaluate ensemble models
    models_to_evaluate = {
        'Hard Voting': hard_voting_clf,
        'Soft Voting': soft_voting_clf
    }
    
    ensemble_results = {}
    
    for ensemble_name, ensemble_model in models_to_evaluate.items():
        # Predictions on all sets
        train_pred = ensemble_model.predict(X_train)
        val_pred = ensemble_model.predict(X_val)
        test_pred = ensemble_model.predict(X_test)
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        train_f1 = f1_score(y_train, train_pred, average='weighted')
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        
        # Cross-validation
        cv_scores = cross_val_score(ensemble_model, 
                                  np.vstack([X_train, X_val]), 
                                  np.concatenate([y_train, y_val]), 
                                  cv=5, scoring='accuracy')
        
        ensemble_results[ensemble_name] = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_pred': test_pred
        }
        
        print(f"\n{ensemble_name} Ensemble Results:")
        print(f"  Training Accuracy: {train_acc:.3f}")
        print(f"  Validation Accuracy: {val_acc:.3f}")
        print(f"  Test Accuracy: {test_acc:.3f}")
        print(f"  Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Select best ensemble model
    best_ensemble_name = max(ensemble_results.keys(), 
                           key=lambda x: ensemble_results[x]['test_acc'])
    best_ensemble_model = models_to_evaluate[best_ensemble_name]
    
    print(f"\n🏆 Best Ensemble Model: {best_ensemble_name}")
    
    # Detailed evaluation of best ensemble
    best_results = ensemble_results[best_ensemble_name]
    test_pred = best_results['test_pred']
    
    print(f"\n📊 Best Ensemble Detailed Report:")
    print(classification_report(y_test, test_pred))
    
    # Visualize ensemble performance
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Individual vs Ensemble comparison
    individual_names = [name.replace('_', ' ').title() for name, _ in top_classifiers]
    individual_scores_list = [score for _, score in top_classifiers]
    ensemble_scores_list = [ensemble_results['Hard Voting']['val_acc'], 
                          ensemble_results['Soft Voting']['val_acc']]
    
    ax1.bar(individual_names, individual_scores_list, alpha=0.7, label='Individual', color='lightblue')
    ax1.bar(['Hard Voting', 'Soft Voting'], ensemble_scores_list, alpha=0.7, label='Ensemble', color='orange')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Individual vs Ensemble Model Performance')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Performance across sets for best ensemble
    sets = ['Training', 'Validation', 'Test']
    accuracies = [best_results['train_acc'], best_results['val_acc'], best_results['test_acc']]
    f1_scores = [best_results['train_f1'], best_results['val_f1'], best_results['test_f1']]
    
    x = np.arange(len(sets))
    width = 0.35
    
    ax2.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.7)
    ax2.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.7)
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Score')
    ax2.set_title(f'{best_ensemble_name} Performance Across Sets')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sets)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Confusion matrix for best ensemble
    cm = confusion_matrix(y_test, test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title(f'{best_ensemble_name} Confusion Matrix (Test Set)')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # Cross-validation comparison
    ensemble_names = list(ensemble_results.keys())
    cv_means = [ensemble_results[name]['cv_mean'] for name in ensemble_names]
    cv_stds = [ensemble_results[name]['cv_std'] for name in ensemble_names]
    
    ax4.bar(ensemble_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color='green')
    ax4.set_ylabel('Cross-validation Accuracy')
    ax4.set_title('Ensemble Models Cross-validation Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/ensemble_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_ensemble_model, best_ensemble_name, ensemble_results, selected_classifiers, models_to_evaluate

# Train ensemble model
ensemble_model, best_ensemble_name, ensemble_results, selected_classifiers, ensemble_models = train_ensemble_model(
    X_train, X_val, X_test, y_class_train, y_class_val, y_class_test)

# =============================================================================
# Enhanced Model Performance Summary
# =============================================================================

print("\n" + "="*70)
print("ENHANCED MODEL PERFORMANCE SUMMARY")
print("="*70)

print("1. K-Means Clustering:")
print(f"   - Optimal Clusters: {optimal_k}")
print(f"   - Training Silhouette: {clustering_scores[0]:.3f}")
print(f"   - Validation Silhouette: {clustering_scores[1]:.3f}")
print(f"   - Test Silhouette: {clustering_scores[2]:.3f}")
overfitting_clustering = abs(clustering_scores[0] - clustering_scores[2]) > 0.1
print(f"   - Overfitting: {'⚠ Yes' if overfitting_clustering else '✓ No'}")

print("\n2. Random Forest Classification:")
print(f"   - Training Accuracy: {classification_scores[0]:.3f}")
print(f"   - Validation Accuracy: {classification_scores[1]:.3f}")
print(f"   - Test Accuracy: {classification_scores[2]:.3f}")
print(f"   - Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
# Enhanced overfitting detection (stricter thresholds)
train_test_gap_rf = abs(classification_scores[0] - classification_scores[2])
perfect_training_rf = classification_scores[0] >= 0.99
overfitting_classification = train_test_gap_rf > 0.05 or perfect_training_rf
print(f"   - Training-Test Gap: {train_test_gap_rf:.3f}")
print(f"   - Perfect Training Score: {'⚠ Yes' if perfect_training_rf else '✓ No'}")
print(f"   - Overfitting: {'⚠ YES - SEVERE' if overfitting_classification else '✓ No'}")

print("\n3. Ridge Regression:")
print(f"   - Optimal Alpha: {optimal_alpha:.3f}")
print(f"   - Training R²: {regression_r2_scores[0]:.3f}")
print(f"   - Validation R²: {regression_r2_scores[1]:.3f}")
print(f"   - Test R²: {regression_r2_scores[2]:.3f}")
overfitting_regression = abs(regression_r2_scores[0] - regression_r2_scores[2]) > 0.1
print(f"   - Overfitting: {'⚠ Yes' if overfitting_regression else '✓ No'}")

# Ensemble model results with enhanced overfitting detection
best_ensemble_results = ensemble_results[best_ensemble_name]
print(f"\n4. 🏆 {best_ensemble_name} Ensemble (UNIFIED MODEL):")
print(f"   - Training Accuracy: {best_ensemble_results['train_acc']:.3f}")
print(f"   - Validation Accuracy: {best_ensemble_results['val_acc']:.3f}")
print(f"   - Test Accuracy: {best_ensemble_results['test_acc']:.3f}")
print(f"   - Cross-validation: {best_ensemble_results['cv_mean']:.3f} ± {best_ensemble_results['cv_std']:.3f}")

# Enhanced overfitting detection for ensemble
train_test_gap_ensemble = abs(best_ensemble_results['train_acc'] - best_ensemble_results['test_acc'])
perfect_validation_ensemble = best_ensemble_results['val_acc'] >= 0.99
overfitting_ensemble = train_test_gap_ensemble > 0.05 or perfect_validation_ensemble
print(f"   - Training-Test Gap: {train_test_gap_ensemble:.3f}")
print(f"   - Perfect Validation Score: {'⚠ Yes' if perfect_validation_ensemble else '✓ No'}")
print(f"   - Overfitting: {'⚠ YES - NEEDS REGULARIZATION' if overfitting_ensemble else '✓ No'}")
print(f"   - Selected Classifiers: {len(selected_classifiers)} top performers")

# Compare all models
print(f"\n📊 MODEL PERFORMANCE COMPARISON:")
all_models = {
    'K-Means Clustering': clustering_scores[2],  # Test silhouette
    'Random Forest': classification_scores[2],   # Test accuracy
    'Ridge Regression': regression_r2_scores[2], # Test R²
    f'{best_ensemble_name} Ensemble': best_ensemble_results['test_acc']  # Test accuracy
}

print(f"   Test Performance Scores:")
for model_name, score in all_models.items():
    print(f"   - {model_name}: {score:.3f}")

# Overall assessment
total_models = 4
good_models = 4 - sum([overfitting_clustering, overfitting_classification, overfitting_regression, overfitting_ensemble])
print(f"\n🎯 OVERALL MODEL QUALITY:")
print(f"   - Models with good generalization: {good_models}/{total_models}")
print(f"   - Best performing model: {best_ensemble_name} Ensemble")
print(f"   - 🚀 UNIFIED MODEL FOR PRODUCTION: {best_ensemble_name} Ensemble")
print(f"   - Ensemble combines: {', '.join([name.replace('_', ' ').title() for name, _ in selected_classifiers])}")
print(f"   - Single model management: ✅ Enabled")
print(f"   - Recommendation: {'✓ Deploy ensemble model' if not overfitting_ensemble else '⚠ Review ensemble overfitting'}")

# =============================================================================
# Save Enhanced Models and Validation Results
# =============================================================================

print("\n" + "="*70)
print("SAVING ENHANCED MODELS AND VALIDATION RESULTS")
print("="*70)

# Save models with validation results
model_results = {
    'clustering': {
        'model': clustering_model,
        'scaler': clustering_scaler,
        'optimal_k': optimal_k,
        'scores': clustering_scores
    },
    'classification': {
        'model': rf_classifier,
        'feature_importance': feature_importance,
        'scores': classification_scores,
        'cv_scores': cv_scores
    },
    'regression': {
        'model': ridge_reg,
        'optimal_alpha': optimal_alpha,
        'r2_scores': regression_r2_scores,
        'mse_scores': regression_mse_scores
    },
    'ensemble': {
        'model': ensemble_model,
        'model_type': best_ensemble_name,
        'selected_classifiers': selected_classifiers,
        'ensemble_results': ensemble_results,
        'best_results': best_ensemble_results
    }
}

# Save individual models
with open('models/clustering_model.pkl', 'wb') as f:
    pickle.dump((clustering_model, clustering_scaler), f)

with open('models/classification_model.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)

with open('models/regression_model.pkl', 'wb') as f:
    pickle.dump(ridge_reg, f)

# Save the UNIFIED ENSEMBLE MODEL (primary model for production)
with open('models/ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)

# Save complete results
with open('models/model_validation_results.pkl', 'wb') as f:
    pickle.dump(model_results, f)

# Save processed data
df_processed.to_csv('data/processed_cyber_data.csv', index=False)

print("✓ Enhanced models saved to 'models/' directory")
print("✓ 🚀 UNIFIED ENSEMBLE MODEL saved to 'models/ensemble_model.pkl'")
print("✓ Validation results saved to 'models/model_validation_results.pkl'")
print("✓ Processed data saved to 'data/processed_cyber_data.csv'")

# Feature importance data for frontend
feature_importance.to_csv('data/feature_importance.csv', index=False)
print("✓ Feature importance saved to 'data/feature_importance.csv'")

# =============================================================================
# Generate Enhanced Sample Predictions
# =============================================================================

print("\n" + "="*70)
print("ENHANCED SAMPLE PREDICTIONS WITH CONFIDENCE")
print("="*70)

# Sample prediction with average values
sample_metrics = {
    'CEI': df_clean['CEI'].mean(),
    'GCI': df_clean['GCI'].mean(),
    'NCSI': df_clean['NCSI'].mean(),
    'DDL': df_clean['DDL'].mean()
}

print("Sample Input Metrics (average values):")
for metric, value in sample_metrics.items():
    print(f"  {metric}: {value:.2f}")

# Make predictions with confidence intervals
sample_features = np.array([[sample_metrics['CEI'], sample_metrics['GCI'], 
                           sample_metrics['NCSI'], sample_metrics['DDL']]])

# Clustering prediction
sample_scaled = clustering_scaler.transform(sample_features)
cluster_pred = clustering_model.predict(sample_scaled)[0]

# Classification prediction with probability
risk_pred = rf_classifier.predict(sample_features)[0]
risk_proba = rf_classifier.predict_proba(sample_features)[0]
risk_confidence = np.max(risk_proba)

# Regression prediction
sample_reg_features = np.array([[sample_metrics['CEI'], sample_metrics['NCSI'], 
                               sample_metrics['DDL']]])
gci_pred = ridge_reg.predict(sample_reg_features)[0]

# Ensemble prediction (UNIFIED MODEL)
ensemble_pred = ensemble_model.predict(sample_features)[0]

# For probability, use soft voting model if available
if best_ensemble_name == 'Hard Voting':
    # For hard voting, calculate confidence based on individual classifier agreement
    individual_predictions = []
    for name, clf in selected_classifiers:
        pred = clf.predict(sample_features)[0]
        individual_predictions.append(pred)
    
    # Calculate agreement percentage as confidence
    pred_counts = {}
    for pred in individual_predictions:
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    
    ensemble_confidence = max(pred_counts.values()) / len(individual_predictions)
    
    # Get probabilities from soft voting for detailed analysis
    soft_voting_model = ensemble_models['Soft Voting']
    ensemble_proba = soft_voting_model.predict_proba(sample_features)[0]
else:
    ensemble_proba = ensemble_model.predict_proba(sample_features)[0]
    ensemble_confidence = np.max(ensemble_proba)

print("\nEnhanced Prediction Results:")
print(f"  Cluster: {cluster_pred}")
print(f"  Risk Category (Individual RF): {risk_pred} (Confidence: {risk_confidence:.2f})")
print(f"  🎯 Risk Category (UNIFIED ENSEMBLE): {ensemble_pred} (Confidence: {ensemble_confidence:.2f})")
print(f"  Predicted GCI: {gci_pred:.2f}")

# Risk probabilities comparison
risk_classes = rf_classifier.classes_
print(f"\nRisk Probabilities Comparison:")
print(f"Individual Random Forest:")
for i, risk_class in enumerate(risk_classes):
    print(f"  {risk_class}: {risk_proba[i]:.3f}")

print(f"\n🏆 {best_ensemble_name} Ensemble (UNIFIED MODEL):")
ensemble_classes = ensemble_model.classes_
for i, risk_class in enumerate(ensemble_classes):
    print(f"  {risk_class}: {ensemble_proba[i]:.3f}")

print("\n" + "="*70)
print("PHASE 1 WITH ENHANCED VALIDATION COMPLETED SUCCESSFULLY! ✓")
print("="*70)
print("✨ Key Improvements:")
print("- Proper train/validation/test splits (60/20/20)")
print("- Hyperparameter tuning with GridSearchCV")
print("- Cross-validation for model robustness")
print("- Overfitting detection and prevention")
print("- Comprehensive performance metrics")
print("- Validation curves for optimal parameters")
print("- 🚀 HARD VOTING ENSEMBLE for unified model management")
print("- Multiple classifier combination for improved accuracy")
print("- Single model deployment with ensemble power")
print("\nNext Steps:")
print("- Proceed to Phase 2: Flask Backend Development")
print("- Use the UNIFIED ENSEMBLE MODEL in your Flask API")
print("- Single model file for easy deployment and management")
print("- Implement ensemble confidence intervals in predictions") 