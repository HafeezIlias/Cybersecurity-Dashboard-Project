#!/usr/bin/env python3
"""
Flask Backend for Cybersecurity Disaster Risk Management Dashboard
================================================================

This Flask application provides REST API endpoints for the cybersecurity
dashboard, integrating with our unified ensemble model for predictions.

Features:
- Unified ensemble model integration
- Comprehensive API endpoints
- CORS support for frontend integration
- Error handling and validation
- Performance monitoring

Author: Cybersecurity Risk Management Team
Date: 2024
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json
import logging
from datetime import datetime
import os
from werkzeug.exceptions import BadRequest
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global variables for models and data
df = None
ensemble_model = None
clustering_model = None
classification_model = None
regression_model = None
clustering_scaler = None
model_metadata = None

def load_models_and_data():
    """Load all models and data at startup"""
    global df, ensemble_model, clustering_model, classification_model, regression_model, clustering_scaler, model_metadata
    
    try:
        logger.info("Loading models and data...")
        
        # Load processed data
        df = pd.read_csv('data/processed_cyber_data.csv')
        logger.info(f"Loaded dataset with {len(df)} countries")
        
        # Load the UNIFIED ENSEMBLE MODEL (Primary model)
        with open('models/ensemble_model.pkl', 'rb') as f:
            ensemble_model = pickle.load(f)
        logger.info("✅ Unified Ensemble Model loaded")
        
        # Load individual models for additional insights
        with open('models/clustering_model.pkl', 'rb') as f:
            clustering_model, clustering_scaler = pickle.load(f)
        logger.info("✅ Clustering Model loaded")
        
        with open('models/classification_model.pkl', 'rb') as f:
            classification_model = pickle.load(f)
        logger.info("✅ Classification Model loaded")
        
        with open('models/regression_model.pkl', 'rb') as f:
            regression_model = pickle.load(f)
        logger.info("✅ Regression Model loaded")
        
        # Load model validation results
        with open('models/model_validation_results.pkl', 'rb') as f:
            model_metadata = pickle.load(f)
        logger.info("✅ Model metadata loaded")
        
        logger.info("🚀 All models and data loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

# Load models at startup
load_models_and_data()

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/', methods=['GET'])
def home():
    """API home endpoint with status information"""
    return jsonify({
        'message': 'Cybersecurity Risk Management Dashboard API',
        'version': '2.0',
        'status': 'active',
        'endpoints': {
            'countries': '/api/countries',
            'correlation': '/api/correlation',
            'regional_analysis': '/api/regional-analysis',
            'predict': '/api/predict',
            'feature_importance': '/api/feature-importance',
            'similar_countries': '/api/similar-countries',
            'recommendations': '/api/recommendations',
            'model_info': '/api/model-info'
        },
        'models_loaded': {
            'ensemble_model': ensemble_model is not None,
            'clustering_model': clustering_model is not None,
            'classification_model': classification_model is not None,
            'regression_model': regression_model is not None
        }
    })

@app.route('/api/countries', methods=['GET'])
def get_countries_data():
    """Return all countries data for visualization"""
    try:
        # Get query parameters for filtering
        region = request.args.get('region')
        risk_category = request.args.get('risk_category')
        
        data = df.copy()
        
        # Apply filters if provided
        if region:
            data = data[data['Region'] == region]
        if risk_category:
            data = data[data['Risk_Category'] == risk_category]
        
        # Convert to records and handle NaN values
        records = data.fillna('N/A').to_dict('records')
        
        return jsonify({
            'data': records,
            'total_countries': len(records),
            'filters_applied': {
                'region': region,
                'risk_category': risk_category
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_countries_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/correlation', methods=['GET'])
def get_correlation_data():
    """Return correlation matrix for heatmap"""
    try:
        numeric_cols = ['CEI', 'GCI', 'NCSI', 'DDL']
        correlation_matrix = df[numeric_cols].corr()
        
        return jsonify({
            'correlation_matrix': correlation_matrix.to_dict(),
            'metrics': numeric_cols,
            'description': {
                'CEI': 'Cybersecurity Exposure Index (lower is better)',
                'GCI': 'Global Cybersecurity Index (higher is better)',
                'NCSI': 'National Cyber Security Index (higher is better)',
                'DDL': 'Digital Development Level (higher is better)'
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_correlation_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/regional-analysis', methods=['GET'])
def get_regional_analysis():
    """Return regional statistics"""
    try:
        regional_stats = df.groupby('Region').agg({
            'CEI': ['mean', 'std', 'count'],
            'GCI': ['mean', 'std', 'count'],
            'NCSI': ['mean', 'std', 'count'],
            'DDL': ['mean', 'std', 'count'],
            'Risk_Score': ['mean', 'std', 'count']
        }).round(3)
        
        # Flatten column names
        regional_stats.columns = ['_'.join(col).strip() for col in regional_stats.columns]
        regional_stats = regional_stats.reset_index()
        
        # Risk category distribution by region
        risk_distribution = df.groupby(['Region', 'Risk_Category']).size().unstack(fill_value=0)
        
        return jsonify({
            'regional_statistics': regional_stats.fillna(0).to_dict('records'),
            'risk_distribution': risk_distribution.to_dict('index'),
            'total_regions': len(regional_stats)
        })
        
    except Exception as e:
        logger.error(f"Error in get_regional_analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_risk():
    """Predict risk category using UNIFIED ENSEMBLE MODEL"""
    try:
        data = request.get_json()
        
        # Validate input data
        required_fields = ['CEI', 'GCI', 'NCSI', 'DDL']
        for field in required_fields:
            if field not in data:
                raise BadRequest(f"Missing required field: {field}")
        
        # Prepare features
        features = np.array([[data['CEI'], data['GCI'], data['NCSI'], data['DDL']]])
        
        # 🚀 PRIMARY PREDICTION: Use Unified Ensemble Model
        ensemble_pred = ensemble_model.predict(features)[0]
        
        # Calculate ensemble confidence (for hard voting)
        individual_predictions = []
        for name, clf in ensemble_model.estimators:
            pred = clf.predict(features)[0]
            individual_predictions.append(pred)
        
        # Calculate agreement percentage as confidence
        pred_counts = {}
        for pred in individual_predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        ensemble_confidence = max(pred_counts.values()) / len(individual_predictions)
        
        # Additional predictions for comparison/analysis
        cluster_pred = clustering_model.predict(clustering_scaler.transform(features))[0]
        individual_rf_pred = classification_model.predict(features)[0]
        individual_rf_proba = classification_model.predict_proba(features)[0]
        
        # Regression prediction (GCI from other metrics)
        reg_features = features[:, [0, 2, 3]]  # CEI, NCSI, DDL
        gci_pred = regression_model.predict(reg_features)[0]
        
        # Risk score calculation
        cei_norm = 1 - (data['CEI'] - df['CEI'].min()) / (df['CEI'].max() - df['CEI'].min())
        gci_norm = (data['GCI'] - df['GCI'].min()) / (df['GCI'].max() - df['GCI'].min())
        ncsi_norm = (data['NCSI'] - df['NCSI'].min()) / (df['NCSI'].max() - df['NCSI'].min())
        ddl_norm = (data['DDL'] - df['DDL'].min()) / (df['DDL'].max() - df['DDL'].min())
        risk_score = (cei_norm + gci_norm + ncsi_norm + ddl_norm) / 4
        
        return jsonify({
            'primary_prediction': {
                'risk_category': ensemble_pred,
                'confidence': float(ensemble_confidence),
                'risk_score': float(risk_score),
                'model_type': 'Hard Voting Ensemble',
                'timestamp': datetime.now().isoformat()
            },
            'additional_insights': {
                'cluster': int(cluster_pred),
                'individual_rf_prediction': individual_rf_pred,
                'individual_rf_probabilities': {
                    cls: float(prob) for cls, prob in 
                    zip(classification_model.classes_, individual_rf_proba)
                },
                'predicted_gci': float(gci_pred),
                'actual_gci': data['GCI']
            },
            'model_info': {
                'ensemble_components': len(ensemble_model.estimators),
                'voting_strategy': ensemble_model.voting,
                'unified_deployment': True,
                'individual_votes': {
                    name: pred for name, pred in 
                    zip([name for name, _ in ensemble_model.estimators], individual_predictions)
                }
            },
            'input_validation': {
                'cei_range': [0, 1],
                'gci_range': [0, 100],
                'ncsi_range': [0, 100],
                'ddl_range': [0, 100],
                'input_within_range': all([
                    0 <= data['CEI'] <= 1,
                    0 <= data['GCI'] <= 100,
                    0 <= data['NCSI'] <= 100,
                    0 <= data['DDL'] <= 100
                ])
            }
        })
        
    except BadRequest as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in predict_risk: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    """Return feature importance from classification model"""
    try:
        features = ['CEI', 'GCI', 'NCSI', 'DDL']
        importance = classification_model.feature_importances_
        
        feature_data = [
            {
                'feature': feat, 
                'importance': float(imp),
                'percentage': float(imp * 100),
                'description': {
                    'CEI': 'Cybersecurity Exposure Index',
                    'GCI': 'Global Cybersecurity Index',
                    'NCSI': 'National Cyber Security Index',
                    'DDL': 'Digital Development Level'
                }.get(feat, feat)
            } 
            for feat, imp in zip(features, importance)
        ]
        
        # Sort by importance
        feature_data.sort(key=lambda x: x['importance'], reverse=True)
        
        return jsonify({
            'feature_importance': feature_data,
            'model_type': 'Random Forest',
            'total_features': len(features)
        })
        
    except Exception as e:
        logger.error(f"Error in get_feature_importance: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/similar-countries', methods=['POST'])
def get_similar_countries():
    """Find countries with similar cybersecurity profiles"""
    try:
        data = request.get_json()
        input_metrics = np.array([data['CEI'], data['GCI'], data['NCSI'], data['DDL']])
        
        # Calculate similarity scores using Euclidean distance
        similarities = []
        for _, country in df.iterrows():
            if pd.isna(country[['CEI', 'GCI', 'NCSI', 'DDL']]).any():
                continue
                
            country_metrics = np.array([country['CEI'], country['GCI'], country['NCSI'], country['DDL']])
            
            # Normalize the distance to similarity score
            distance = np.linalg.norm(input_metrics - country_metrics)
            similarity = 1 / (1 + distance)
            
            similarities.append({
                'country': country['Country'],
                'region': country['Region'],
                'similarity': float(similarity),
                'distance': float(distance),
                'risk_category': country['Risk_Category'],
                'risk_score': float(country['Risk_Score']),
                'metrics': {
                    'CEI': float(country['CEI']),
                    'GCI': float(country['GCI']),
                    'NCSI': float(country['NCSI']),
                    'DDL': float(country['DDL'])
                }
            })
        
        # Sort by similarity and return top 10
        similar_countries = sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:10]
        
        return jsonify({
            'similar_countries': similar_countries,
            'total_matches': len(similarities),
            'input_metrics': {
                'CEI': data['CEI'],
                'GCI': data['GCI'],
                'NCSI': data['NCSI'],
                'DDL': data['DDL']
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_similar_countries: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Generate improvement recommendations based on input metrics"""
    try:
        data = request.get_json()
        recommendations = []
        
        # CEI recommendations (lower is better)
        if data['CEI'] > 0.7:
            recommendations.append({
                'metric': 'CEI',
                'current_value': data['CEI'],
                'priority': 'High',
                'recommendation': 'Implement comprehensive cybersecurity framework and incident response plan',
                'impact': 'Significantly reduce exposure to cyber threats',
                'target_value': 0.4,
                'improvement_percentage': ((data['CEI'] - 0.4) / data['CEI']) * 100
            })
        elif data['CEI'] > 0.5:
            recommendations.append({
                'metric': 'CEI',
                'current_value': data['CEI'],
                'priority': 'Medium',
                'recommendation': 'Strengthen cybersecurity policies and employee training',
                'impact': 'Moderate reduction in cyber risk exposure',
                'target_value': 0.3,
                'improvement_percentage': ((data['CEI'] - 0.3) / data['CEI']) * 100
            })
        
        # GCI recommendations (higher is better)
        if data['GCI'] < 40:
            recommendations.append({
                'metric': 'GCI',
                'current_value': data['GCI'],
                'priority': 'High',
                'recommendation': 'Develop national cybersecurity strategy and enhance technical capabilities',
                'impact': 'Improve overall cybersecurity posture significantly',
                'target_value': 60,
                'improvement_percentage': ((60 - data['GCI']) / data['GCI']) * 100
            })
        elif data['GCI'] < 60:
            recommendations.append({
                'metric': 'GCI',
                'current_value': data['GCI'],
                'priority': 'Medium',
                'recommendation': 'Enhance cybersecurity cooperation and capacity building',
                'impact': 'Moderate improvement in cybersecurity capabilities',
                'target_value': 75,
                'improvement_percentage': ((75 - data['GCI']) / data['GCI']) * 100
            })
        
        # NCSI recommendations (higher is better)
        if data['NCSI'] < 40:
            recommendations.append({
                'metric': 'NCSI',
                'current_value': data['NCSI'],
                'priority': 'High',
                'recommendation': 'Establish cybersecurity governance and legal framework',
                'impact': 'Build foundational cybersecurity infrastructure',
                'target_value': 60,
                'improvement_percentage': ((60 - data['NCSI']) / data['NCSI']) * 100
            })
        
        # DDL recommendations (higher is better)
        if data['DDL'] < 40:
            recommendations.append({
                'metric': 'DDL',
                'current_value': data['DDL'],
                'priority': 'Medium',
                'recommendation': 'Invest in digital infrastructure and connectivity',
                'impact': 'Improve digital foundation for cybersecurity',
                'target_value': 55,
                'improvement_percentage': ((55 - data['DDL']) / data['DDL']) * 100
            })
        
        # Overall risk assessment
        risk_score = (1 - data['CEI'] + data['GCI']/100 + data['NCSI']/100 + data['DDL']/100) / 4
        
        if risk_score < 0.4:
            overall_recommendation = "Critical: Immediate comprehensive cybersecurity overhaul required"
        elif risk_score < 0.6:
            overall_recommendation = "Moderate: Focus on key areas for systematic improvement"
        else:
            overall_recommendation = "Good: Maintain current practices with continuous improvement"
        
        return jsonify({
            'recommendations': recommendations,
            'overall_assessment': {
                'risk_score': float(risk_score),
                'recommendation': overall_recommendation,
                'priority_areas': len([r for r in recommendations if r['priority'] == 'High'])
            },
            'total_recommendations': len(recommendations)
        })
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Return information about loaded models"""
    try:
        ensemble_info = {
            'type': type(ensemble_model).__name__,
            'voting_strategy': ensemble_model.voting,
            'n_estimators': len(ensemble_model.estimators),
            'estimator_names': [name for name, _ in ensemble_model.estimators],
            'classes': ensemble_model.classes_.tolist() if hasattr(ensemble_model, 'classes_') else []
        }
        
        return jsonify({
            'ensemble_model': ensemble_info,
            'model_performance': {
                'ensemble': model_metadata.get('ensemble', {}).get('best_results', {}),
                'classification': {
                    'test_accuracy': model_metadata.get('classification', {}).get('scores', [None, None, None])[2]
                },
                'clustering': {
                    'test_silhouette': model_metadata.get('clustering', {}).get('scores', [None, None, None])[2]
                },
                'regression': {
                    'test_r2': model_metadata.get('regression', {}).get('r2_scores', [None, None, None])[2]
                }
            },
            'data_info': {
                'total_countries': len(df),
                'features': ['CEI', 'GCI', 'NCSI', 'DDL'],
                'risk_categories': df['Risk_Category'].value_counts().to_dict(),
                'regions': df['Region'].value_counts().to_dict()
            }
        })
        
    except Exception as e:
        logger.error(f"Error in get_model_info: {str(e)}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(BadRequest)
def bad_request(error):
    return jsonify({'error': str(error)}), 400

# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 CYBERSECURITY RISK MANAGEMENT DASHBOARD API")
    print("=" * 60)
    print("✅ Unified Ensemble Model: Loaded")
    print("✅ Individual Models: Loaded")
    print("✅ Dataset: Loaded")
    print("✅ API Endpoints: Ready")
    print("-" * 60)
    print("🌐 Starting Flask server...")
    print("📡 API will be available at: http://localhost:5000")
    print("📚 API Documentation: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000) 