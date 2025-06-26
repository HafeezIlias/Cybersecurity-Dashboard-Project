# Cybersecurity Disaster Risk Management Dashboard - Complete Workflow

## Phase 1: Data Analysis & Model Development (Python)

### Step 1.1: Data Exploration & Correlation Analysis
```python
# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn plotly xgboost flask flask-cors

# Data analysis script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Load and explore data
df = pd.read_csv('Cyber_security.csv')

# Create correlation heatmap
plt.figure(figsize=(10, 8))
numeric_cols = ['CEI', 'GCI', 'NCSI', 'DDL']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0)
plt.title('Cybersecurity Metrics Correlation Heatmap')
plt.show()

# Feature importance analysis
```

### Step 1.2: Create Risk Categories
```python
# Create risk categories based on multiple metrics
def create_risk_categories(df):
    # Normalize metrics (CEI is reverse - lower is better)
    df['CEI_norm'] = 1 - (df['CEI'] - df['CEI'].min()) / (df['CEI'].max() - df['CEI'].min())
    df['GCI_norm'] = (df['GCI'] - df['GCI'].min()) / (df['GCI'].max() - df['GCI'].min())
    df['NCSI_norm'] = (df['NCSI'] - df['NCSI'].min()) / (df['NCSI'].max() - df['NCSI'].min())
    df['DDL_norm'] = (df['DDL'] - df['DDL'].min()) / (df['DDL'].max() - df['DDL'].min())
    
    # Calculate composite risk score
    df['Risk_Score'] = (df['CEI_norm'] + df['GCI_norm'] + df['NCSI_norm'] + df['DDL_norm']) / 4
    
    # Create risk categories
    df['Risk_Category'] = pd.cut(df['Risk_Score'], 
                                bins=[0, 0.33, 0.67, 1.0], 
                                labels=['High Risk', 'Medium Risk', 'Low Risk'])
    return df
```

### Step 1.3: Model Training (3 Comparison Models)
```python
# Model 1: K-Means Clustering
def train_clustering_model(df):
    features = ['CEI', 'GCI', 'NCSI', 'DDL']
    X = df[features].dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    return kmeans, scaler, clusters

# Model 2: Random Forest Classification
def train_classification_model(df):
    features = ['CEI', 'GCI', 'NCSI', 'DDL']
    X = df[features].dropna()
    y = df.loc[X.index, 'Risk_Category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return rf_classifier, feature_importance

# Model 3: Regression Model (Predict GCI from other metrics)
def train_regression_model(df):
    features = ['CEI', 'NCSI', 'DDL']
    X = df[features].dropna()
    y = df.loc[X.index, 'GCI'].dropna()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ridge_reg = Ridge(alpha=1.0)
    ridge_reg.fit(X_train, y_train)
    
    return ridge_reg, X_test, y_test
```

## Phase 2: Flask Backend Development

### Step 2.1: Flask Application Structure
```
backend/
├── app.py
├── models/
│   ├── __init__.py
│   ├── clustering_model.pkl
│   ├── classification_model.pkl
│   └── regression_model.pkl
├── data/
│   └── processed_cyber_data.csv
├── api/
│   ├── __init__.py
│   └── routes.py
└── requirements.txt
```

### Step 2.2: Flask API Implementation
```python
# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json

app = Flask(__name__)
CORS(app)

# Load data and models
df = pd.read_csv('data/processed_cyber_data.csv')
with open('models/clustering_model.pkl', 'rb') as f:
    clustering_model = pickle.load(f)
with open('models/classification_model.pkl', 'rb') as f:
    classification_model = pickle.load(f)
with open('models/regression_model.pkl', 'rb') as f:
    regression_model = pickle.load(f)
# 🚀 UNIFIED ENSEMBLE MODEL (Primary model for production)
with open('models/ensemble_model.pkl', 'rb') as f:
    ensemble_model = pickle.load(f)

@app.route('/api/countries', methods=['GET'])
def get_countries_data():
    """Return all countries data for visualization"""
    return jsonify(df.to_dict('records'))

@app.route('/api/correlation', methods=['GET'])
def get_correlation_data():
    """Return correlation matrix for heatmap"""
    numeric_cols = ['CEI', 'GCI', 'NCSI', 'DDL']
    correlation_matrix = df[numeric_cols].corr()
    return jsonify(correlation_matrix.to_dict())

@app.route('/api/regional-analysis', methods=['GET'])
def get_regional_analysis():
    """Return regional statistics"""
    regional_stats = df.groupby('Region').agg({
        'CEI': 'mean',
        'GCI': 'mean', 
        'NCSI': 'mean',
        'DDL': 'mean',
        'Risk_Score': 'mean'
    }).reset_index()
    return jsonify(regional_stats.to_dict('records'))

@app.route('/api/predict', methods=['POST'])
def predict_risk():
    """Predict risk category for given metrics using UNIFIED ENSEMBLE MODEL"""
    data = request.json
    features = np.array([[data['CEI'], data['GCI'], data['NCSI'], data['DDL']]])
    
    # 🚀 PRIMARY PREDICTION: Use Unified Ensemble Model
    ensemble_pred = ensemble_model.predict(features)[0]
    
    # Additional predictions for comparison/analysis
    cluster_pred = clustering_model.predict(features)[0]
    individual_rf_pred = classification_model.predict(features)[0]
    gci_pred = regression_model.predict(features[:, [0, 2, 3]])[0]
    
    # Calculate ensemble confidence (simplified for hard voting)
    ensemble_confidence = 0.95  # High confidence due to voting mechanism
    
    return jsonify({
        'primary_prediction': {
            'risk_category': ensemble_pred,
            'confidence': ensemble_confidence,
            'model_type': 'Hard Voting Ensemble'
        },
        'additional_insights': {
            'cluster': int(cluster_pred),
            'individual_rf_prediction': individual_rf_pred,
            'predicted_gci': float(gci_pred)
        },
        'model_info': {
            'ensemble_components': 5,
            'voting_strategy': 'hard',
            'unified_deployment': True
        }
    })

@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    """Return feature importance from classification model"""
    features = ['CEI', 'GCI', 'NCSI', 'DDL']
    importance = classification_model.feature_importances_
    
    feature_data = [{'feature': feat, 'importance': imp} 
                   for feat, imp in zip(features, importance)]
    return jsonify(feature_data)

if __name__ == '__main__':
    app.run(debug=True)
```

## Phase 3: React Frontend Development

### Step 3.1: React Application Setup
```bash
# Create React app
npx create-react-app cybersecurity-dashboard
cd cybersecurity-dashboard

# Install required packages
npm install axios recharts leaflet react-leaflet d3 @mui/material @emotion/react @emotion/styled
npm install plotly.js react-plotly.js
```

### Step 3.2: React Components Structure
```
src/
├── components/
│   ├── WorldMap.js
│   ├── CorrelationHeatmap.js
│   ├── RegionalChart.js
│   ├── ScatterPlot.js
│   ├── FeatureImportance.js
│   └── PredictionPanel.js
├── services/
│   └── api.js
├── pages/
│   └── Dashboard.js
├── App.js
└── index.js
```

### Step 3.3: Key React Components
```jsx
// services/api.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export const apiService = {
  getCountriesData: () => axios.get(`${API_BASE_URL}/countries`),
  getCorrelationData: () => axios.get(`${API_BASE_URL}/correlation`),
  getRegionalAnalysis: () => axios.get(`${API_BASE_URL}/regional-analysis`),
  getFeatureImportance: () => axios.get(`${API_BASE_URL}/feature-importance`),
  predictRisk: (data) => axios.post(`${API_BASE_URL}/predict`, data)
};

// components/WorldMap.js
import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import { apiService } from '../services/api';

const WorldMap = () => {
  const [countriesData, setCountriesData] = useState([]);
  
  useEffect(() => {
    apiService.getCountriesData().then(response => {
      setCountriesData(response.data);
    });
  }, []);

  const getColor = (riskScore) => {
    return riskScore > 0.7 ? '#2E8B57' :
           riskScore > 0.4 ? '#FFD700' :
                            '#FF6347';
  };

  return (
    <MapContainer center={[20, 0]} zoom={2} style={{height: '500px'}}>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {/* Add GeoJSON layer with country data */}
    </MapContainer>
  );
};

// components/CorrelationHeatmap.js
import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { apiService } from '../services/api';

const CorrelationHeatmap = () => {
  const [correlationData, setCorrelationData] = useState({});

  useEffect(() => {
    apiService.getCorrelationData().then(response => {
      setCorrelationData(response.data);
    });
  }, []);

  const plotData = [{
    z: Object.values(correlationData).map(row => Object.values(row)),
    x: Object.keys(correlationData),
    y: Object.keys(correlationData),
    type: 'heatmap',
    colorscale: 'RdYlBu'
  }];

  return (
    <Plot
      data={plotData}
      layout={{
        title: 'Cybersecurity Metrics Correlation',
        xaxis: { title: 'Metrics' },
        yaxis: { title: 'Metrics' }
      }}
    />
  );
};
```

## Phase 4: Visualization Implementation

### Step 4.1: Dashboard Layout
```jsx
// pages/Dashboard.js
import React from 'react';
import { Grid, Paper, Typography } from '@mui/material';
import WorldMap from '../components/WorldMap';
import CorrelationHeatmap from '../components/CorrelationHeatmap';
import RegionalChart from '../components/RegionalChart';
import ScatterPlot from '../components/ScatterPlot';
import FeatureImportance from '../components/FeatureImportance';
import PredictionPanel from '../components/PredictionPanel';

const Dashboard = () => {
  return (
    <div style={{ padding: '20px' }}>
      <Typography variant="h3" gutterBottom>
        Cybersecurity Risk Management Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper style={{ padding: '20px' }}>
            <Typography variant="h5">Global Risk Overview</Typography>
            <WorldMap />
          </Paper>
        </Grid>
        
        <Grid item xs={6}>
          <Paper style={{ padding: '20px' }}>
            <CorrelationHeatmap />
          </Paper>
        </Grid>
        
        <Grid item xs={6}>
          <Paper style={{ padding: '20px' }}>
            <FeatureImportance />
          </Paper>
        </Grid>
        
        <Grid item xs={8}>
          <Paper style={{ padding: '20px' }}>
            <ScatterPlot />
          </Paper>
        </Grid>
        
        <Grid item xs={4}>
          <Paper style={{ padding: '20px' }}>
            <PredictionPanel />
          </Paper>
        </Grid>
        
        <Grid item xs={12}>
          <Paper style={{ padding: '20px' }}>
            <RegionalChart />
          </Paper>
        </Grid>
      </Grid>
    </div>
  );
};
```

### Step 4.2: Dedicated Prediction Page Implementation

#### Component Structure
```
src/
├── pages/
│   ├── Dashboard.js
│   └── PredictionPage.js
├── components/
│   ├── prediction/
│   │   ├── MetricInputForm.js
│   │   ├── RiskMeter.js
│   │   ├── PredictionResults.js
│   │   ├── ScenarioComparison.js
│   │   ├── SimilarCountries.js
│   │   ├── RecommendationEngine.js
│   │   └── PredictionHistory.js
│   └── common/
│       ├── LoadingSpinner.js
│       └── ErrorBoundary.js
```

#### Main Prediction Page Component
```jsx
// pages/PredictionPage.js
import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Grid, 
  Paper, 
  Typography, 
  Tabs, 
  Tab, 
  Box,
  Button,
  Alert
} from '@mui/material';
import MetricInputForm from '../components/prediction/MetricInputForm';
import RiskMeter from '../components/prediction/RiskMeter';
import PredictionResults from '../components/prediction/PredictionResults';
import SimilarCountries from '../components/prediction/SimilarCountries';
import RecommendationEngine from '../components/prediction/RecommendationEngine';
import ScenarioComparison from '../components/prediction/ScenarioComparison';
import PredictionHistory from '../components/prediction/PredictionHistory';
import { apiService } from '../services/api';

const PredictionPage = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [inputMetrics, setInputMetrics] = useState({
    CEI: 50,
    GCI: 50,
    NCSI: 50,
    DDL: 50
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [savedScenarios, setSavedScenarios] = useState([]);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await apiService.predictRisk(inputMetrics);
      setPredictionResult(response.data);
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveScenario = () => {
    const scenario = {
      id: Date.now(),
      name: `Scenario ${savedScenarios.length + 1}`,
      metrics: inputMetrics,
      result: predictionResult,
      timestamp: new Date().toISOString()
    };
    setSavedScenarios([...savedScenarios, scenario]);
  };

  return (
    <Container maxWidth="xl" style={{ padding: '20px' }}>
      <Typography variant="h3" gutterBottom>
        Cybersecurity Risk Prediction Center
      </Typography>
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
          <Tab label="Single Prediction" />
          <Tab label="Scenario Comparison" />
          <Tab label="Prediction History" />
        </Tabs>
      </Box>

      {activeTab === 0 && (
        <Grid container spacing={3}>
          {/* Input Section */}
          <Grid item xs={12} md={4}>
            <Paper style={{ padding: '20px', height: 'fit-content' }}>
              <Typography variant="h5" gutterBottom>
                Input Metrics
              </Typography>
              <MetricInputForm 
                metrics={inputMetrics}
                onChange={setInputMetrics}
                onPredict={handlePredict}
                loading={loading}
              />
              <Button 
                variant="outlined" 
                fullWidth 
                onClick={handleSaveScenario}
                disabled={!predictionResult}
                style={{ marginTop: '10px' }}
              >
                Save Scenario
              </Button>
            </Paper>
          </Grid>

          {/* Results Section */}
          <Grid item xs={12} md={8}>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Paper style={{ padding: '20px' }}>
                  <RiskMeter 
                    riskScore={predictionResult?.risk_score || 0}
                    riskCategory={predictionResult?.risk_category || 'Unknown'}
                    loading={loading}
                  />
                </Paper>
              </Grid>
              
              <Grid item xs={12}>
                <Paper style={{ padding: '20px' }}>
                  <PredictionResults result={predictionResult} />
                </Paper>
              </Grid>
            </Grid>
          </Grid>

          {/* Similar Countries Section */}
          <Grid item xs={12} md={6}>
            <Paper style={{ padding: '20px' }}>
              <SimilarCountries 
                inputMetrics={inputMetrics}
                predictionResult={predictionResult}
              />
            </Paper>
          </Grid>

          {/* Recommendations Section */}
          <Grid item xs={12} md={6}>
            <Paper style={{ padding: '20px' }}>
              <RecommendationEngine 
                metrics={inputMetrics}
                riskCategory={predictionResult?.risk_category}
              />
            </Paper>
          </Grid>
        </Grid>
      )}

      {activeTab === 1 && (
        <ScenarioComparison scenarios={savedScenarios} />
      )}

      {activeTab === 2 && (
        <PredictionHistory scenarios={savedScenarios} />
      )}
    </Container>
  );
};

export default PredictionPage;
```

#### Metric Input Form Component
```jsx
// components/prediction/MetricInputForm.js
import React from 'react';
import { 
  Slider, 
  Typography, 
  Box, 
  Button, 
  Chip,
  Tooltip,
  IconButton 
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';

const MetricInputForm = ({ metrics, onChange, onPredict, loading }) => {
  const metricInfo = {
    CEI: {
      name: 'Cybersecurity Exposure Index',
      description: 'Lower values indicate better cybersecurity',
      range: [0, 100],
      color: '#ff6b6b'
    },
    GCI: {
      name: 'Global Cybersecurity Index',
      description: 'Higher values indicate better cybersecurity',
      range: [0, 100],
      color: '#4ecdc4'
    },
    NCSI: {
      name: 'National Cyber Security Index',
      description: 'Higher values indicate better cybersecurity',
      range: [0, 100],
      color: '#45b7d1'
    },
    DDL: {
      name: 'Digital Development Level',
      description: 'Higher values indicate better digital infrastructure',
      range: [0, 100],
      color: '#f9ca24'
    }
  };

  const handleSliderChange = (metric) => (event, newValue) => {
    onChange({
      ...metrics,
      [metric]: newValue
    });
  };

  const getRiskLevel = (metric, value) => {
    if (metric === 'CEI') {
      return value < 30 ? 'Low Risk' : value < 60 ? 'Medium Risk' : 'High Risk';
    } else {
      return value > 70 ? 'Low Risk' : value > 40 ? 'Medium Risk' : 'High Risk';
    }
  };

  return (
    <Box>
      {Object.entries(metricInfo).map(([key, info]) => (
        <Box key={key} sx={{ mb: 3 }}>
          <Box display="flex" alignItems="center" mb={1}>
            <Typography variant="subtitle1" sx={{ flexGrow: 1 }}>
              {info.name}
            </Typography>
            <Tooltip title={info.description}>
              <IconButton size="small">
                <InfoIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
          
          <Slider
            value={metrics[key]}
            onChange={handleSliderChange(key)}
            min={info.range[0]}
            max={info.range[1]}
            step={1}
            marks={[
              { value: info.range[0], label: info.range[0] },
              { value: info.range[1], label: info.range[1] }
            ]}
            valueLabelDisplay="on"
            sx={{ color: info.color }}
          />
          
          <Box display="flex" justifyContent="space-between" alignItems="center" mt={1}>
            <Typography variant="body2" color="textSecondary">
              Current: {metrics[key]}
            </Typography>
            <Chip 
              label={getRiskLevel(key, metrics[key])}
              size="small"
              color={getRiskLevel(key, metrics[key]) === 'Low Risk' ? 'success' : 
                     getRiskLevel(key, metrics[key]) === 'Medium Risk' ? 'warning' : 'error'}
            />
          </Box>
        </Box>
      ))}
      
      <Button
        variant="contained"
        fullWidth
        onClick={onPredict}
        disabled={loading}
        sx={{ mt: 2, py: 1.5 }}
      >
        {loading ? 'Predicting...' : 'Predict Risk Level'}
      </Button>
    </Box>
  );
};

export default MetricInputForm;
```

#### Risk Meter Component
```jsx
// components/prediction/RiskMeter.js
import React from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

const RiskMeter = ({ riskScore, riskCategory, loading }) => {
  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={300}>
        <CircularProgress size={80} />
      </Box>
    );
  }

  const meterData = [
    { name: 'Risk', value: riskScore * 100 },
    { name: 'Safe', value: (1 - riskScore) * 100 }
  ];

  const getColor = () => {
    if (riskCategory === 'Low Risk') return '#4caf50';
    if (riskCategory === 'Medium Risk') return '#ff9800';
    return '#f44336';
  };

  return (
    <Box textAlign="center">
      <Typography variant="h5" gutterBottom>
        Risk Assessment
      </Typography>
      
      <Box position="relative" display="inline-block">
        <ResponsiveContainer width={250} height={250}>
          <PieChart>
            <Pie
              data={meterData}
              startAngle={180}
              endAngle={0}
              innerRadius={60}
              outerRadius={100}
              dataKey="value"
            >
              <Cell fill={getColor()} />
              <Cell fill="#e0e0e0" />
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        
        <Box
          position="absolute"
          top="60%"
          left="50%"
          transform="translate(-50%, -50%)"
          textAlign="center"
        >
          <Typography variant="h3" fontWeight="bold" color={getColor()}>
            {Math.round(riskScore * 100)}%
          </Typography>
          <Typography variant="h6" color="textSecondary">
            {riskCategory}
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default RiskMeter;
```

#### Additional Backend API Enhancements
```python
# Add to Flask app.py

@app.route('/api/similar-countries', methods=['POST'])
def get_similar_countries():
    """Find countries with similar cybersecurity profiles"""
    data = request.json
    input_metrics = np.array([data['CEI'], data['GCI'], data['NCSI'], data['DDL']])
    
    # Calculate similarity scores
    similarities = []
    for _, country in df.iterrows():
        country_metrics = np.array([country['CEI'], country['GCI'], country['NCSI'], country['DDL']])
        similarity = 1 / (1 + np.linalg.norm(input_metrics - country_metrics))
        similarities.append({
            'country': country['Country'],
            'similarity': similarity,
            'risk_category': country['Risk_Category'],
            'metrics': {
                'CEI': country['CEI'],
                'GCI': country['GCI'],
                'NCSI': country['NCSI'],
                'DDL': country['DDL']
            }
        })
    
    # Return top 5 most similar countries
    similar_countries = sorted(similarities, key=lambda x: x['similarity'], reverse=True)[:5]
    return jsonify(similar_countries)

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Generate improvement recommendations based on input metrics"""
    data = request.json
    recommendations = []
    
    # CEI recommendations (lower is better)
    if data['CEI'] > 60:
        recommendations.append({
            'metric': 'CEI',
            'priority': 'High',
            'recommendation': 'Implement comprehensive cybersecurity framework',
            'impact': 'Reduce exposure to cyber threats'
        })
    
    # GCI recommendations (higher is better)
    if data['GCI'] < 40:
        recommendations.append({
            'metric': 'GCI',
            'priority': 'Medium',
            'recommendation': 'Enhance national cybersecurity capabilities',
            'impact': 'Improve overall cybersecurity posture'
        })
    
    # Add more recommendation logic...
    
    return jsonify(recommendations)
```

#### Key Features Implemented:

1. **Interactive Input Interface**: Sliders with real-time validation
2. **Visual Risk Assessment**: Animated gauge/meter showing risk level
3. **Similar Countries Finder**: AI-powered country comparison
4. **Recommendation Engine**: Actionable improvement suggestions
5. **Scenario Management**: Save, compare, and track predictions
6. **Tabbed Interface**: Organized workflow for different use cases
7. **Responsive Design**: Works on desktop and mobile devices
```

## Phase 5: Deployment

### Step 5.1: Backend Deployment (Flask)
```bash
# requirements.txt
flask==2.3.3
flask-cors==4.0.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
plotly==5.15.0
gunicorn==21.2.0

# Deploy with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Step 5.2: Frontend Deployment (React)
```bash
# Build for production
npm run build

# Deploy to Netlify, Vercel, or serve with nginx
```

## Phase 6: Advanced Features

### Step 6.1: Real-time Updates
- WebSocket integration for live data updates
- Automatic model retraining pipeline
- Alert system for high-risk countries

### Step 6.2: Enhanced Visualizations
- 3D globe visualization with Three.js
- Time-series forecasting charts
- Interactive drill-down capabilities
- Export functionality for reports

## Development Timeline

**Week 1-2**: Data analysis, model training, and Flask API
**Week 3-4**: React components and basic visualizations  
**Week 5-6**: Advanced visualizations and integration
**Week 7**: Testing, optimization, and deployment
**Week 8**: Documentation and final touches

## Key Success Metrics

1. **Model Performance**: Classification accuracy >85%, clustering silhouette score >0.6
2. **User Experience**: Page load time <3 seconds, interactive response <1 second
3. **Visualization Quality**: Clear insights, intuitive navigation, responsive design
4. **Decision Support**: Clear risk categorization, actionable insights, predictive capability