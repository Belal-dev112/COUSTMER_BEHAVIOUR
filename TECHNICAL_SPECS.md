# Technical Specification Document
## Customer Churn Prediction System

**Version**: 1.0  
**Date**: February 8, 2026  
**Status**: Production Ready

---

## System Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ CSV Import   │  │ API Connect  │  │ Database     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  DATA PREPROCESSING LAYER                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ DataPreprocessor Class                                    │  │
│  │  • Missing value imputation                               │  │
│  │  • Outlier detection & capping                           │  │
│  │  • Categorical encoding                                   │  │
│  │  • Feature scaling                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING LAYER                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 14 Derived Features:                                      │  │
│  │  • CLV, ARPM, Engagement Score                           │  │
│  │  • Decay rates, Intensity metrics                        │  │
│  │  • Binary indicators, Composite scores                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      MODELING LAYER                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   LR     │  │    RF    │  │    GB    │  │   SVM    │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│  ┌──────────┐                                                   │
│  │   MLP    │  ← Class balancing (SMOTE)                       │
│  └──────────┘  ← Hyperparameter tuning                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   EXPLAINABILITY LAYER                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ ModelExplainability Class                                 │  │
│  │  • Feature importance calculation                         │  │
│  │  • Top driver identification                             │  │
│  │  • SHAP value computation (optional)                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   SEGMENTATION LAYER                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ CustomerSegmentation Class                                │  │
│  │  • K-Means clustering                                     │  │
│  │  • PCA dimensionality reduction                          │  │
│  │  • Segment profiling & naming                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     ACTION LAYER                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ ChurnPreventionEngine Class                               │  │
│  │  • Risk scoring                                           │  │
│  │  • Issue identification                                   │  │
│  │  • Strategy recommendation                               │  │
│  │  • Priority assignment                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                                │
│  • CSV exports  • PNG visualizations  • JSON reports            │
│  • PKL models   • API endpoints       • Dashboards              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. CustomerDataGenerator

**Purpose**: Generate synthetic but realistic customer data

**Key Methods**:
```python
generate_data() -> pd.DataFrame
    Returns: 10,000 customer records with realistic churn patterns
    
_calculate_churn_probability(...)
    Logic: Probabilistic model based on multiple risk factors
    Formula: P(churn) = f(tenure, contract, support, payment, engagement)
```

**Churn Probability Model**:
```
P_churn = min(0.9, sum([
    0.3 * exp(-tenure/12),                    # Tenure effect
    0.25 if contract="MTM" else 0.1/0.05,    # Contract risk
    0.05 * support_tickets,                   # Support issues
    0.1 * complaints,                         # Complaint impact
    0.02 * payment_delay,                     # Payment risk
    0.01 * (30 - login_freq),                # Engagement
    0.01 * days_since_login,                 # Recency
    0.002 * (100 - feature_usage),           # Usage depth
    0.001 * (charges - 50)                   # Price sensitivity
]))
```

**Data Quality Issues Introduced**:
- 5-10% missing values in numerical columns
- 2% outliers (2-5x normal values)
- 1% inconsistent categorical entries

---

### 2. DataPreprocessor

**Purpose**: Clean and prepare data for modeling

#### 2.1 Data Cleaning

**Missing Value Strategy**:
```python
Numerical columns: Median imputation
Categorical columns: Mode imputation
Reason: Robust to outliers, preserves distribution
```

**Outlier Detection**:
```python
Method: Interquartile Range (IQR)
Formula: 
    Q1 = 25th percentile
    Q3 = 75th percentile
    IQR = Q3 - Q1
    Lower bound = Q1 - 3*IQR
    Upper bound = Q3 + 3*IQR
Action: Cap values at bounds (not removal)
```

**Categorical Encoding**:
```python
Method: Label Encoding
Handled: gender, location, contract_type, payment_method, tenure_category
Classes: Stored in self.label_encoders dict for inverse transform
```

**Feature Scaling**:
```python
Method: StandardScaler (z-score normalization)
Formula: z = (x - μ) / σ
Applied: All numerical features before modeling
```

#### 2.2 Feature Engineering

**Engineered Features Specification**:

| Feature | Formula | Type | Purpose |
|---------|---------|------|---------|
| customer_lifetime_value | monthly_charges × tenure_months | Continuous | Revenue metric |
| avg_revenue_per_month | total_charges / (tenure_months + 1) | Continuous | Spending pattern |
| engagement_score | 0.4×login_freq_norm + 0.3×feature_usage_norm + 0.3×session_norm | Continuous | Composite engagement |
| engagement_decay | days_since_last_login / (tenure_months + 1) | Continuous | Disengagement rate |
| support_intensity | (support_tickets + 2×complaints) / (tenure_months + 1) | Continuous | Issue frequency |
| payment_reliability | 100 - (payment_delay_days × 2) | Continuous | Payment behavior |
| value_cost_ratio | feature_usage_score / (monthly_charges + 1) | Continuous | Perceived value |
| tenure_category | Binned: 0-6, 6-12, 12-24, 24-72 months | Categorical | Lifecycle stage |
| is_high_complaints | complaints > 75th percentile | Binary | Risk flag |
| is_payment_delayed | payment_delay_days > 5 | Binary | Risk flag |
| is_low_engagement | engagement_score < 25th percentile | Binary | Risk flag |
| is_active_referrer | num_referrals > 0 | Binary | Advocacy indicator |
| usage_per_dollar | download_volume_gb / (monthly_charges + 1) | Continuous | Value extraction |
| recency_score | 100 - (days_since_last_login × 2) | Continuous | Activity recency |

---

### 3. SimpleSMOTE Implementation

**Purpose**: Handle class imbalance without external dependencies

**Algorithm**:
```python
For each minority class sample needed:
    1. Select random minority class sample (x)
    2. Find k nearest neighbors in minority class
    3. Select random neighbor (x_neighbor)
    4. Generate synthetic sample:
       x_synthetic = x + α × (x_neighbor - x)
       where α ~ Uniform(0, 1)
```

**Parameters**:
- `k_neighbors`: 5 (default)
- `random_state`: 42 (reproducibility)

**Performance**:
- Time Complexity: O(n²) for distance calculation
- Space Complexity: O(n)
- Suitable for: n < 100,000 samples

---

### 4. ChurnPredictionModels

**Purpose**: Train and evaluate multiple ML models

#### 4.1 Model Specifications

**Logistic Regression**:
```python
Parameters:
    max_iter: 1000
    random_state: 42
    solver: 'lbfgs' (default)
Strengths: Fast, interpretable, good baseline
Weaknesses: Linear decision boundary
Use Case: Quick iterations, coefficient analysis
```

**Random Forest**:
```python
Parameters:
    n_estimators: 100
    random_state: 42
    n_jobs: -1
    class_weight: None (handled by SMOTE)
Strengths: Handles non-linearity, robust to outliers
Weaknesses: Can overfit, less interpretable
Use Case: Production deployment, feature importance
```

**Gradient Boosting**:
```python
Parameters:
    n_estimators: 100
    learning_rate: 0.1 (default)
    max_depth: 3 (default)
    random_state: 42
Strengths: High accuracy, sequential learning
Weaknesses: Longer training time
Use Case: When accuracy is critical
```

**Support Vector Machine**:
```python
Parameters:
    kernel: 'rbf'
    probability: True (for predict_proba)
    random_state: 42
Strengths: Effective in high-dimensional space
Weaknesses: Slow on large datasets
Use Case: Medium-sized datasets, complex boundaries
```

**Neural Network (MLP)**:
```python
Parameters:
    hidden_layer_sizes: (100, 50)
    max_iter: 500
    random_state: 42
    activation: 'relu' (default)
Strengths: Can learn complex patterns
Weaknesses: Requires more data, tuning
Use Case: Large datasets, deep patterns
```

#### 4.2 Hyperparameter Tuning

**Grid Search Configuration**:
```python
Method: GridSearchCV
CV Strategy: 3-fold stratified cross-validation
Scoring: ROC-AUC
n_jobs: -1 (parallel processing)

Random Forest Grid:
    n_estimators: [50, 100, 200]
    max_depth: [10, 20, 30, None]
    min_samples_split: [2, 5, 10]
    min_samples_leaf: [1, 2, 4]
Total combinations: 3 × 4 × 3 × 3 = 108

Expected runtime: 5-15 minutes (depending on hardware)
```

#### 4.3 Evaluation Metrics

**Metrics Computed**:
```python
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 × (precision × recall) / (precision + recall)
roc_auc = ∫[0,1] TPR(FPR) d(FPR)
```

**Confusion Matrix Layout**:
```
                Predicted
              0         1
Actual  0  [[TN,       FP],
        1   [FN,       TP]]
```

**Interpretation Guidelines**:
- **Accuracy > 0.80**: Good overall performance
- **ROC-AUC > 0.85**: Strong discrimination ability
- **Recall > 0.75**: Catching most churners
- **Precision > 0.70**: Predictions are reliable

---

### 5. ModelExplainability

**Purpose**: Explain model predictions and feature importance

**Feature Importance Methods**:

1. **Tree-based models** (Random Forest, Gradient Boosting):
```python
Method: Mean Decrease in Impurity
Formula: Importance(f) = Σ [node uses f] × (impurity decrease)
```

2. **Linear models** (Logistic Regression):
```python
Method: Absolute coefficient values
Formula: Importance(f) = |β_f|
```

3. **SHAP (if available)**:
```python
Method: Shapley values from game theory
Measures: Marginal contribution of each feature
```

**Output Format**:
```python
DataFrame with columns:
    - feature: str
    - importance: float
Sorted by importance (descending)
Top 15 features returned
```

---

### 6. CustomerSegmentation

**Purpose**: Group customers into behavioral segments

**Algorithm**: K-Means Clustering

**Parameters**:
```python
n_clusters: 4
random_state: 42
n_init: 10
max_iter: 300 (default)
```

**Feature Selection for Clustering**:
```python
Features used: [
    'tenure_months',
    'monthly_charges',
    'login_frequency',
    'support_tickets',
    'complaints',
    'payment_delay_days'
]
Rationale: Behavioral and transactional mix
```

**Preprocessing**:
```python
StandardScaler applied before clustering
Reason: Equal weight to all features
```

**Segment Naming Logic**:
```python
if churn_rate > 0.5:
    name = "High Risk"
elif avg_tenure > 24 and churn_rate < 0.2:
    name = "Loyal Customers"
elif avg_logins < 10:
    name = "Low Engagement"
else:
    name = "Standard"
```

**PCA Visualization**:
```python
Components: 2
Variance explained: Typically 40-60%
Purpose: 2D visualization of high-dimensional data
```

---

### 7. ChurnPreventionEngine

**Purpose**: Generate personalized retention strategies

**Risk Threshold**:
```python
High-risk defined as: Top 20% churn probability
Typical cutoff: P(churn) >= 0.80
```

**Strategy Rules**:

| Condition | Issue | Action |
|-----------|-------|--------|
| support_tickets > 3 | High support tickets | Priority support escalation |
| complaints > 2 | Multiple complaints | Personal outreach from account manager |
| payment_delay_days > 10 | Payment delays | Flexible payment plan offer |
| login_frequency < 10 | Low engagement | Re-engagement campaign with tutorial |
| days_since_last_login > 20 | Inactive account | Win-back offer with discount |
| contract_type = MTM | No long-term commitment | Annual contract upgrade incentive |
| monthly_charges > 100 | High price point | Customized package review |

**Priority Assignment**:
```python
Priority = "High" if P(churn) > 0.8 else "Medium"
```

---

## Data Schemas

### Input Schema

```python
{
    "customer_id": str,           # Unique identifier
    "age": int,                   # 18-75
    "gender": str,                # M, F, Other
    "location": str,              # Urban, Suburban, Rural
    "tenure_months": int,         # 0-72
    "contract_type": str,         # Month-to-Month, 1-Year, 2-Year
    "monthly_charges": float,     # 20-150
    "total_charges": float,       # Cumulative charges
    "login_frequency": int,       # Logins per month
    "feature_usage_score": float, # 0-100
    "avg_session_duration": float,# Minutes
    "support_tickets": int,       # Count
    "complaints": int,            # Count
    "days_since_last_login": int, # 0-60
    "payment_delay_days": int,    # 0-30
    "payment_method": str,        # Credit Card, Bank Transfer, Digital Wallet
    "num_referrals": int,         # Count
    "promotional_offers_used": int,# Count
    "download_volume_gb": float,  # 0-500
    "churn": int                  # 0 or 1 (training only)
}
```

### Output Schema (Predictions)

```python
{
    "customer_id": str,
    "churn_probability": float,   # 0-1
    "risk_level": str,            # Low, Medium, High
    "segment": int,               # 0-3
    "primary_issues": list[str],
    "recommended_actions": list[str],
    "priority": str               # High, Medium
}
```

---

## Performance Benchmarks

### Training Performance

**Hardware**: Standard CPU (4 cores, 16GB RAM)

| Component | Time | Memory |
|-----------|------|--------|
| Data Generation | 5s | 200MB |
| Preprocessing | 10s | 150MB |
| Feature Engineering | 5s | 50MB |
| SMOTE Resampling | 15s | 300MB |
| Logistic Regression | 5s | 10MB |
| Random Forest | 60s | 500MB |
| Gradient Boosting | 90s | 300MB |
| SVM | 120s | 200MB |
| Neural Network | 45s | 100MB |
| Total | ~6 minutes | Peak 500MB |

### Inference Performance

**Latency (per customer)**:
- Feature engineering: <0.1ms
- Prediction: <0.5ms
- Total: <1ms

**Throughput**:
- Single customer: 1,000 predictions/second
- Batch (1000): 100,000 predictions/second

---

## Scalability Analysis

### Current Limits
- **Max customers (in-memory)**: 1M
- **Training data size**: 10GB
- **Model file size**: 138KB (pipeline)

### Scaling Strategies

**Horizontal Scaling**:
```python
# Partition data by customer segment
# Train separate models per segment
# Aggregate predictions
```

**Vertical Scaling**:
```python
# Use sparse matrices for categorical features
# Implement mini-batch training
# Use incremental learning (SGDClassifier)
```

**Database Integration**:
```python
# Replace in-memory processing
# Use SQL for aggregations
# Stream predictions to database
```

---

## Error Handling

### Data Quality Checks

```python
def validate_input(df):
    """Validate input data schema and quality."""
    
    # Check required columns
    required = [...]  # 19 columns
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Check data types
    if df['age'].dtype not in [int, float]:
        raise TypeError("age must be numeric")
    
    # Check value ranges
    if (df['age'] < 18).any() or (df['age'] > 75).any():
        raise ValueError("age must be between 18 and 75")
    
    # Check for impossible values
    if (df['tenure_months'] < 0).any():
        raise ValueError("tenure_months cannot be negative")
    
    return True
```

### Model Prediction Errors

```python
try:
    predictions = model.predict_proba(X)[:, 1]
except Exception as e:
    # Fallback to simple heuristic
    predictions = calculate_simple_risk_score(X)
    log_error(e)
```

---

## Monitoring & Maintenance

### Model Performance Monitoring

**Metrics to Track**:
```python
Weekly:
    - Prediction distribution
    - Average churn probability
    - High-risk customer count
    
Monthly:
    - Actual churn rate
    - Model accuracy drift
    - Feature importance changes
    
Quarterly:
    - ROC-AUC on holdout set
    - Calibration curve
    - Retrain decision
```

### Retraining Triggers

```python
Retrain if:
    1. Accuracy drops > 5%
    2. ROC-AUC drops > 0.05
    3. Feature distribution shifts significantly
    4. Business logic changes
    5. 3+ months since last training
```

---

## Security & Privacy

### Data Protection

```python
# PII Handling
SENSITIVE_COLUMNS = ['customer_id', 'email', 'phone']

def anonymize_data(df):
    """Hash sensitive identifiers."""
    for col in SENSITIVE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(hash_function)
    return df
```

### Model Access Control

```python
# Role-based permissions
PERMISSIONS = {
    'data_scientist': ['train', 'evaluate', 'export'],
    'analyst': ['predict', 'view'],
    'business_user': ['view_predictions']
}
```

---

## Testing Strategy

### Unit Tests

```python
def test_preprocessing():
    """Test data cleaning pipeline."""
    df = generate_test_data_with_issues()
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    
    assert df_clean.isnull().sum().sum() == 0
    assert all(df_clean['age'] >= 18)
    assert all(df_clean['age'] <= 75)
```

### Integration Tests

```python
def test_end_to_end_pipeline():
    """Test complete prediction pipeline."""
    df = load_test_customers()
    pipeline = ModelPipeline.load_pipeline('pipeline.pkl')
    predictions = pipeline.predict_pipeline(df)
    
    assert len(predictions) == len(df)
    assert all(0 <= p <= 1 for p in predictions)
```

### Performance Tests

```python
def test_inference_latency():
    """Ensure predictions are fast enough."""
    import time
    
    df = generate_customers(n=1000)
    pipeline = load_pipeline()
    
    start = time.time()
    predictions = pipeline.predict_pipeline(df)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # <1s for 1000 customers
```

---

## Deployment Options

### Option 1: Batch Processing

```python
# Cron job runs daily
0 2 * * * python predict_batch.py

# predict_batch.py
df = load_from_database()
predictions = pipeline.predict_pipeline(df)
save_to_database(predictions)
send_alerts(high_risk_customers)
```

### Option 2: REST API

```python
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    prob = pipeline.predict_pipeline(df)[0]
    return jsonify({'churn_probability': prob})
```

### Option 3: Streaming

```python
from kafka import KafkaConsumer, KafkaProducer

consumer = KafkaConsumer('customer_events')
producer = KafkaProducer('churn_predictions')

for message in consumer:
    customer = parse_message(message)
    prediction = pipeline.predict_pipeline(customer)
    producer.send({'customer_id': ..., 'prob': prediction})
```

---

## Version Control & Changelog

### Model Versioning

```python
MODEL_VERSION = {
    'major': 1,  # Breaking changes
    'minor': 0,  # New features
    'patch': 0   # Bug fixes
}

# Stored in model metadata
pipeline_data['version'] = MODEL_VERSION
pipeline_data['trained_date'] = datetime.now()
pipeline_data['training_data_hash'] = hash(df)
```

---

## Appendix: Mathematical Formulas

### ROC-AUC Calculation

```
AUC = ∫[0 to 1] TPR(t) d(FPR(t))

where:
    TPR(t) = True Positive Rate at threshold t
    FPR(t) = False Positive Rate at threshold t
    
Interpretation:
    0.5: Random classifier
    0.7-0.8: Acceptable
    0.8-0.9: Excellent
    0.9-1.0: Outstanding
```

### F1-Score Derivation

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2TP / (2TP + FP + FN)
   
Use case: When precision and recall are equally important
```

### Standard Score (Z-score)

```
z = (x - μ) / σ

where:
    x = raw score
    μ = population mean
    σ = population standard deviation
    
Result: Mean = 0, Standard Deviation = 1
```

---

**Document End**

For implementation questions, refer to:
- `churn_prediction_system.py` - Fully commented source code
- `README.md` - User documentation
- `QUICKSTART.md` - Getting started guide
