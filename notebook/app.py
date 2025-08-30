import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
import os

# Define the DataSanitizer class with lowercase handling
class DataSanitizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_config):
        self.feature_config = feature_config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Normalize column names
        X = X.rename(columns=lambda x: x.strip().lower())
        
        # Ensure expected columns
        expected = [col.lower() for col in self.feature_config['expected_features']]
        missing = [f for f in expected if f not in X.columns]
        extra = [f for f in X.columns if f not in expected]
        
        # Add missing columns with defaults
        for col in missing:
            X[col] = self.feature_config['numeric_means'][col]
        
        # Remove extra columns
        X = X[expected]
        
        # Fix data types and handle missing values
        for col in expected:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(self.feature_config['numeric_means'][col])
        
        return X

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .header { font-size: 36px !important; font-weight: bold; color: #1f77b4; padding: 10px 0; }
    .subheader { font-size: 24px !important; color: #2ca02c; margin: 20px 0 10px 0; }
    .critical { color: #ff4b4b; font-weight: bold; }
    .safe { color: #0db518; font-weight: bold; }
    .info-box { background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        bundle = joblib.load('churn_model.pkl')
        return bundle
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Prediction function with error handling
def predict_churn(new_data):
    try:
        bundle = load_model()
        pipeline = bundle['pipeline']
        threshold = bundle['threshold']
        
        # Get probabilities
        proba = pipeline.predict_proba(new_data)[:, 1]
        
        # Apply threshold
        predictions = (proba >= threshold).astype(int)
        
        return predictions, proba
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Function to read uploaded file
def read_uploaded_file(uploaded_file):
    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel.")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

# Main app
def main():
    # Load model
    try:
        bundle = load_model()
        feature_config = bundle['feature_config']
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose mode:", ["Batch Prediction", "Single Prediction", "Model Info"])
    
    # Main content
    st.markdown('<div class="header">📊 Customer Churn Prediction</div>', unsafe_allow_html=True)
    
    if app_mode == "Batch Prediction":
        st.markdown('<div class="subheader">📁 Batch Prediction</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            df = read_uploaded_file(uploaded_file)
            
            if df is not None:
                st.success("✅ File uploaded successfully!")
                
                # Show expected columns
                expected_cols = [col.lower() for col in feature_config['expected_features']]
                st.info(f"📋 Expected columns: {', '.join(expected_cols)}")
                
                st.dataframe(df.head(3))
                
                if st.button("🔮 Predict Churn", type="primary"):
                    with st.spinner("Making predictions..."):
                        predictions, probabilities = predict_churn(df)
                        
                        if predictions is None:
                            return
                            
                        results_df = df.copy()
                        results_df['Churn_Probability'] = probabilities
                        results_df['Churn_Prediction'] = ['Churned' if p == 1 else 'Retained' for p in predictions]
                        
                        st.subheader("📊 Results")
                        st.dataframe(results_df)
                        
                        # Statistics
                        churned_count = sum(predictions)
                        total_customers = len(predictions)
                        churn_rate = (churned_count / total_customers) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Customers", total_customers)
                        with col2:
                            st.metric("At Risk", f"{churned_count} ({churn_rate:.1f}%)")
                        with col3:
                            st.metric("Avg Risk Score", f"{np.mean(probabilities)*100:.1f}%")
                        
                        # Download results
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name='churn_predictions.csv',
                            mime='text/csv'
                        )
    
   # ... [keep all the previous code above] ...

    elif app_mode == "Single Prediction":
        st.markdown('<div class="subheader">👤 Single Customer Prediction</div>', unsafe_allow_html=True)
        
        with st.form("customer_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Customer Details**")
                tenure = st.slider("Tenure (months)", 0, 36, 12, 
                                  help="How long the customer has been with the company")
                addresses = st.slider("Number of Addresses", 1, 10, 2,
                                     help="Number of saved shipping addresses")
                order_count = st.slider("Total Orders", 1, 100, 5,
                                       help="Lifetime number of orders placed")
            
            with col2:
                st.markdown("**🛒 Behavior Metrics**")
                last_order = st.slider("Days Since Last Order", 0, 90, 7,
                                      help="Inactivity period since last purchase")
                cashback = st.slider("Cashback Amount ($)", 0.0, 500.0, 125.0,
                                    help="Available rewards balance")
                satisfaction = st.slider("Satisfaction Score (1-5)", 1.0, 5.0, 4.0, 0.1,
                                        help="Customer satisfaction rating")
            
            submit_button = st.form_submit_button("🔮 Predict Churn Risk")
        
        if submit_button:
            # CORRECTED: Using new feature set
            input_data = {
                'Tenure': [tenure],
                'NumberOfAddress': [addresses],
                'CashbackAmount': [cashback],
                'DaySinceLastOrder': [last_order],
                'OrderCount': [order_count],
                'SatisfactionScore': [satisfaction]
            }
            
            input_df = pd.DataFrame(input_data)
            
            # Show what's being sent to the model (for debugging)
            st.write("📋 Input data being analyzed:")
            st.dataframe(input_df)
            
            with st.spinner("Analyzing customer data..."):
                prediction, probability = predict_churn(input_df)
                
                if prediction is not None:
                    churn_prob = probability[0] * 100
                    threshold = bundle['threshold'] * 100
                    
                    st.subheader("🎯 Prediction Result")
                    
                    # Enhanced risk display
                    risk_level = "🚨 CRITICAL RISK" if churn_prob > 70 else \
                                "⚠️ HIGH RISK" if churn_prob > 50 else \
                                "🔶 MEDIUM RISK" if churn_prob > 30 else \
                                "✅ LOW RISK"
                    
                    st.markdown(f"### {risk_level}")
                    
                    if prediction[0] == 1:
                        st.error(f"**{churn_prob:.1f}% chance of churning**")
                        st.markdown(f"<p class='critical'>Exceeds safety threshold of {threshold:.1f}%</p>", unsafe_allow_html=True)
                    else:
                        st.success(f"**{churn_prob:.1f}% chance of churning**")
                        st.markdown(f"<p class='safe'>Below risk threshold of {threshold:.1f}%</p>", unsafe_allow_html=True)
                    
                    # Enhanced visual indicator
                    fig, ax = plt.subplots(figsize=(12, 3))
                    colors = ['green'] * 25 + ['yellow'] * 25 + ['orange'] * 25 + ['red'] * 25
                    ax.barh(['Churn Risk'], [churn_prob], color=colors[int(churn_prob)-1] if churn_prob <= 100 else 'red')
                    ax.axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.1f}%)')
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('Churn Probability (%)')
                    ax.set_title('Customer Risk Assessment')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # AI Recommendations (placeholder for Claude integration)
                    st.markdown("---")
                    st.subheader("🤖 Recommended Actions")
                    
                    if churn_prob > 70:
                        st.warning("""
                        **Immediate Action Required:**
                        - Personal phone call from account manager
                        - 25% discount on next order
                        - Free expedited shipping
                        - Satisfaction survey with incentive
                        """)
                    elif churn_prob > 50:
                        st.info("""
                        **Proactive Engagement:**
                        - Personalized email campaign
                        - 15% loyalty discount
                        - Request feedback on experience
                        - Special early access to new products
                        """)
                    elif churn_prob > 30:
                        st.info("""
                        **Maintenance Mode:**
                        - Monthly newsletter inclusion
                        - Occasional promotional offers
                        - Monitor engagement metrics
                        - Standard loyalty program benefits
                        """)
                    else:
                        st.success("""
                        **Healthy Relationship:**
                        - Continue current engagement strategy
                        - Quarterly satisfaction check-in
                        - Standard marketing communications
                        - Maintain service quality
                        """)

# ... [keep the rest of the code below] ...
    
    elif app_mode == "Model Info":
        st.markdown('<div class="subheader">ℹ️ Model Information</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write("**📋 Features Used:**")
        for feature in feature_config['expected_features']:
            st.write(f"- {feature}")
        
        st.write(f"**⚖️ Decision Threshold:** {bundle['threshold']:.2f}")
        st.write("**🤖 Model Type:** Calibrated XGBoost")
        
        # Show feature means for reference
        st.write("**📊 Feature Default Values (for missing data):**")
        means_df = pd.DataFrame.from_dict(feature_config['numeric_means'], orient='index', columns=['Default Value'])
        st.dataframe(means_df)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()