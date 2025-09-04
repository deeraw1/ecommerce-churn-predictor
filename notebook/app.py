import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load env file
load_dotenv(".env")

# Read key
hf_api_key = os.getenv("HF_API_KEY")

# Init client
client = InferenceClient(api_key=hf_api_key)

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

# Helper function for consistent column access
def get_feature_value(customer_data, feature_name):
    """Safely get feature value with case-insensitive lookup"""
    lower_cols = [col.lower() for col in customer_data.columns]
    if feature_name.lower() in lower_cols:
        actual_col = customer_data.columns[lower_cols.index(feature_name.lower())]
        return customer_data[actual_col].iloc[0]
    return "N/A"

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
    .ai-insight { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #4CAF50; }
</style>
""", unsafe_allow_html=True)

# Load the model with better error handling
@st.cache_resource
def load_model():
    try:
        # Try relative path first
        try:
            bundle = joblib.load("model/churn_model.pkl")
            return bundle
        except:
            # Try absolute path as fallback
            bundle = joblib.load("C:\\Users\\USER\\Documents\\github_repo\\ecommerce-churn-predictor\\model\\churn_model.pkl")
            return bundle
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please make sure the model file exists in the correct location.")
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

def get_ai_insights(customer_data, churn_prob, prediction):
    try:
        # Use the helper function to get values safely
        prompt = f"""
        As a customer retention expert, analyze this customer data and provide specific recommendations to reduce churn risk.

        Customer Profile:
        - Tenure: {get_feature_value(customer_data, 'Tenure')} months
        - Number of Addresses: {get_feature_value(customer_data, 'NumberOfAddress')}
        - Cashback Amount: ${get_feature_value(customer_data, 'CashbackAmount')}
        - Days Since Last Order: {get_feature_value(customer_data, 'DaySinceLastOrder')}
        - Order Count: {get_feature_value(customer_data, 'OrderCount')}
        - Satisfaction Score: {get_feature_value(customer_data, 'SatisfactionScore')}/5
    
        Churn Risk: {churn_prob:.1f}% ({'High Risk' if prediction[0] == 1 else 'Low Risk'})

        Please provide:
        1. A brief analysis of why this customer might be at risk
        2. 3-5 specific, actionable recommendations to retain this customer
        3. Any immediate actions that should be taken

        Format the response with clear sections and bullet points.
        """

        # Try using the chat completion endpoint instead
        try:
            response = client.chat_completion(
                model="HuggingFaceH4/zephyr-7b-beta",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except:
            # Fallback to text generation
            response = client.text_generation(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                prompt=prompt,
                max_new_tokens=400,
                temperature=0.7
            )
            return response

    except Exception as e:
        # Provide fallback recommendations if API fails
        return f"""
        **AI Insights Unavailable - Using Fallback Recommendations**
        
        Based on the customer's {churn_prob:.1f}% churn risk, here are our standard recommendations:
        
        **Immediate Actions:**
        - Personal outreach from customer success team
        - Special discount or promotion offer
        - Request feedback on their experience
        
        **Retention Strategies:**
        - Implement a win-back campaign if churn risk > 50%
        - Offer personalized product recommendations
        - Provide exclusive content or early access to new features
        
        **Why this customer might be at risk:**
        - High inactivity ({get_feature_value(customer_data, 'DaySinceLastOrder')} days since last order)
        - Satisfaction score of {get_feature_value(customer_data, 'SatisfactionScore')}/5
        
        *Note: Could not connect to AI service. Error: {str(e)}*
        """
    
# Main app
def main():
    try:
        bundle = load_model()
        feature_config = bundle['feature_config']
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose mode:", ["Batch Prediction", "Single Prediction", "Model Info"])
    
    st.markdown('<div class="header">📊 Customer Churn Prediction</div>', unsafe_allow_html=True)
    
    if app_mode == "Batch Prediction":
        st.markdown('<div class="subheader">📁 Batch Prediction</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            df = read_uploaded_file(uploaded_file)
            
            if df is not None:
                st.success("✅ File uploaded successfully!")
                expected_cols = [col.lower() for col in feature_config['expected_features']]
                st.info(f"📋 Expected columns: {', '.join(expected_cols)}")
                st.dataframe(df.head(3))
                
                if st.button("🔮 Predict Churn", type="primary"):
                    with st.spinner("Making predictions..."):
                        predictions, probabilities = predict_churn(df)
                        if predictions is None: return
                        
                        results_df = df.copy()
                        results_df['Churn_Probability'] = probabilities
                        results_df['Churn_Prediction'] = ['Churned' if p == 1 else 'Retained' for p in predictions]
                        
                        st.subheader("📊 Results")
                        st.dataframe(results_df)
                        
                        churned_count = sum(predictions)
                        total_customers = len(predictions)
                        churn_rate = (churned_count / total_customers) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1: st.metric("Total Customers", total_customers)
                        with col2: st.metric("At Risk", f"{churned_count} ({churn_rate:.1f}%)")
                        with col3: st.metric("Avg Risk Score", f"{np.mean(probabilities)*100:.1f}%")
                        
                        if churned_count > 0:
                            st.subheader("🤖 AI Insights for High-Risk Customers")
                            high_risk_idx = np.argmax(probabilities)
                            high_risk_customer = df.iloc[[high_risk_idx]]
                            high_risk_prob = probabilities[high_risk_idx] * 100
                            
                            with st.spinner("Generating AI insights..."):
                                ai_insights = get_ai_insights(high_risk_customer, high_risk_prob, [predictions[high_risk_idx]])
                                st.markdown(f'<div class="ai-insight">{ai_insights}</div>', unsafe_allow_html=True)
                        
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name='churn_predictions.csv',
                            mime='text/csv'
                        )
    
    elif app_mode == "Single Prediction":
        st.markdown('<div class="subheader">👤 Single Customer Prediction</div>', unsafe_allow_html=True)
        
        with st.form("customer_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📊 Customer Details**")
                tenure = st.slider("Tenure (months)", 0, 36, 12)
                addresses = st.slider("Number of Addresses", 1, 10, 2)
                order_count = st.slider("Total Orders", 1, 100, 5)
            with col2:
                st.markdown("**🛒 Behavior Metrics**")
                last_order = st.slider("Days Since Last Order", 0, 90, 7)
                cashback = st.slider("Cashback Amount ($)", 0.0, 500.0, 125.0)
                satisfaction = st.slider("Satisfaction Score (1-5)", 1.0, 5.0, 4.0, 0.1)
            
            submit_button = st.form_submit_button("🔮 Predict Churn Risk")
        
        if submit_button:
            input_data = {
                'Tenure': [tenure],
                'NumberOfAddress': [addresses],
                'CashbackAmount': [cashback],
                'DaySinceLastOrder': [last_order],
                'OrderCount': [order_count],
                'SatisfactionScore': [satisfaction]
            }
            input_df = pd.DataFrame(input_data)
            
            # Convert to lowercase for consistency with DataSanitizer
            input_df_lower = input_df.rename(columns=lambda x: x.lower())
            
            st.write("📋 Input data being analyzed:")
            st.dataframe(input_df)
            
            with st.spinner("Analyzing customer data..."):
                prediction, probability = predict_churn(input_df)
                if prediction is not None:
                    churn_prob = probability[0] * 100
                    threshold = bundle['threshold'] * 100
                    
                    st.subheader("🎯 Prediction Result")
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
                    
                    fig, ax = plt.subplots(figsize=(12, 3))
                    colors = ['green'] * 25 + ['yellow'] * 25 + ['orange'] * 25 + ['red'] * 25
                    ax.barh(['Churn Risk'], [churn_prob], color=colors[int(churn_prob)-1] if churn_prob <= 100 else 'red')
                    ax.axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.1f}%)')
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('Churn Probability (%)')
                    ax.set_title('Customer Risk Assessment')
                    ax.legend()
                    st.pyplot(fig)
                    
                    st.markdown("---")
                    st.subheader("🤖 AI-Powered Retention Recommendations")
                    with st.spinner("Generating personalized recommendations..."):
                        # Use the lowercase version for AI insights
                        ai_insights = get_ai_insights(input_df_lower, churn_prob, prediction)
                        st.markdown(f'<div class="ai-insight">{ai_insights}</div>', unsafe_allow_html=True)
    
    elif app_mode == "Model Info":
        st.markdown('<div class="subheader">ℹ️ Model Information</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write("**📋 Features Used:**")
        for feature in feature_config['expected_features']:
            st.write(f"- {feature}")
        st.write(f"**⚖️ Decision Threshold:** {bundle['threshold']:.2f}")
        st.write("**🤖 Model Type:** Calibrated XGBoost")
        st.write("**📊 Feature Default Values (for missing data):**")
        means_df = pd.DataFrame.from_dict(feature_config['numeric_means'], orient='index', columns=['Default Value'])
        st.dataframe(means_df)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()