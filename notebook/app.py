import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load env file
load_dotenv(".env")

# Read key
hf_api_key = os.getenv("HF_API_KEY")

# Init client
client = InferenceClient(api_key=hf_api_key)

# Define the DataSanitizer class
class DataSanitizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_config):
        self.feature_config = feature_config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.rename(columns=lambda x: x.strip().lower())
        expected = [col.lower() for col in self.feature_config['expected_features']]
        missing = [f for f in expected if f not in X.columns]
        for col in missing:
            X[col] = self.feature_config['numeric_means'][col]
        X = X[expected]
        for col in expected:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(self.feature_config['numeric_means'][col])
        return X

# Helper function
def get_feature_value(customer_data, feature_name):
    lower_cols = [col.lower() for col in customer_data.columns]
    if feature_name.lower() in lower_cols:
        actual_col = customer_data.columns[lower_cols.index(feature_name.lower())]
        return customer_data[actual_col].iloc[0]
    return "N/A"

# Load model
@st.cache_resource
def load_model():
    try:
        try:
            bundle = joblib.load("model/churn_model.pkl")
            return bundle
        except:
            bundle = joblib.load("C:\\Users\\USER\\Documents\\github_repo\\ecommerce-churn-predictor\\model\\churn_model.pkl")
            return bundle
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Prediction
def predict_churn(new_data):
    try:
        bundle = load_model()
        pipeline = bundle['pipeline']
        threshold = bundle['threshold']
        proba = pipeline.predict_proba(new_data)[:, 1]
        predictions = (proba >= threshold).astype(int)
        return predictions, proba
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# File reader
def read_uploaded_file(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext == '.csv':
            return pd.read_csv(uploaded_file)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

# AI insights - Updated function
def get_ai_insights(customer_data, churn_prob, prediction):
    try:
        # Extract values using helper function
        tenure = get_feature_value(customer_data, 'Tenure')
        addresses = get_feature_value(customer_data, 'NumberOfAddress')
        cashback = get_feature_value(customer_data, 'CashbackAmount')
        last_order = get_feature_value(customer_data, 'DaySinceLastOrder')
        order_count = get_feature_value(customer_data, 'OrderCount')
        satisfaction = get_feature_value(customer_data, 'SatisfactionScore')
        
        prompt = f"""
        As a customer retention expert, analyze this customer data:

        Customer Profile:
        - Tenure: {tenure} months
        - Number of Addresses: {addresses}
        - Cashback Amount: ${cashback}
        - Days Since Last Order: {last_order}
        - Order Count: {order_count}
        - Satisfaction Score: {satisfaction}/5

        Churn Risk: {churn_prob:.1f}% ({'High Risk' if prediction[0] == 1 else 'Low Risk'})

        Please provide:
        1. Brief analysis of why this customer might be at risk
        2. 3-5 specific recommendations to retain them
        3. Immediate actions to take

        Format with clear sections and bullet points.
        """

        # Try using conversational endpoint with supported models
        try:
            # Try a model that definitely supports conversational task
            response = client.chat_completion(
                model="HuggingFaceH4/zephyr-7b-beta",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5000,
                temperature=0.7
            )
            return response.choices[0].message.content
            
        except Exception as chat_error:
            st.warning(f"Chat completion failed: {chat_error}. Trying text generation...")
            
            # Try text generation with a supported model
            response = client.text_generation(
                prompt=prompt,
                model="mistralai/Mistral-7B-Instruct-v0.2",  # This model supports text-generation
                max_new_tokens=400,
                temperature=0.7
            )
            return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ AI Insights unavailable. Error: {str(e)}"

# Main app
def main():
    bundle = load_model()
    feature_config = bundle['feature_config']

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose mode:", ["Batch Prediction", "Model Info"])

    st.title("📊 Customer Churn Prediction")

    if app_mode == "Batch Prediction":
        st.header("📁 Batch Prediction")
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

        if uploaded_file is not None:
            df = read_uploaded_file(uploaded_file)
            if df is not None:
                st.success("✅ File uploaded successfully!")
                st.info(f"Expected columns: {', '.join(feature_config['expected_features'])}")
                st.dataframe(df.head(3))

                if st.button("🔮 Predict Churn"):
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
                        col1.metric("Total Customers", total_customers)
                        col2.metric("At Risk", f"{churned_count} ({churn_rate:.1f}%)")
                        col3.metric("Avg Risk Score", f"{np.mean(probabilities)*100:.1f}%")

                        high_risk_customers = np.where(predictions == 1)[0]
                        for idx in high_risk_customers[:3]:
                            customer = df.iloc[[idx]]
                            prob = probabilities[idx] * 100
                            ai_insights = get_ai_insights(customer, prob, [predictions[idx]])
                            st.markdown(f"### Customer {idx+1}")
                            st.markdown(ai_insights)

                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name='churn_predictions.csv',
                            mime='text/csv'
                        )

    elif app_mode == "Model Info":
        st.header("ℹ️ Model Information")
        st.write("**Features Used:**")
        for feature in feature_config['expected_features']:
            st.write(f"- {feature}")
        st.write(f"**Decision Threshold:** {bundle['threshold']:.2f}")
        st.write("**Model Type:** Calibrated XGBoost")
        st.write("**Default Values:**")
        st.dataframe(pd.DataFrame.from_dict(feature_config['numeric_means'], orient='index', columns=['Default Value']))

if __name__ == "__main__":
    main()
