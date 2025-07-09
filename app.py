import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import re
import io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# --- Core AI and Helper Functions ---

@st.cache_resource
def get_ai_model():
    """Configures and returns the AI model, cached for performance."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
        if not api_key:
            st.error("GEMINI_API_KEY not found. Please set it in your Streamlit secrets.")
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        st.error(f"❌ Failed to configure AI model: {e}")
        return None

def get_ai_response(model, prompt):
    """
    Uses the LLM to generate a conversational response or a command.
    """
    if model is None: return "ERROR: AI model is not configured."
    
    system_prompt = """
    You are a helpful AI assistant. Your job is to determine the user's intent.
    - If the user asks to "train", "build a model", or "run the predictor", respond with the single word: TRAIN
    - If the user asks to "predict", "score new companies", or "use the model", respond with the single word: PREDICT
    - For any other question or greeting, provide a friendly, conversational response.
    """
    try:
        response = model.generate_content([system_prompt, prompt])
        return response.text.strip()
    except Exception as e:
        return f"ERROR: AI generation failed: {e}"

# --- Advanced Data Processing and Modeling (from predictor.py) ---

def clean_and_feature_engineer(df):
    """
    Performs all advanced cleaning and feature engineering from predictor.py.
    """
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    # 1. Clean Headquarters Location -> Only Country
    def get_country(location):
        return str(location).split(',')[-1].strip() if isinstance(location, str) else 'Unknown'
    df['Headquarters Country'] = df['Headquarters Location'].apply(get_country)

    # 2. Create Top Industry
    industry_priority = ['AI', 'Fintech', 'HealthTech', 'F&B & AgriTech', 'DeepTech & IoT', 'MarTech', 'Web3', 'Mobility & Logistics', 'Proptech', 'SaaS', 'EdTech', 'Ecommerce', 'HRTech']
    industry_map = {'Artificial Intelligence (AI)': 'AI', 'AgTech': 'F&B & AgriTech', 'Food and Beverage': 'F&B & AgriTech', 'Internet of Things': 'DeepTech & IoT', 'Logistics': 'Mobility & Logistics', 'E-Commerce': 'Ecommerce', 'Human Resources': 'HRTech', 'PropTech': 'Proptech', 'Edutech': 'EdTech'}
    def get_top_industry(row):
        text = f"{row.get('Industry Groups', '')} {row.get('Industries', '')}"
        for industry in industry_priority:
            if re.search(r'\b' + re.escape(industry) + r'\b', text, re.IGNORECASE): return industry
        for term, top_industry in industry_map.items():
            if term in text: return top_industry
        return 'Other'
    df['Top Industry'] = df.apply(get_top_industry, axis=1)

    # 3. Clean Date Columns -> Year only
    def get_year(date):
        match = re.search(r'\b\d{4}\b', str(date))
        return pd.to_numeric(match.group(0), errors='coerce') if match else pd.NA
    df['Founded Year'] = df['Founded Date'].apply(get_year)
    df['Exit Year'] = df['Exit Date'].apply(get_year)

    # 4. Convert all monetary data to USD
    exchange_rates = {'₹': 0.012, 'INR': 0.012, 'SGD': 0.79, 'A$': 0.66, 'AUD': 0.66, 'MYR': 0.24, 'IDR': 0.000062, '¥': 0.0070, 'JPY': 0.0070, 'CNY': 0.14, '$': 1}
    def convert_to_usd(value):
        if not isinstance(value, str): return pd.NA
        value_cleaned = value.replace(',', '')
        for symbol, rate in exchange_rates.items():
            if symbol in value_cleaned:
                numeric_part = re.search(r'[\d\.]+', value_cleaned)
                if numeric_part: return float(numeric_part.group(0)) * rate
        numeric_part = re.search(r'[\d\.]+', value_cleaned)
        return float(numeric_part.group(0)) if numeric_part else pd.NA
        
    money_cols = ['Last Funding Amount', 'Total Equity Funding Amount', 'Total Funding Amount']
    for col in money_cols:
        df[f"{col} (USD)"] = df[col].apply(convert_to_usd)

    # 5. Create the final 'Exit' column if it doesn't exist
    if 'Exit' not in df.columns:
        df['Exit'] = df.apply(lambda row: 1.0 if pd.notna(row['Exit Year']) or str(row.get('Funding Status')) in ['M&A', 'IPO'] else 0.0, axis=1)
    
    return df

def train_and_score(df_train, df_predict):
    """
    Trains the advanced model on df_train and scores df_predict.
    """
    # Define features for the advanced model
    target = 'Exit'
    numeric_features = ['Founded Year', 'Number of Founders', 'Number of Funding Rounds', 'Total Equity Funding Amount (USD)', 'Total Funding Amount (USD)']
    categorical_features = ['Headquarters Country', 'Top Industry', 'Funding Status', 'Last Funding Type']
    text_features = ['Description', 'Top 5 Investors']

    # Convert pd.NA to np.nan for numeric columns
    for col in numeric_features:
        if col in df_train.columns:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        if col in df_predict.columns:
            df_predict[col] = pd.to_numeric(df_predict[col], errors='coerce')

    # Ensure all feature columns exist, fill missing with placeholders
    for col in numeric_features + categorical_features + text_features:
        if col not in df_train.columns: df_train[col] = 'Unknown' if col in categorical_features or col in text_features else 0
        if col not in df_predict.columns: df_predict[col] = 'Unknown' if col in categorical_features or col in text_features else 0

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    description_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(stop_words='english', max_features=150, ngram_range=(1,2)))])
    investor_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(tokenizer=lambda x: [i.strip() for i in str(x).split(',')], max_features=100))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('desc', description_transformer, 'Description'),
            ('inv', investor_transformer, 'Top 5 Investors')],
        remainder='drop')

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', oob_score=True))
    ])

    # Train the model
    X_train = df_train.drop(columns=[target, 'Organization Name'], errors='ignore')
    y_train = df_train[target]
    model.fit(X_train, y_train)
    accuracy = model.named_steps['classifier'].oob_score_

    # Predict on the new data
    X_predict = df_predict.drop(columns=[target, 'Organization Name'], errors='ignore')
    probabilities = model.predict_proba(X_predict)[:, 1]
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Organization Name': df_predict['Organization Name'],
        'Success Score': (probabilities * 100).round().astype(int)
    }).sort_values(by='Success Score', ascending=False)
    
    st.session_state.trained_model = model # Save the trained model
    st.session_state.trained_on_file = st.session_state.get('training_file_name', 'Unknown')

    message = f"✅ Advanced model trained with an estimated accuracy of {accuracy:.2%}. Here are the scores for the prediction data:"
    return results_df, message

# --- Streamlit App UI and Logic ---

st.set_page_config(layout="wide")
st.title("🚀 Advanced AI Exit Predictor")
st.caption("Powered by a custom data pipeline and machine learning model.")

model = get_ai_model()

# Initialize session state
if "training_data" not in st.session_state: st.session_state.training_data = None
if "prediction_data" not in st.session_state: st.session_state.prediction_data = None
if "messages" not in st.session_state: st.session_state.messages = []
if "trained_model" not in st.session_state: st.session_state.trained_model = None
if "trained_on_file" not in st.session_state: st.session_state.trained_on_file = None

with st.sidebar:
    st.header("1. Upload Training Data")
    train_file = st.file_uploader("Upload your historical data with exit info", type=["xlsx", "csv"])
    if train_file:
        df_raw = pd.read_csv(train_file) if train_file.name.endswith('.csv') else pd.read_excel(train_file)
        st.session_state.training_data = clean_and_feature_engineer(df_raw)
        st.session_state.training_file_name = train_file.name
        st.success(f"Loaded and prepared '{train_file.name}'.")

    st.header("2. Upload Prediction Data")
    predict_file = st.file_uploader("Upload the data you want to score", type=["xlsx", "csv"])
    if predict_file:
        df_raw = pd.read_csv(predict_file) if predict_file.name.endswith('.csv') else pd.read_excel(predict_file)
        st.session_state.prediction_data = clean_and_feature_engineer(df_raw)
        st.success(f"Loaded and prepared '{predict_file.name}'.")

    # --- RESTORED: "Model Trained" status display ---
    st.header("Analysis Status")
    model_status = "Yes" if st.session_state.trained_model else "No"
    trained_file_info = f" (on `{st.session_state.trained_on_file}`)" if st.session_state.trained_on_file else ""
    st.write(f"**Model Trained:** {model_status}{trained_file_info}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if message.get("data") is not None:
            st.dataframe(message["data"])

if prompt := st.chat_input("What would you like to do?"):
    st.session_state.messages.append({"role": "user", "content": prompt, "data": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # --- ROBUST FIX: Initialize variables at the top ---
        response_content, response_data = "An unknown action occurred.", None

        if st.session_state.training_data is None or st.session_state.prediction_data is None:
            st.warning("Please upload both a training and a prediction file first.")
        else:
            with st.spinner("🧠 AI is thinking..."):
                ai_response = get_ai_response(model, prompt)
                
                if ai_response == "TRAIN" or ai_response == "PREDICT":
                    try:
                        results_df, message = train_and_score(st.session_state.training_data, st.session_state.prediction_data)
                        response_content = message
                        response_data = results_df
                        st.markdown(response_content)
                        st.dataframe(response_data)
                    except Exception as e:
                        response_content = f"❌ Error during model training/prediction: {e}"
                        st.error(response_content)
                else: # It's a conversational response
                    response_content = ai_response
                    st.markdown(response_content)
            
        st.session_state.messages.append({"role": "assistant", "content": response_content, "data": response_data})
