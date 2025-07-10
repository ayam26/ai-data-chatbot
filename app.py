import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import re
import io
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px

# --- Suppress specific, harmless warnings from scikit-learn ---
warnings.filterwarnings("ignore", message="Skipping features without any observed values")
warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None")


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

def get_ai_response(model, prompt, df_dict):
    """
    Uses the LLM to generate either a conversational response or a command.
    """
    if model is None: return "ERROR: AI model is not configured."
    
    df_names = list(df_dict.keys())
    context_df = st.session_state.get('prediction_data') or st.session_state.get('training_data') or pd.DataFrame()
    primary_df_columns = list(context_df.columns)

    system_prompt = f"""
    You are a helpful AI assistant. Your job is to determine the user's intent and respond in one of three modes.

    1.  **Visualizer Mode**: If the prompt contains "plot", "chart", "visualize", or "graph", you MUST generate a single line of Python code using `plotly.express as px` and assign it to a variable named `fig`.
    2.  **Data Analyst Mode**: If the prompt contains other data-related keywords like "filter", "sort", "train", or "predict", you MUST generate a single line of Python code to perform that action.
    3.  **Conversational Mode**: For any other question or greeting, provide a friendly, conversational text response.

    **Code Generation Rules:**
    - The output MUST be a single line of code. Do NOT use markdown or comments.
    - You MUST use the exact column names provided in the list below. Do NOT invent or assume column names.
    - The dataframe to use is ALWAYS named `df`.

    **Available Columns:**
    `{primary_df_columns}`

    **Plotting Examples:**
    - User: "plot a bar chart of the total 'Total Funding Amount (USD)' for each 'Last Funding Type'"
    - AI: `fig = px.bar(df.groupby('Last Funding Type')['Total Funding Amount (USD)'].sum().reset_index(), x='Last Funding Type', y='Total Funding Amount (USD)', title='Total Funding by Type')`
    
    **Training/Prediction Examples:**
    - User: "train the model"
    - AI: `message = train_and_score()`
    """
    try:
        response = model.generate_content([system_prompt, prompt])
        return response.text.strip()
    except Exception as e:
        return f"ERROR: AI generation failed: {e}"

# --- Advanced Data Processing and Modeling ---

def clean_and_feature_engineer(df):
    """
    Performs all advanced cleaning and feature engineering.
    """
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    def get_country(location):
        return str(location).split(',')[-1].strip() if isinstance(location, str) else 'Unknown'
    df['Headquarters Country'] = df['Headquarters Location'].apply(get_country)

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

    def get_year(date):
        match = re.search(r'\b\d{4}\b', str(date))
        return pd.to_numeric(match.group(0), errors='coerce') if match else pd.NA
    df['Founded Year'] = df['Founded Date'].apply(get_year)
    df['Exit Year'] = df['Exit Date'].apply(get_year)

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

    if 'Exit' not in df.columns:
        df['Exit'] = df.apply(lambda row: 1.0 if pd.notna(row['Exit Year']) or str(row.get('Funding Status')) in ['M&A', 'IPO'] else 0.0, axis=1)
    
    return df

def train_and_score():
    """
    Trains the advanced model and scores the prediction data.
    """
    df_train = st.session_state.training_data
    df_predict = st.session_state.prediction_data
    
    target = 'Exit'
    numeric_features = ['Founded Year', 'Number of Founders', 'Number of Funding Rounds', 'Total Equity Funding Amount (USD)', 'Total Funding Amount (USD)']
    categorical_features = ['Headquarters Country', 'Top Industry', 'Funding Status', 'Last Funding Type']
    text_features = ['Description', 'Top 5 Investors']

    for col in numeric_features:
        if col in df_train.columns: df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        if col in df_predict.columns: df_predict[col] = pd.to_numeric(df_predict[col], errors='coerce')

    for col in text_features:
        if col in df_train.columns: df_train[col] = df_train[col].fillna('')
        if col in df_predict.columns: df_predict[col] = df_predict[col].fillna('')

    for col in numeric_features + categorical_features + text_features:
        if col not in df_train.columns: df_train[col] = 'Unknown' if col in categorical_features or col in text_features else 0
        if col not in df_predict.columns: df_predict[col] = 'Unknown' if col in categorical_features or col in text_features else 0

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

    X_train = df_train.drop(columns=[target, 'Organization Name'], errors='ignore')
    y_train = df_train[target]
    model.fit(X_train, y_train)
    accuracy = model.named_steps['classifier'].oob_score_

    X_predict = df_predict.drop(columns=[target, 'Organization Name'], errors='ignore')
    probabilities = model.predict_proba(X_predict)[:, 1]
    
    results_df = pd.DataFrame({
        'Organization Name': df_predict['Organization Name'],
        'Success Score': (probabilities * 100).round().astype(int)
    }).sort_values(by='Success Score', ascending=False)
    
    st.session_state.trained_model = model
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

    st.header("Analysis Status")
    model_status = "Yes" if st.session_state.trained_model else "No"
    trained_file_info = f" (on `{st.session_state.trained_on_file}`)" if st.session_state.trained_on_file else ""
    st.write(f"**Model Trained:** {model_status}{trained_file_info}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if message.get("data") is not None:
            st.dataframe(message["data"])
        if message.get("chart") is not None:
            st.plotly_chart(message["chart"], use_container_width=True)

if prompt := st.chat_input("What would you like to do?"):
    st.session_state.messages.append({"role": "user", "content": prompt, "data": None, "chart": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_content, response_data, response_chart = None, None, None

        if st.session_state.training_data is None or st.session_state.prediction_data is None:
            st.warning("Please upload both a training and a prediction file first.")
            response_content = "Please upload both a training and a prediction file first."
        else:
            with st.spinner("🧠 AI is thinking..."):
                # --- FINAL FIX IS HERE: Pass df_dict to the AI response function ---
                ai_response = get_ai_response(model, prompt, st.session_state.df_dict)
                
                cleaned_response = ai_response.strip().strip('`')

                code_keywords = ['fig =', 'train_and_score']
                is_code = any(keyword in cleaned_response for keyword in code_keywords)

                if cleaned_response.startswith("ERROR:"):
                    response_content = f"❌ {cleaned_response}"
                    st.error(response_content)
                elif is_code:
                    st.code(cleaned_response, language="python")
                    # Determine which dataframe to use for the operation
                    df_for_exec = st.session_state.prediction_data if "plot" in prompt or "chart" in prompt else st.session_state.training_data
                    local_vars = {"df": df_for_exec, "px": px, "train_and_score": train_and_score}
                    
                    try:
                        exec(cleaned_response, globals(), local_vars)
                        
                        if local_vars.get("fig") is not None:
                            response_chart = local_vars["fig"]
                            response_content = f"✅ Here is your interactive chart for '{prompt}'"
                        elif 'results_df' in local_vars:
                            results_df, message = local_vars["results_df"], local_vars["message"]
                            response_content = message
                            response_data = results_df
                        else:
                            response_content = "✅ Command executed." # Fallback message

                        st.markdown(response_content)
                        if response_data is not None:
                            st.dataframe(response_data)
                        if response_chart is not None:
                            st.plotly_chart(response_chart, use_container_width=True)

                    except Exception as e:
                        response_content = f"❌ Error executing code: {e}"
                        st.error(response_content)
                else: # It's a conversational response
                    response_content = cleaned_response
                    st.markdown(response_content)
            
        st.session_state.messages.append({"role": "assistant", "content": response_content, "data": response_data, "chart": response_chart})
