import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import re
import io
import json
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px

# --- Suppress specific, harmless warnings ---
warnings.filterwarnings("ignore", message="Skipping features without any observed values")
warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used")

# --- Core AI and Helper Functions ---

@st.cache_resource
def get_ai_model():
    """Configures and returns the AI model, cached for performance."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY not found. Please set it in your Streamlit secrets.")
            st.stop()
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"❌ Failed to configure AI model: {e}")
        st.stop()
        return None

def get_ai_response(model, prompt, df_columns):
    """Uses the LLM to generate a command based on user intent."""
    if model is None: return "ERROR: AI model is not configured."
    system_prompt = f"""
    You are an expert data analysis AI. Your job is to translate natural language into a single, executable line of Python code. You operate in one of four modes.

    1.  **Visualizer Mode**: If the prompt is to "plot", "chart", "visualize", or "graph", generate a `plotly.express` command and assign it to a variable named `fig`.
    2.  **Training Mode**: If the prompt is to "train", "predict", or "run model", generate the specific command `results_df, message = train_and_score()`.
    3.  **Feature Engineering Mode**: If the prompt is to "create", "make", or "generate" a new column/variable, generate the pandas command to create it and assign the modified dataframe back to `df`.
    4.  **Conversational Mode**: For anything else, provide a friendly text response without generating code.

    **Rules:**
    - The output MUST be a single line of Python code, or a conversational response.
    - Use the exact column names provided: `{df_columns}`.
    - The dataframe is ALWAYS named `df`.

    **Feature Engineering Example:**
    - User: "Create a new column 'Age' from 2025 minus 'Founded Year'"
    - AI: `df = df.assign(Age=2025 - df['Founded Year'])`
    """
    try:
        response = model.generate_content([system_prompt, prompt])
        return response.text.strip()
    except Exception as e:
        return f"ERROR: AI generation failed: {e}"

@st.cache_data
def get_column_mapping(_model, columns):
    """Uses AI to map columns to required roles for VC prediction."""
    prompt = f"""
    Analyze the following list of column names from a venture capital dataset: {columns}.
    Your task is to identify the columns that best correspond to these four essential roles:
    1.  `TARGET_VARIABLE`: The column to predict. This is likely named 'Exit', 'Status', or 'Is Exited'.
    2.  `ORGANIZATION_IDENTIFIER`: The name of the company. Likely 'Organization Name', 'Company Name', etc.
    3.  `TEXT_DESCRIPTION`: A long text description of the company. Likely 'Description', 'Overview', etc.
    4.  `CATEGORICAL_INDUSTRY`: The primary industry category. Likely 'Industry', 'Top Industry', 'Vertical', etc.

    Return your answer as a JSON object with these four keys. For each key, the value should be your best guess for the column name from the list. If you cannot find a suitable column for a role, use the value "N/A".

    Example:
    Input: ['Company Name', 'Description', 'Exit', 'Industry', 'Total Funding']
    Output: {{"TARGET_VARIABLE": "Exit", "ORGANIZATION_IDENTIFIER": "Company Name", "TEXT_DESCRIPTION": "Description", "CATEGORICAL_INDUSTRY": "Industry"}}
    """
    try:
        response = _model.generate_content(prompt)
        # Clean the response to ensure it's valid JSON
        json_string = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_string)
    except (json.JSONDecodeError, Exception) as e:
        st.warning(f"AI column mapping failed: {e}. Please map columns manually.")
        return {
            "TARGET_VARIABLE": "N/A", "ORGANIZATION_IDENTIFIER": "N/A",
            "TEXT_DESCRIPTION": "N/A", "CATEGORICAL_INDUSTRY": "N/A"
        }

# --- Data Processing and Modeling ---

def clean_data(df):
    """Performs essential, universal data cleaning."""
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df

def train_and_score():
    """Dynamically trains the model based on user-confirmed column mappings."""
    df_train = st.session_state.training_data.copy()
    df_predict = st.session_state.prediction_data.copy()
    mapping = st.session_state.column_mapping

    target = mapping['TARGET_VARIABLE']
    if target == "N/A" or target not in df_train.columns:
        return None, "ERROR: A valid 'Target Variable' must be selected from your training data columns."

    # Dynamically identify feature types, excluding mapped special roles
    special_cols = list(mapping.values())
    numeric_features = df_train.select_dtypes(include=np.number).columns.drop(target, errors='ignore').tolist()
    
    object_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_features = [c for c in object_cols if c not in special_cols]
    
    # Use the mapped text description column
    text_features = [mapping['TEXT_DESCRIPTION']] if mapping['TEXT_DESCRIPTION'] != "N/A" else []
    
    st.info(f"**Model Features Identified:**\n- **Numeric:** {numeric_features}\n- **Categorical:** {categorical_features}\n- **Text:** {text_features}")

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    text_transformers = [(f'text_{col}', TfidfVectorizer(stop_words='english', max_features=50, ngram_range=(1,2)), col) for col in text_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)] + text_transformers,
        remainder='drop')

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', oob_score=True))
    ])

    X_train = df_train.drop(columns=[target], errors='ignore')
    y_train = df_train[target]
    model.fit(X_train, y_train)
    accuracy = model.named_steps['classifier'].oob_score_

    X_predict = df_predict.drop(columns=[target], errors='ignore')
    probabilities = model.predict_proba(X_predict)[:, 1]
    
    org_name_col = mapping['ORGANIZATION_IDENTIFIER']
    results_df = pd.DataFrame({
        'Organization Name': df_predict[org_name_col] if org_name_col != "N/A" else df_predict.index,
        'Success Score': (probabilities * 100).round().astype(int)
    }).sort_values(by='Success Score', ascending=False)
    
    message = f"✅ Model trained with an estimated accuracy of {accuracy:.2%}. Here are the scores:"
    return results_df, message

# --- Streamlit App UI and Logic ---

st.set_page_config(layout="wide")
st.title("🚀 Highly Adaptable AI Exit Predictor")
st.caption("With AI-powered column mapping and feature engineering.")

# Initialize session state
if "messages" not in st.session_state: st.session_state.messages = []
if "training_data" not in st.session_state: st.session_state.training_data = None
if "prediction_data" not in st.session_state: st.session_state.prediction_data = None
if "column_mapping" not in st.session_state: st.session_state.column_mapping = None

with st.sidebar:
    st.header("1. Upload Data")
    train_file = st.file_uploader("Upload Training Data", type=["xlsx", "csv"])
    if train_file:
        df_raw = pd.read_csv(train_file) if train_file.name.endswith('.csv') else pd.read_excel(train_file)
        st.session_state.training_data = clean_data(df_raw)
        st.success(f"Loaded '{train_file.name}'.")

    predict_file = st.file_uploader("Upload Prediction Data", type=["xlsx", "csv"])
    if predict_file:
        df_raw = pd.read_csv(predict_file) if predict_file.name.endswith('.csv') else pd.read_excel(predict_file)
        st.session_state.prediction_data = clean_data(df_raw)
        st.success(f"Loaded '{predict_file.name}'.")

    if st.session_state.training_data is not None:
        st.header("2. Confirm Column Roles")
        st.info("Our AI has suggested roles for your columns. Please confirm or correct them.")
        
        all_cols = ["N/A"] + st.session_state.training_data.columns.tolist()
        
        if st.session_state.column_mapping is None:
            model = get_ai_model()
            st.session_state.column_mapping = get_column_mapping(model, all_cols[1:])

        mapping = st.session_state.column_mapping
        
        def get_index(key):
            return all_cols.index(mapping[key]) if mapping[key] in all_cols else 0

        mapping['TARGET_VARIABLE'] = st.selectbox("Target Variable (what to predict)", all_cols, index=get_index('TARGET_VARIABLE'))
        mapping['ORGANIZATION_IDENTIFIER'] = st.selectbox("Organization Identifier (company name)", all_cols, index=get_index('ORGANIZATION_IDENTIFIER'))
        mapping['TEXT_DESCRIPTION'] = st.selectbox("Text Description Column", all_cols, index=get_index('TEXT_DESCRIPTION'))
        mapping['CATEGORICAL_INDUSTRY'] = st.selectbox("Categorical Industry Column", all_cols, index=get_index('CATEGORICAL_INDUSTRY'))

# --- Main chat interface ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("data") is not None: st.dataframe(message["data"])
        if message.get("chart") is not None: st.plotly_chart(message["chart"], use_container_width=True, key=f"history_chart_{i}")

if prompt := st.chat_input("What would you like to do?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.training_data is None or st.session_state.prediction_data is None:
            st.warning("Please upload both a training and a prediction file first.")
            st.stop()
        
        with st.spinner("🧠 AI is thinking..."):
            model = get_ai_model()
            context_df = st.session_state.prediction_data if "plot" in prompt or "show" in prompt else st.session_state.training_data
            ai_response = get_ai_response(model, prompt, list(context_df.columns))
            cleaned_response = ai_response.strip().strip('`').strip()

            code_keywords = ['fig =', 'train_and_score', 'df =']
            is_code = any(keyword in cleaned_response for keyword in code_keywords)

            if cleaned_response.startswith("ERROR:"):
                st.error(cleaned_response)
                st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
            elif is_code:
                st.code(cleaned_response, language="python")
                local_vars = {"df": context_df.copy(), "px": px, "train_and_score": train_and_score}
                
                try:
                    exec(cleaned_response, globals(), local_vars)
                    response_content, response_data, response_chart = "✅ Command executed.", None, None
                    
                    if local_vars.get("fig") is not None:
                        response_chart = local_vars["fig"]
                        response_content = f"✅ Here is your interactive chart for '{prompt}'"
                    elif 'results_df' in local_vars:
                        response_data, response_content = local_vars["results_df"], local_vars["message"]
                    elif 'df' in local_vars and not context_df.equals(local_vars['df']):
                        response_content = f"✅ Feature engineering successful. Data updated."
                        if "plot" in prompt or "show" in prompt: st.session_state.prediction_data = local_vars['df']
                        else: st.session_state.training_data = local_vars['df']
                        st.dataframe(local_vars['df'].head())

                    st.markdown(response_content)
                    if response_data is not None: st.dataframe(response_data)
                    if response_chart is not None: st.plotly_chart(response_chart, use_container_width=True, key="new_chart")
                    st.session_state.messages.append({"role": "assistant", "content": response_content, "data": response_data, "chart": response_chart})
                
                except Exception as e:
                    error_message = f"❌ Error executing code: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
            else: 
                st.markdown(cleaned_response)
                st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
