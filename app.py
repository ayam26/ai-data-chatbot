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
import plotly.graph_objects as go

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
    You are an expert data analysis AI. Your job is to translate natural language into a single, executable line of Python code. You operate in one of six modes.

    1.  **Correlation Heatmap Mode**: If the prompt asks for "correlation" or "heatmap", call `plot_correlation_heatmap()`.
    2.  **Comparison Mode**: If the prompt asks to "compare means" or "compare distributions" for exited vs. non-exited companies, call `plot_comparison_boxplot(y_col='column_name')`, extracting the column name to compare.
    3.  **Interaction Scatter Mode**: If the prompt asks to see the "interaction between" two variables, call `plot_interactive_scatter(x_col='col1', y_col='col2')`, extracting the two column names.
    4.  **Training Mode**: If the prompt is to "train" or "predict", generate `results_df, message = train_and_score()`.
    5.  **Driver Analysis Mode**: If the prompt asks to "explain the model" or "find drivers", generate `fig = get_feature_importance_plot()`.
    6.  **Conversational Mode**: For anything else, provide a friendly text response.

    **Rules:**
    - The output MUST be a single line of Python code, or a conversational response.
    - Use the exact column names provided: `{df_columns}`.

    **Advanced Analysis Examples:**
    - User: "Show me the correlation heatmap of financial metrics." -> AI: `fig = plot_correlation_heatmap()`
    - User: "Compare the Total Funding Amount for exited vs non-exited companies." -> AI: `fig = plot_comparison_boxplot(y_col='Total Funding Amount (USD)')`
    - User: "Plot the interaction between Founded Year and Total Funding." -> AI: `fig = plot_interactive_scatter(x_col='Founded Year', y_col='Total Funding Amount (USD)')`
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
    """
    try:
        response = _model.generate_content(prompt)
        json_string = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_string)
    except (json.JSONDecodeError, Exception) as e:
        st.warning(f"AI column mapping failed: {e}. Please map columns manually.")
        return {"TARGET_VARIABLE": "N/A", "ORGANIZATION_IDENTIFIER": "N/A", "TEXT_DESCRIPTION": "N/A", "CATEGORICAL_INDUSTRY": "N/A"}

# --- Data Processing and Modeling ---

def clean_data(df):
    """Performs essential, universal data cleaning."""
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df

def train_and_score():
    """Dynamically trains the model and saves it to session state."""
    df_train = st.session_state.training_data.copy()
    mapping = st.session_state.column_mapping
    target = mapping['TARGET_VARIABLE']

    if target == "N/A" or target not in df_train.columns:
        return None, "ERROR: A valid 'Target Variable' must be selected."

    # --- FIX: More robust feature type identification ---
    special_cols = list(mapping.values())
    
    # Attempt to convert object columns to numeric where possible
    for col in df_train.columns:
        if col in special_cols or col == target:
            continue
        if df_train[col].dtype == 'object':
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')

    # Now, identify types based on the cleaned dtypes
    numeric_features = df_train.select_dtypes(include=np.number).columns.drop(target, errors='ignore').tolist()
    object_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_features = [c for c in object_cols if c not in special_cols]
    text_features = [mapping['TEXT_DESCRIPTION']] if mapping['TEXT_DESCRIPTION'] != "N/A" and mapping['TEXT_DESCRIPTION'] in df_train.columns else []
    
    # Ensure categorical columns are uniformly strings to prevent encoder errors
    for col in categorical_features:
        df_train[col] = df_train[col].astype(str).fillna('Unknown')

    st.session_state.model_features = {"numeric": numeric_features, "categorical": categorical_features, "text": text_features}
    st.info(f"**Model Features Identified:**\n- **Numeric:** {numeric_features}\n- **Categorical:** {categorical_features}\n- **Text:** {text_features}")

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    text_transformers = [(f'text_{col}', TfidfVectorizer(stop_words='english', max_features=50, ngram_range=(1,2)), col) for col in text_features]

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)] + text_transformers, remainder='drop')
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', oob_score=True))])

    X_train = df_train.drop(columns=[target], errors='ignore')
    y_train = df_train[target]
    model.fit(X_train, y_train)
    st.session_state.trained_model = model

    accuracy = model.named_steps['classifier'].oob_score_
    message = f"✅ Model trained with an estimated accuracy of {accuracy:.2%}. You can now score the prediction data or analyze feature importance."
    
    if st.session_state.prediction_data is not None:
        df_predict = st.session_state.prediction_data.copy()
        # Apply the same data type conversions to prediction data
        for col in df_predict.columns:
            if df_predict[col].dtype == 'object':
                df_predict[col] = pd.to_numeric(df_predict[col], errors='coerce')
        for col in categorical_features:
             if col in df_predict.columns:
                df_predict[col] = df_predict[col].astype(str).fillna('Unknown')

        X_predict = df_predict.drop(columns=[target], errors='ignore')
        probabilities = model.predict_proba(X_predict)[:, 1]
        org_name_col = mapping['ORGANIZATION_IDENTIFIER']
        results_df = pd.DataFrame({'Organization Name': df_predict[org_name_col] if org_name_col != "N/A" else df_predict.index, 'Success Score': (probabilities * 100).round().astype(int)}).sort_values(by='Success Score', ascending=False)
        return results_df, message
    else:
        return None, message

# --- Advanced Analysis Functions ---

def get_feature_importance_plot():
    """Extracts feature importances from the trained model and returns a plot."""
    if 'trained_model' not in st.session_state or st.session_state.trained_model is None:
        st.error("You must train a model first.")
        return None
    model, features = st.session_state.trained_model, st.session_state.model_features
    preprocessor, classifier = model.named_steps['preprocessor'], model.named_steps['classifier']
    
    try: onehot_cols = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(features['categorical'])
    except: onehot_cols = []
    text_cols = []
    for col_name in features['text']:
        try: text_cols.extend(preprocessor.named_transformers_[f'text_{col_name}'].named_steps['tfidf'].get_feature_names_out())
        except: continue
    
    feature_names = np.concatenate([features['numeric'], onehot_cols, text_cols])
    importances = classifier.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(20)
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Top 20 Drivers of Successful Exits')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def plot_correlation_heatmap():
    """Calculates and plots a heatmap of correlations for numeric features."""
    df = st.session_state.training_data
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
    fig.update_layout(title='Correlation Between Financial Metrics and Exits')
    return fig

def plot_comparison_boxplot(y_col):
    """Compares the distribution of a numeric column for exited vs. non-exited companies."""
    df = st.session_state.training_data
    target = st.session_state.column_mapping['TARGET_VARIABLE']
    fig = px.box(df, x=target, y=y_col, title=f'Comparison of {y_col} for Exited vs. Non-Exited Companies')
    return fig

def plot_interactive_scatter(x_col, y_col):
    """Creates a scatter plot to show the interaction between two variables, colored by exit type."""
    df = st.session_state.training_data
    target = st.session_state.column_mapping['TARGET_VARIABLE']
    fig = px.scatter(df, x=x_col, y=y_col, color=target, title=f'Interaction between {x_col} and {y_col}')
    return fig

# --- Streamlit App UI and Logic ---

st.set_page_config(layout="wide")
st.title("🚀 Highly Adaptable AI Exit Predictor")
st.caption("With AI-powered column mapping and advanced driver analysis.")

# Initialize session state
if "messages" not in st.session_state: st.session_state.messages = []
if "training_data" not in st.session_state: st.session_state.training_data = None
if "prediction_data" not in st.session_state: st.session_state.prediction_data = None
if "column_mapping" not in st.session_state: st.session_state.column_mapping = None
if "trained_model" not in st.session_state: st.session_state.trained_model = None

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
        st.info("Our AI suggests roles for your columns. Please confirm or correct them.")
        all_cols = ["N/A"] + st.session_state.training_data.columns.tolist()
        if st.session_state.column_mapping is None:
            model = get_ai_model()
            st.session_state.column_mapping = get_column_mapping(model, all_cols[1:])
        mapping = st.session_state.column_mapping
        def get_index(key): return all_cols.index(mapping.get(key, "N/A")) if mapping.get(key) in all_cols else 0
        mapping['TARGET_VARIABLE'] = st.selectbox("Target Variable", all_cols, index=get_index('TARGET_VARIABLE'))
        mapping['ORGANIZATION_IDENTIFIER'] = st.selectbox("Organization Identifier", all_cols, index=get_index('ORGANIZATION_IDENTIFIER'))
        mapping['TEXT_DESCRIPTION'] = st.selectbox("Text Description", all_cols, index=get_index('TEXT_DESCRIPTION'))
        mapping['CATEGORICAL_INDUSTRY'] = st.selectbox("Categorical Industry", all_cols, index=get_index('CATEGORICAL_INDUSTRY'))

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
        if st.session_state.training_data is None:
            st.warning("Please upload at least a training file first.")
            st.stop()
        
        with st.spinner("🧠 AI is thinking..."):
            model = get_ai_model()
            context_df = st.session_state.training_data # Most analysis is on training data
            ai_response = get_ai_response(model, prompt, list(context_df.columns))
            
            cleaned_response = ai_response.replace("```python", "").replace("```", "").strip()

            code_keywords = ['fig =', 'train_and_score', 'df =']
            is_code = any(keyword in cleaned_response for keyword in code_keywords)

            if cleaned_response.startswith("ERROR:"):
                st.error(cleaned_response)
                st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
            elif is_code:
                st.code(cleaned_response, language="python")
                local_vars = {
                    "df": context_df.copy(), "px": px, "go": go,
                    "train_and_score": train_and_score, 
                    "get_feature_importance_plot": get_feature_importance_plot,
                    "plot_correlation_heatmap": plot_correlation_heatmap,
                    "plot_comparison_boxplot": plot_comparison_boxplot,
                    "plot_interactive_scatter": plot_interactive_scatter
                }
                
                try:
                    exec(cleaned_response, globals(), local_vars)
                    response_content, response_data, response_chart = "✅ Command executed.", None, None
                    
                    if local_vars.get("fig") is not None:
                        response_chart = local_vars["fig"]
                        response_content = f"✅ Here is your analysis for '{prompt}'"
                    elif 'results_df' in local_vars:
                        response_data, response_content = local_vars["results_df"], local_vars["message"]
                    
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
