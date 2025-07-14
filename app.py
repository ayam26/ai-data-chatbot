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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
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
    You are an expert data analysis AI. Your job is to translate natural language into a single, executable line of Python code. You operate in several modes.

    **Modeling & Analysis:**
    - To "train model" or "predict", generate `results_df, message = train_and_score()`.
    - To "find drivers" or "explain the model", generate `fig = get_feature_importance_plot()`.
    - To "show correlation" or "heatmap", call `fig = plot_correlation_heatmap()`.
    - To "compare means" or "compare distributions", call `fig = plot_comparison_boxplot(y_col='column_name')`.
    - To see the "interaction between" two variables, call `fig = plot_interactive_scatter(x_col='col1', y_col='col2')`.

    **Conversational Mode:**
    - For anything else, provide a friendly text response.

    **Rules:**
    - The output MUST be a single line of Python code, or a conversational response.
    - Use the exact column names provided: `{df_columns}`.
    - The dataframe is ALWAYS named `df`.

    **Examples:**
    - User: "Compare the Total Funding Amount (USD) for exited vs non-exited companies." -> AI: `fig = plot_comparison_boxplot(y_col='Total Funding Amount (USD)')`
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
def convert_to_usd(value):
    """A robust function to convert a single currency string to a float."""
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float)): return float(value)
    if not isinstance(value, str): return np.nan

    exchange_rates = {'₹': 0.012, 'INR': 0.012, 'SGD': 0.79, 'A$': 0.66, 'AUD': 0.66, 'MYR': 0.24, 'IDR': 0.000062, '¥': 0.0070, 'JPY': 0.0070, 'CNY': 0.14, '€': 1.08, 'EUR': 1.08, '$': 1.0}
    
    value_cleaned = value.strip().replace(',', '')
    
    if value_cleaned in ['-', '—', 'Undisclosed']: return np.nan
    
    # --- FIX: Handle ranges with "to" or "-" ---
    if ' to ' in value_cleaned.lower():
        value_cleaned = value_cleaned.lower().split(' to ')[0].strip()
    elif '-' in value_cleaned:
        value_cleaned = value_cleaned.split('-')[0].strip()

    multiplier = 1.0
    if 'B' in value_cleaned.upper(): multiplier = 1_000_000_000
    elif 'M' in value_cleaned.upper(): multiplier = 1_000_000
    elif 'K' in value_cleaned.upper(): multiplier = 1_000
    
    numeric_part_str = re.sub(r'[^\d\.]', '', value_cleaned)
    if not numeric_part_str: return np.nan
    
    try: numeric_value = float(numeric_part_str)
    except (ValueError, TypeError): return np.nan
    
    rate = 1.0
    for symbol, r in exchange_rates.items():
        if symbol != '$' and symbol in value_cleaned:
            rate = r
            break
    
    return numeric_value * multiplier * rate

def full_data_prep(df):
    """
    Loads a dataframe and performs all cleaning and feature engineering.
    """
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    # 1. Clean Headquarters Location
    if 'Headquarters Location' in df.columns:
        df['Headquarters Country'] = df['Headquarters Location'].apply(lambda loc: loc.split(',')[-1].strip() if isinstance(loc, str) else 'Unknown')

    # 2. Create Top Industry
    industry_priority = ['AI', 'Fintech', 'HealthTech', 'F&B & AgriTech', 'DeepTech & IoT', 'MarTech', 'Web3', 'Mobility & Logistics', 'Proptech', 'SaaS', 'EdTech', 'Ecommerce', 'HRTech']
    industry_map = {'Artificial Intelligence (AI)': 'AI', 'FinTech': 'Fintech', 'AgTech': 'F&B & AgriTech', 'Food and Beverage': 'F&B & AgriTech', 'Internet of Things': 'DeepTech & IoT', 'Logistics': 'Mobility & Logistics', 'E-Commerce': 'Ecommerce', 'Human Resources': 'HRTech', 'PropTech': 'Proptech', 'Edutech': 'EdTech'}
    def get_top_industry(row):
        combined_industries = f"{row.get('Industry Groups', '')} {row.get('Industries', '')}"
        for industry in industry_priority:
            if re.search(r'\b' + re.escape(industry) + r'\b', combined_industries, re.IGNORECASE): return industry
        for term, top_industry in industry_map.items():
            if term in combined_industries: return top_industry
        return 'Other'
    df['Top Industry'] = df.apply(get_top_industry, axis=1)

    # 3. Clean Date Columns
    if 'Founded Date' in df.columns:
        df['Founded Year'] = pd.to_numeric(df['Founded Date'].astype(str).str.extract(r'(\d{4})'), errors='coerce')
    if 'Exit Date' in df.columns:
        df['Exit Year'] = pd.to_numeric(df['Exit Date'].astype(str).str.extract(r'(\d{4})'), errors='coerce')

    # 4. Convert all monetary data to USD
    money_cols = ['Last Funding Amount', 'Total Equity Funding Amount', 'Total Funding Amount']
    for col in money_cols:
        if col in df.columns:
            new_col_name = f"{col} (USD)"
            df[new_col_name] = df[col].apply(convert_to_usd)
            df[new_col_name] = df[new_col_name].fillna(0)

    # 5. Create the final 'Exit' column
    if 'Exit' not in df.columns and 'Exit Year' in df.columns and 'Funding Status' in df.columns:
        df['Exit'] = df.apply(lambda row: 1.0 if pd.notna(row['Exit Year']) or row['Funding Status'] in ['M&A', 'IPO'] else 0.0, axis=1)
    
    return df

def train_and_score():
    """Trains the model using the pre-defined robust feature set."""
    df_train = st.session_state.training_data.copy()
    
    if 'column_mapping' not in st.session_state or st.session_state.column_mapping is None:
        model = get_ai_model()
        st.session_state.column_mapping = get_column_mapping(model, df_train.columns.tolist())

    mapping = st.session_state.column_mapping
    target = mapping['TARGET_VARIABLE']

    if target == "N/A" or target not in df_train.columns:
        return None, "ERROR: A valid 'Target Variable' must be identified or selected."

    numeric_features = ['Founded Year', 'Number of Founders', 'Number of Funding Rounds', 'Total Equity Funding Amount (USD)', 'Total Funding Amount (USD)']
    categorical_features = ['Headquarters Country', 'Top Industry', 'Funding Status', 'Last Funding Type']
    text_features = ['Description', 'Top 5 Investors']

    for col in numeric_features:
        if col in df_train.columns: df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        else: df_train[col] = np.nan
    for col in categorical_features:
        if col in df_train.columns: df_train[col] = df_train[col].astype(str).fillna('Unknown')
        else: df_train[col] = 'Unknown'
    for col in text_features:
        if col in df_train.columns: df_train[col] = df_train[col].astype(str).fillna('')
        else: df_train[col] = ''

    st.session_state.model_features = {"numeric": numeric_features, "categorical": categorical_features, "text": text_features}
    st.info(f"**Model Features Identified:**\n- **Numeric:** {numeric_features}\n- **Categorical:** {categorical_features}\n- **Text:** {text_features}")

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    text_transformers = [
        ('text_Description', TfidfVectorizer(stop_words='english', max_features=50, ngram_range=(1,2)), 'Description'),
        ('text_Investors', TfidfVectorizer(stop_words='english', max_features=50, ngram_range=(1,2)), 'Top 5 Investors')
    ]

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
        for col in numeric_features:
            if col in df_predict.columns: df_predict[col] = pd.to_numeric(df_predict[col], errors='coerce')
            else: df_predict[col] = np.nan
        for col in categorical_features:
             if col in df_predict.columns: df_predict[col] = df_predict[col].astype(str).fillna('Unknown')
             else: df_predict[col] = 'Unknown'
        for col in text_features:
            if col in df_predict.columns: df_predict[col] = df_predict[col].astype(str).fillna('')
            else: df_predict[col] = ''
        X_predict = df_predict.drop(columns=[target], errors='ignore')
        probabilities = model.predict_proba(X_predict)[:, 1]
        org_name_col = mapping['ORGANIZATION_IDENTIFIER']
        results_df = pd.DataFrame({'Organization Name': df_predict[org_name_col] if org_name_col != "N/A" and org_name_col in df_predict.columns else df_predict.index, 'Success Score': (probabilities * 100).round().astype(int)}).sort_values(by='Success Score', ascending=False)
        return results_df, message
    else:
        return None, message

# --- Advanced Analysis Functions ---

def get_feature_importance_plot():
    """Extracts feature importances from the trained model and returns a plot."""
    if 'trained_model' not in st.session_state or st.session_state.trained_model is None:
        st.error("You must train a model first.")
        return None
    model = st.session_state.trained_model
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        st.error(f"Could not get feature names from the model preprocessor: {e}")
        return None
    importances = classifier.feature_importances_
    if len(feature_names) != len(importances):
        st.error(f"Feature name count ({len(feature_names)}) does not match importance value count ({len(importances)}).")
        return None
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    
    importance_df['Feature'] = importance_df['Feature'].str.replace('num__', '')
    importance_df['Feature'] = importance_df['Feature'].str.replace('cat__', '')
    importance_df['Feature'] = importance_df['Feature'].str.replace('text_Description__', 'desc_')
    importance_df['Feature'] = importance_df['Feature'].str.replace('text_Investors__', 'investor_')
    
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)
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
    if 'column_mapping' not in st.session_state or st.session_state.column_mapping['TARGET_VARIABLE'] == 'N/A':
        st.error("Please ensure a Target Variable is identified in the sidebar before plotting.")
        return None
    if y_col not in df.columns:
        st.error(f"Column '{y_col}' not found in the data. Available numeric columns: {df.select_dtypes(include=np.number).columns.tolist()}")
        return None
    target = st.session_state.column_mapping['TARGET_VARIABLE']
    
    # --- NEW: Pre-plot diagnostic ---
    st.subheader("Pre-Plot Diagnostic")
    agg_data = df.groupby(target)[y_col].describe()
    st.dataframe(agg_data)
    if df[y_col].sum() == 0:
        st.error("Plotting failed because all values in the selected column are zero. This indicates a data cleaning failure.")
        return None
        
    fig = px.box(df, x=target, y=y_col, title=f'Comparison of {y_col} for Exited vs. Non-Exited Companies')
    return fig

def plot_interactive_scatter(x_col, y_col):
    """Creates a scatter plot to show the interaction between two variables, colored by exit type."""
    df = st.session_state.training_data
    if 'column_mapping' not in st.session_state or st.session_state.column_mapping['TARGET_VARIABLE'] == 'N/A':
        st.error("Please ensure a Target Variable is identified in the sidebar before plotting.")
        return None
    if x_col not in df.columns or y_col not in df.columns:
        st.error(f"One or both columns ('{x_col}', '{y_col}') not found in the data. Please check available columns.")
        return None
    target = st.session_state.column_mapping['TARGET_VARIABLE']
    fig = px.scatter(df, x=x_col, y=y_col, color=target, title=f'Interaction between {x_col} and {y_col}')
    return fig

# --- Streamlit App UI and Logic ---

st.set_page_config(layout="wide")
st.title("🚀 Autonomous AI Exit Predictor")
st.caption("With fully automatic column mapping and advanced driver analysis.")

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
        with st.spinner("Processing your file... This may take a moment."):
            df_raw = pd.read_csv(train_file, na_values=['—']) if train_file.name.endswith('.csv') else pd.read_excel(train_file, na_values=['—'])
            st.session_state.training_data = full_data_prep(df_raw)
            st.success(f"Loaded and prepared '{train_file.name}'.")

            # --- Automatically get column mapping after cleaning ---
            model = get_ai_model()
            all_cols = st.session_state.training_data.columns.tolist()
            st.session_state.column_mapping = get_column_mapping(model, all_cols)
        
        # --- NEW: Data Health Check Dashboard ---
        st.subheader("Data Health Check")
        with st.expander("View Cleaned Data Preview"):
            st.dataframe(st.session_state.training_data.head())
            st.write("**Data Types:**")
            st.dataframe(st.session_state.training_data.dtypes.astype(str))
        
        st.subheader("AI Column Role Analysis")
        st.json(st.session_state.column_mapping)


    predict_file = st.file_uploader("Upload Prediction Data", type=["xlsx", "csv"])
    if predict_file:
        with st.spinner("Processing your file..."):
            df_raw = pd.read_csv(predict_file, na_values=['—']) if predict_file.name.endswith('.csv') else pd.read_excel(predict_file, na_values=['—'])
            st.session_state.prediction_data = full_data_prep(df_raw)
            st.success(f"Loaded and prepared '{predict_file.name}'.")

# --- Main chat interface ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("data") is not None: st.dataframe(message["data"])
        if message.get("chart") is not None: st.plotly_chart(message["chart"], use_container_width=True, key=f"history_chart_{i}")

if prompt := st.chat_input("What would you like to do? (e.g., 'train model')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.training_data is None:
            st.warning("Please upload at least a training file first.")
            st.stop()
        
        with st.spinner("🧠 AI is thinking..."):
            model = get_ai_model()
            context_df = st.session_state.training_data
            ai_response = get_ai_response(model, prompt, list(context_df.columns))
            
            cleaned_response = ai_response.replace("```python", "").replace("```", "").strip()
            cleaned_response = cleaned_response.replace("`", "")
            cleaned_response = cleaned_response.replace("‘", "'").replace("’", "'")

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
