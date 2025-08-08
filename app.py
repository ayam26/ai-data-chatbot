# app.py (corrected)
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
import shap

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
        # Using a specific, stable model version
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
    - To "explain the score for" a specific company, generate `fig = explain_single_prediction(company_name='Company Name')`, extracting the company name.
    - To "explain score for row", generate `fig = explain_single_prediction(row_index=Number)`, extracting the row number.
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
    - User: "explain the score for FutureTech" -> AI: `fig = explain_single_prediction(company_name='FutureTech')`
    - User: "explain score for row 21" -> AI: `fig = explain_single_prediction(row_index=21)`
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

    Return your answer as a JSON object with these four keys. For each key, the value should be your best guess for the column name from the list. If you cannot find a suitable column for a role, use "N/A".
    """
    try:
        response = _model.generate_content(prompt)
        json_string = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_string)
    except (json.JSONDecodeError, Exception) as e:
        st.warning(f"AI column mapping failed: {e}. Please map columns manually.")
        return {"TARGET_VARIABLE": "N/A", "ORGANIZATION_IDENTIFIER": "N/A", "TEXT_DESCRIPTION": "N/A", "CATEGORICAL_INDUSTRY": "N/A"}


# --- Data Processing and Modeling ---
def full_data_prep(df):
    """
    Loads a dataframe and performs basic cleaning.
    """
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    # Convert all object columns to string to prevent mixed type errors later
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).fillna('Unknown')
    return df

def train_and_score():
    """Dynamically identifies features and trains the model based on user-confirmed column mappings."""
    if 'training_data' not in st.session_state or st.session_state.training_data is None:
        return None, "ERROR: Training data has not been uploaded."
    if 'prediction_data' not in st.session_state or st.session_state.prediction_data is None:
        return None, "ERROR: Prediction data has not been uploaded."
        
    df_train = st.session_state.training_data.copy()
    df_predict = st.session_state.prediction_data.copy()
    
    mapping = st.session_state.column_mapping
    target = mapping['TARGET_VARIABLE']

    if target == "N/A" or target not in df_train.columns:
        return None, "ERROR: A valid 'Target Variable' must be selected in the sidebar."
        
    # Drop rows with missing target values (do NOT coerce target to numeric; keep labels as-is)
    df_train.dropna(subset=[target], inplace=True)
    if df_train.empty:
        return None, f"ERROR: After removing rows with missing '{target}' values, the training dataset is empty."

    special_cols = [c for c in mapping.values() if c != "N/A"] + [target]
    
    numeric_features = df_train.select_dtypes(include=np.number).columns.drop(target, errors='ignore').tolist()
    
    object_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    text_features = []
    categorical_features = []
    for col in object_cols:
        if col in special_cols:
            continue
        # Heuristic to decide if a column is text or categorical
        if df_train[col].nunique() / len(df_train) > 0.5:
            text_features.append(col)
        else:
            categorical_features.append(col)
            
    for col in categorical_features:
        df_train[col] = df_train[col].astype(str).fillna('Unknown')

    non_empty_text_features = []
    for col in text_features:
        if col in df_train.columns and df_train[col].str.strip().astype(bool).any():
            non_empty_text_features.append(col)
    
    if len(non_empty_text_features) < len(text_features):
        st.warning("One or more text columns were found to be empty and will be excluded from the model.")
        text_features = non_empty_text_features

    # Save identified features to session state for later use
    st.session_state.model_features = {"numeric": numeric_features, "categorical": categorical_features, "text": text_features}
    st.info(f"**Model Features Identified:**\n- **Numeric:** {numeric_features}\n- **Categorical:** {categorical_features}\n- **Text:** {text_features}")

    # Prepare prediction data to match training data structure
    for col in numeric_features:
        if col in df_predict.columns: df_predict[col] = pd.to_numeric(df_predict[col], errors='coerce')
        else: df_predict[col] = np.nan
    for col in categorical_features:
        if col in df_predict.columns: df_predict[col] = df_predict[col].astype(str).fillna('Unknown')
        else: df_predict[col] = 'Unknown'
    for col in text_features:
        if col in df_predict.columns: df_predict[col] = df_predict[col].astype(str).fillna('')
        else: df_predict[col] = ''

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    text_transformers = [(f'text_{col}', TfidfVectorizer(stop_words='english', max_features=50), col) for col in text_features]

    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)] + text_transformers, remainder='drop')
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', oob_score=True))])

    X_train = df_train.drop(columns=[target], errors='ignore')
    y_train = df_train[target]
    
    # Save the column order the model was trained on
    st.session_state.model_column_order = X_train.columns.tolist()

    model.fit(X_train, y_train)
    st.session_state.trained_model = model

    # oob_score_ exists because we set oob_score=True
    accuracy = model.named_steps['classifier'].oob_score_ if hasattr(model.named_steps['classifier'], 'oob_score_') else None
    if accuracy is not None:
        message = f"✅ Model trained with an estimated accuracy of {accuracy:.2%}. You can now score the prediction data or analyze feature importance."
    else:
        message = "✅ Model trained. You can now score the prediction data or analyze feature importance."
    
    # Save the cleaned prediction data to session state for explainability
    st.session_state.prediction_data_cleaned = df_predict.copy()

    X_predict = df_predict.drop(columns=[target], errors='ignore')
    probabilities = model.predict_proba(X_predict)[:, 1]
    org_name_col = mapping['ORGANIZATION_IDENTIFIER']
    results_df = pd.DataFrame({'Organization Name': df_predict[org_name_col] if org_name_col != "N/A" and org_name_col in df_predict.columns else df_predict.index, 'Success Score': (probabilities * 100).round().astype(int)}).sort_values(by='Success Score', ascending=False)
    return results_df, message

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
    
    # Clean up feature names for better readability
    importance_df['Feature'] = importance_df['Feature'].str.replace(r'.*__', '', regex=True)
    
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Top 20 Drivers of Successful Exits')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def explain_single_prediction(company_name=None, row_index=None):
    """
    Generates a SHAP plot to explain the prediction for a single company,
    accepting either a name or a row index.
    """
    # --- 1. Essential checks ---
    if 'trained_model' not in st.session_state or st.session_state.trained_model is None:
        st.error("You must train a model first.")
        return None
    if 'prediction_data_cleaned' not in st.session_state:
        st.error("Please run 'train model' to prepare the prediction data before explaining a score.")
        return None
    if 'model_features' not in st.session_state or 'model_column_order' not in st.session_state:
        st.error("Model features or column order not found. Please retrain the model.")
        return None

    # --- 2. Load necessary objects from session state ---
    model = st.session_state.trained_model
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    model_features = st.session_state.model_features
    model_columns = st.session_state.model_column_order
    mapping = st.session_state.column_mapping
    id_col = mapping['ORGANIZATION_IDENTIFIER']
    
    # --- 3. Find the single row from the CLEANED prediction data ---
    df_predict_cleaned = st.session_state.prediction_data_cleaned.copy()
    company_row = None

    if company_name:
        # match exactly; if you want case-insensitive, adjust here
        company_row = df_predict_cleaned.loc[df_predict_cleaned.get(id_col, '') == company_name]
        if company_row.empty:
            st.error(f"Company '{company_name}' not found in the prediction data.")
            return None
    elif row_index is not None:
        try:
            company_row = df_predict_cleaned.iloc[[row_index]]
        except IndexError:
            st.error(f"Row index {row_index} is out of bounds for the prediction data.")
            return None
    else:
        st.error("Please provide a company name or a row index to explain.")
        return None
    
    # --- 4. Safeguard: Re-apply structure and dtypes ---
    # Although we start from cleaned data, slicing can still alter dtypes.
    # This ensures the row is perfect before transforming.
    company_row_prepared = company_row.reindex(columns=model_columns)

    for col in model_columns:
        if col in model_features['numeric']:
            company_row_prepared[col] = pd.to_numeric(company_row_prepared[col], errors='coerce')
        elif col in model_features['categorical']:
            company_row_prepared[col] = company_row_prepared[col].fillna('Unknown').astype(str)
        elif col in model_features['text']:
            company_row_prepared[col] = company_row_prepared[col].fillna('').astype(str)
            
    # --- 5. Transform and Explain ---
    try:
        transformed_row = preprocessor.transform(company_row_prepared)
        # --- SHAP expects numerical floats; ensure we pass it a dense float array ---
        if hasattr(transformed_row, "toarray"):
            transformed_row = transformed_row.toarray()
        # force numeric dtype (this will raise if truly non-numeric values exist)
        transformed_row = np.asarray(transformed_row, dtype=float)
        feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        st.error(f"Failed to transform data for explanation: {e}")
        st.write("Debug Info: Data types of the row just before transformation:")
        st.dataframe(company_row_prepared.dtypes.to_frame('Data Type').T)
        return None

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed_row)

    # If classifier has two classes, shap_values is a list; we pick the 'positive' class index 1
    idx_to_use = 1 if isinstance(shap_values, list) and len(shap_values) > 1 else 0
    shap_vals_for_row = shap_values[idx_to_use][0] if isinstance(shap_values, list) else shap_values[0]

    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_vals_for_row
    })
    shap_df['Feature'] = shap_df['Feature'].str.replace(r'.*__', '', regex=True)
    shap_df['Color'] = ['red' if v < 0 else 'green' for v in shap_df['SHAP Value']]
