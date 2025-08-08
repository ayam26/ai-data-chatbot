import streamlit as st
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
import json
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

# Suppress warnings
warnings.filterwarnings("ignore", message="Skipping features without any observed values")
warnings.filterwarnings("ignore", message="The parameter 'token_pattern' will not be used")

@st.cache_resource
def get_ai_model():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please set it in your Streamlit secrets.")
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def get_ai_response(model, prompt, df_columns):
    if model is None:
        return "ERROR: AI model is not configured."
    system_prompt = f"""... (your existing system prompt using {df_columns}) ..."""
    try:
        response = model.generate_content([system_prompt, prompt])
        return response.text.strip()
    except Exception as e:
        return f"ERROR: AI generation failed: {e}"

@st.cache_data
def get_column_mapping(_model, columns):
    prompt = f"""... (your existing prompt for column mapping) ..."""
    try:
        response = _model.generate_content(prompt)
        json_string = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_string)
    except Exception as e:
        st.warning(f"AI column mapping failed: {e}. Please map columns manually.")
        return {"TARGET_VARIABLE": "N/A", "ORGANIZATION_IDENTIFIER": "N/A", "TEXT_DESCRIPTION": "N/A", "CATEGORICAL_INDUSTRY": "N/A"}

def full_data_prep(df):
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).fillna('Unknown')
    return df

def train_and_score():
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

    df_train.dropna(subset=[target], inplace=True)
    if df_train.empty:
        return None, f"ERROR: After removing rows with missing '{target}' values, the training dataset is empty."

    df_train[target] = pd.to_numeric(df_train[target], errors='coerce')  # Ensure numeric
    df_train.dropna(subset=[target], inplace=True)
    if df_train.empty:
        return None, f"ERROR: '{target}' column isn't numeric or has no valid values."

    special_cols = [c for c in mapping.values() if c != "N/A"] + [target]
    numeric_features = df_train.select_dtypes(include=np.number).columns.drop(target, errors='ignore').tolist()
    object_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()

    text_features, categorical_features = [], []
    for col in object_cols:
        if col in special_cols:
            continue
        if df_train[col].nunique() / len(df_train) > 0.5:
            text_features.append(col)
        else:
            categorical_features.append(col)

    df_train[categorical_features] = df_train[categorical_features].fillna('Unknown').astype(str)
    non_empty_text = [col for col in text_features if df_train[col].str.strip().astype(bool).any()]
    if len(non_empty_text) < len(text_features):
        st.warning("One or more text columns were found empty and excluded.")
        text_features = non_empty_text

    st.session_state.model_features = {
        "numeric": numeric_features,
        "categorical": categorical_features,
        "text": text_features
    }
    st.info(f"Model Features Identified:\n• Numeric: {numeric_features}\n• Categorical: {categorical_features}\n• Text: {text_features}")

    # Prepare predict set
    for col in numeric_features:
        df_predict[col] = pd.to_numeric(df_predict.get(col, np.nan), errors='coerce')
    for col in categorical_features:
        df_predict[col] = df_predict.get(col, 'Unknown').astype(str).fillna('Unknown')
    for col in text_features:
        df_predict[col] = df_predict.get(col, '').astype(str).fillna('')

    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    text_transformers = [(f'text_{col}', TfidfVectorizer(stop_words='english', max_features=50), col) for col in text_features]

    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features),
                      ('cat', categorical_transformer, categorical_features)] + text_transformers,
        remainder='drop'
    )
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', oob_score=True))
    ])

    X_train = df_train.drop(columns=[target], errors='ignore')
    y_train = df_train[target]
    st.session_state.model_column_order = X_train.columns.tolist()
    model.fit(X_train, y_train)
    st.session_state.trained_model = model
    accuracy = model.named_steps['classifier'].oob_score_
    st.session_state.prediction_data_cleaned = df_predict.copy()
    X_predict = df_predict.drop(columns=[target], errors='ignore')
    probs = model.predict_proba(X_predict)[:, 1]
    name_col = mapping['ORGANIZATION_IDENTIFIER']
    results_df = pd.DataFrame({
        'Organization Name': df_predict[name_col] if name_col in df_predict.columns else df_predict.index,
        'Success Score': (probs * 100).round().astype(int)
    }).sort_values(by='Success Score', ascending=False)
    return results_df, f"✅ Model trained with estimated accuracy {accuracy:.2%}."

def get_feature_importance_plot():
    if 'trained_model' not in st.session_state or st.session_state.trained_model is None:
        st.error("You must train a model first.")
        return None
    model = st.session_state.trained_model
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        st.error(f"Could not get feature names: {e}")
        return None
    importances = classifier.feature_importances_
    if feature_names.shape[0] != importances.shape[0]:
        st.error("Feature names length mismatch with importances.")
        return None
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    imp_df['Feature'] = imp_df['Feature'].str.replace(r'.*__', '', regex=True)
    imp_df = imp_df.nlargest(20, 'Importance')
    fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                 title='Top 20 Drivers of Successful Exits')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def explain_single_prediction(company_name=None, row_index=None):
    if 'trained_model' not in st.session_state or not st.session_state.trained_model:
        st.error("Train a model first.")
        return None
    if 'prediction_data_cleaned' not in st.session_state:
        st.error("Run 'train model' first to prepare prediction data.")
        return None
    if 'model_features' not in st.session_state or 'model_column_order' not in st.session_state:
        st.error("Model metadata missing; please retrain.")
        return None

    model = st.session_state.trained_model
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    mapping = st.session_state.column_mapping
    id_col = mapping['ORGANIZATION_IDENTIFIER']
    df_clean = st.session_state.prediction_data_cleaned.copy()

    if company_name:
        row = df_clean[df_clean.get(id_col, '') == company_name]
        if row.empty:
            st.error(f"Company '{company_name}' not found.")
            return None
        company_row = row
    elif row_index is not None:
        try:
            company_row = df_clean.iloc[[row_index]]
        except IndexError:
            st.error(f"Row index {row_index} out of bounds.")
            return None
    else:
        st.error("Provide either a company name or row index.")
        return None

    prepared = company_row.reindex(columns=st.session_state.model_column_order)
    for col in st.session_state.model_column_order:
        if col in st.session_state.model_features['numeric']:
            prepared[col] = pd.to_numeric(prepared[col], errors='coerce')
        elif col in st.session_state.model_features['categorical']:
            prepared[col] = prepared[col].fillna('Unknown').astype(str)
        elif col in st.session_state.model_features['text']:
            prepared[col] = prepared[col].fillna('').astype(str)

    try:
        transformed = preprocessor.transform(prepared)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = transformed.astype(float)
        feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        st.error(f"Failed to transform data: {e}")
        st.write("Data dtypes:", prepared.dtypes.to_frame('type').T)
        return None

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)
    shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': shap_values[1][0]})
    shap_df['Feature'] = shap_df['Feature'].str.replace(r'.*__', '', regex=True)
    shap_df['Color'] = shap_df['SHAP Value'].apply(lambda v: 'red' if v < 0 else 'green')
    shap_df = shap_df.reindex(shap_df['SHAP Value'].abs().sort_values(ascending=True).index).tail(15)

    fig = go.Figure(go.Bar(
        x=shap_df['SHAP Value'],
        y=shap_df['Feature'],
        orientation='h',
        marker_color=shap_df['Color']
    ))
    fig.update_layout(
        title=f"Prediction Drivers for {company_name or f'Row {row_index}'}",
        xaxis_title="SHAP Value (Impact on Success Probability)",
        yaxis_title="Feature"
    )
    return fig

def plot_correlation_heatmap():
    df = st.session_state.training_data
    corr = df.select_dtypes(include=np.number).corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'
    ))
    fig.update_layout(title='Correlation Between Metrics and Exits')
    return fig

def plot_comparison_boxplot(y_col):
    df = st.session_state.training_data
    mapping = st.session_state.column_mapping
    if mapping['TARGET_VARIABLE'] == 'N/A':
        st.error("Select a Target Variable first.")
        return None
    if y_col not in df.columns:
        st.error(f"{y_col} not in data; available numeric columns: {df.select_dtypes(include=np.number).columns.tolist()}")
        return None
    target = mapping['TARGET_VARIABLE']
    fig = px.box(df, x=target, y=y_col, title=f'Dist. of {y_col} by {target}')
    return fig

def plot_interactive_scatter(x_col, y_col):
    df = st.session_state.training_data
    mapping = st.session_state.column_mapping
    if mapping['TARGET_VARIABLE'] == 'N/A':
        st.error("Select a Target Variable first.")
        return None
    if x_col not in df.columns or y_col not in df.columns:
        st.error(f"Columns '{x_col}' or '{y_col}' not found.")
        return None
    target = mapping['TARGET_VARIABLE']
    fig = px.scatter(df, x=x_col, y=y_col, color=target,
                     title=f'Interaction: {x_col} vs {y_col}')
    return fig

# Streamlit UI Setup
st.set_page_config(layout="wide")
st.title("🚀 Autonomous AI Exit Predictor")
st.caption("Automatic column mapping + powerful explainability.")

for key in ["messages", "training_data", "prediction_data", "column_mapping",
            "trained_model", "model_features", "model_column_order"]:
    if key not in st.session_state:
        st.session_state[key] = None

with st.sidebar:
    st.header("1. Upload Data")
    train_file = st.file_uploader("Training Data", type=["csv", "xlsx"], key="train")
    if train_file:
        try:
            df_raw = pd.read_csv(train_file, na_values=['—']) if train_file.name.endswith('.csv') \
                     else pd.read_excel(train_file, na_values=['—'])
            st.session_state.training_data = full_data_prep(df_raw)
            st.success(f"Loaded {train_file.name}")
        except Exception as e:
            st.error(f"Error reading training file: {e}")

    predict_file = st.file_uploader("Prediction Data", type=["csv", "xlsx"], key="predict")
    if predict_file:
        try:
            df_raw = pd.read_csv(predict_file, na_values=['—']) if predict_file.name.endswith('.csv') \
                     else pd.read_excel(predict_file, na_values=['—'])
            st.session_state.prediction_data = full_data_prep(df_raw)
            st.success(f"Loaded {predict_file.name}")
        except Exception as e:
            st.error(f"Error reading prediction file: {e}")

    if st.session_state.training_data is not None:
        st.header("2. Confirm Column Roles")
        all_cols = ["N/A"] + st.session_state.training_data.columns.tolist()
        if st.session_state.column_mapping is None:
            model = get_ai_model()
            st.session_state.column_mapping = get_column_mapping(model, all_cols[1:])
        mapping = st.session_state.column_mapping

        def safe_idx(k):
            return all_cols.index(mapping.get(k, "N/A")) if mapping.get(k, "N/A") in all_cols else 0

        mapping['TARGET_VARIABLE'] = st.selectbox("Target Variable", all_cols, index=safe_idx('TARGET_VARIABLE'))
        mapping['ORGANIZATION_IDENTIFIER'] = st.selectbox("Company Name Column", all_cols, index=safe_idx('ORGANIZATION_IDENTIFIER'))
        mapping['TEXT_DESCRIPTION'] = st.selectbox("Text Description Column", all_cols, index=safe_idx('TEXT_DESCRIPTION'))
        mapping['CATEGORICAL_INDUSTRY'] = st.selectbox("Industry Column", all_cols, index=safe_idx('CATEGORICAL_INDUSTRY'))

# Chat UI & Logic
if st.session_state.messages is None:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Upload data, confirm columns, then try commands like 'train model' or 'explain the score for [Company]'.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("data") is not None:
            st.dataframe(msg["data"])
        if msg.get("chart") is not None:
            st.plotly_chart(msg["chart"], use_container_width=True)

if prompt := st.chat_input("What would you like to do?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        if st.session_state.training_data is None:
            st.warning("Upload training data first.")
            st.stop()

        model = get_ai_model()
        ai_code = get_ai_response(model, prompt, list(st.session_state.training_data.columns))
        cleaned = ai_code.replace("```python", "").replace("```", "").strip().replace("`", "")
        is_code = any(kw in cleaned for kw in ['fig =', 'train_and_score', 'df ='])

        if cleaned.startswith("ERROR:"):
            st.error(cleaned)
            st.session_state.messages.append({"role": "assistant", "content": cleaned})
        elif is_code:
            st.code(cleaned, language="python")
            local_vars = {
                **globals(),
                **{
                    "train_and_score": train_and_score,
                    "get_feature_importance_plot": get_feature_importance_plot,
                    "plot_correlation_heatmap": plot_correlation_heatmap,
                    "plot_comparison_boxplot": plot_comparison_boxplot,
                    "plot_interactive_scatter": plot_interactive_scatter,
                    "explain_single_prediction": explain_single_prediction
                }
            }
            try:
                exec(cleaned, globals(), local_vars)
                content, data, chart = "✅ Command executed.", None, None
                if local_vars.get("fig") is not None:
                    chart = local_vars["fig"]
                    content = f"✅ Here’s the visualization for '{prompt}'"
                elif 'results_df' in local_vars:
                    data, content = local_vars["results_df"], local_vars["message"]
                st.markdown(content)
                if data is not None:
                    st.dataframe(data)
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)
                st.session_state.messages.append({"role": "assistant", "content": content, "data": data, "chart": chart})
            except Exception as e:
                err = f"❌ Error executing code: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
        else:
            st.markdown(cleaned)
            st.session_state.messages.append({"role": "assistant", "content": cleaned})
