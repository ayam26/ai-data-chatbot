import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import re
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"❌ Failed to configure AI model: {e}")
        return None

def generate_code(model, prompt, df_dict):
    """Uses the LLM to convert a prompt into executable code."""
    if model is None: return "st.error('AI model is not configured.')"
    df_names = list(df_dict.keys())
    primary_df_columns = list(df_dict.get(df_names[0], pd.DataFrame()).columns)

    system_prompt = f"""
    You are an expert Python data analyst. Your task is to convert a user's request into a single,
    executable line of Python code using pandas, plotly express, or a custom function.

    - You have access to a dictionary of DataFrames named `df_dict`. Available DataFrames: {df_names}.
    - The primary DataFrame is `df = df_dict['{df_names[0]}']` if it exists.
    - The output MUST be a single line of code. Do NOT use markdown or comments.
    - Available columns in the primary DataFrame are: {primary_df_columns}.

    NEW CAPABILITIES:
    1. TRAINING: If the user wants to TRAIN a model, generate this exact code:
       `df, message = train_exit_model(df, feature_cols=['FEATURE_1', 'FEATURE_2'])`
    2. PREDICTING: If the user wants to PREDICT on a file using the SAVED model, generate:
       `df, message = predict_with_saved_model(df)`

    Examples:
    User prompt: train a model to predict exits using Amount and Valuation as features
    Generated code: df, message = train_exit_model(df, feature_cols=['Amount', 'Valuation'])

    User prompt: use the trained model to predict on the 'new_companies' data
    Generated code: df, message = predict_with_saved_model(df_dict['new_companies'])

    User prompt: create a bar chart of the total Amount by Round
    Generated code: fig = px.bar(df.groupby('Round')['Amount'].sum().reset_index(), x='Round', y='Amount', title='Total Investment Amount by Round')
    """
    try:
        response = model.generate_content([system_prompt, prompt])
        return response.text.strip()
    except Exception as e:
        return f"st.error('❌ AI generation failed: {e}')"

# --- Machine Learning Functions ---

def train_exit_model(df, feature_cols, target_col='Exit'):
    """Trains a model and saves it to the session state."""
    if target_col not in df.columns:
        return None, f"Error: Training data must have a '{target_col}' column."
    
    df = df.copy()

    # --- SMARTER CLEANING STEP ---
    # Only clean columns that are specified as features and look like currency/numbers.
    for col in feature_cols:
        if col in df.columns and df[col].dtype == 'object':
            # Check if the column name suggests it's a monetary value
            if 'amount' in col.lower() or 'funding' in col.lower() or 'equity' in col.lower():
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), 
                    errors='coerce'
                )

    # Now, the rest of the function can proceed with a cleaner dataframe
    df_processed = pd.get_dummies(df[feature_cols]).fillna(0)
    st.session_state.trained_features = df_processed.columns.tolist()

    X = df_processed
    y = df[target_col].fillna(0)
    
    # Align data to ensure rows match after any potential cleaning/dropping
    y = y[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    st.session_state.trained_model = model

    accuracy = accuracy_score(y_test, model.predict(X_test))
    message = f"✅ Model trained with an accuracy of {accuracy:.2%}. It is now saved and ready for predictions."
    
    return df, message # Return the dataframe with cleaned numeric columns

def predict_with_saved_model(df):
    """Uses the saved model to make predictions on a new DataFrame."""
    if 'trained_model' not in st.session_state or st.session_state.trained_model is None:
        return None, "Error: You must train a model first before making predictions."
    
    model = st.session_state.trained_model
    trained_features = st.session_state.trained_features
    
    df = df.copy()
    
    # --- SMARTER CLEANING for prediction data ---
    for col in df.columns:
        if df[col].dtype == 'object':
            if 'amount' in col.lower() or 'funding' in col.lower() or 'equity' in col.lower():
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), 
                    errors='coerce'
                )

    df_processed = pd.get_dummies(df).fillna(0)
    df_processed = df_processed.reindex(columns=trained_features, fill_value=0)
    
    df['Exit_Probability'] = model.predict_proba(df_processed)[:, 1].round(4)
    message = f"✅ Predictions made using the saved model. Added 'Exit_Probability' column."
    return df, message

# --- Streamlit App UI and Logic ---

st.set_page_config(layout="wide")
st.title("🧠 AI Predictive Analyst")
st.caption("Train a model on one dataset, then predict on another.")

model = get_ai_model()

# Initialize session state
if "df_dict" not in st.session_state: st.session_state.df_dict = {}
if "messages" not in st.session_state: st.session_state.messages = []
if "trained_model" not in st.session_state: st.session_state.trained_model = None
if "trained_features" not in st.session_state: st.session_state.trained_features = None

with st.sidebar:
    st.header("Upload Your Data")
    uploaded_files = st.file_uploader("Upload training and prediction files", type=["xlsx", "csv"], accept_multiple_files=True)
    if uploaded_files:
        st.session_state.df_dict = {}
        for uploaded_file in uploaded_files:
            file_key = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.splitext(uploaded_file.name)[0]).lower()
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.df_dict[file_key] = df
            st.success(f"Loaded '{uploaded_file.name}' as '{file_key}'.")
    
    st.header("Analysis Status")
    st.write(f"**Datasets Loaded:** {', '.join(st.session_state.df_dict.keys()) or 'None'}")
    st.write(f"**Model Trained:** {'Yes' if st.session_state.trained_model else 'No'}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if message.get("data") is not None: st.dataframe(message["data"])

if prompt := st.chat_input("Train a model or make a prediction?"):
    st.session_state.messages.append({"role": "user", "content": prompt, "data": None})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.df_dict:
            st.warning("Please upload at least one data file first.")
        else:
            with st.spinner("🧠 AI is thinking..."):
                primary_df_name = list(st.session_state.df_dict.keys())[0]
                df_copy = st.session_state.df_dict[primary_df_name].copy()

                code_to_run = generate_code(model, prompt, st.session_state.df_dict)
                st.code(code_to_run, language="python")
                
                local_vars = {"df": df_copy, "pd": pd, "px": px, "train_exit_model": train_exit_model, "predict_with_saved_model": predict_with_saved_model, "df_dict": st.session_state.df_dict}
                response_content, response_data = "An unknown action occurred.", None
                
                try:
                    exec(code_to_run, globals(), local_vars)
                    st.session_state.df_dict = local_vars['df_dict']
                    
                    if 'message' in local_vars:
                        response_content = local_vars['message']
                        df_result = local_vars.get('df')
                        # Check if df_result is a DataFrame before calling .head()
                        if isinstance(df_result, pd.DataFrame):
                            response_data = df_result.head()
                            # If a prediction was made, update the specific dataframe in the dict
                            if "predict" in prompt:
                                predicted_df_name_match = re.search(r"on the '(.*?)'", prompt)
                                if predicted_df_name_match:
                                    predicted_df_name = predicted_df_name_match.group(1)
                                    if predicted_df_name in st.session_state.df_dict:
                                        st.session_state.df_dict[predicted_df_name] = df_result

                    else:
                        response_content = "✅ Command executed."

                    st.markdown(response_content)
                    if response_data is not None: st.dataframe(response_data)

                except Exception as e:
                    response_content = f"❌ Error executing code: {e}"
                    st.error(response_content)
            
            st.session_state.messages.append({"role": "assistant", "content": response_content, "data": response_data})
