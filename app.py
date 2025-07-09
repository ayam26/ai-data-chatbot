import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
import re
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import io

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
    Uses the LLM to generate either a conversational response or executable code.
    """
    if model is None: return "ERROR: AI model is not configured."
    df_names = list(df_dict.keys())
    primary_df_name = df_names[0] if df_names else ''
    primary_df_columns = list(df_dict.get(primary_df_name, pd.DataFrame()).columns)

    system_prompt = f"""
    You are a helpful AI assistant with two modes: Data Analyst and Conversationalist.

    1.  **Data Analyst Mode**: If the user gives a direct command to manipulate or analyze data (e.g., 'filter...', 'sort...', 'plot...', 'train...'), you MUST respond ONLY with a single, executable line of Python code.
    2.  **Conversational Mode**: If the user asks a question (especially starting with 'how', 'what', 'why', 'explain') or gives a greeting, respond with a friendly, helpful text message. Do NOT generate code for these.

    **Decision-Making:**
    - If the prompt starts with "how", "what", "why", "explain", "tell me", or is a greeting, ALWAYS use Conversational Mode.
    - Otherwise, if the prompt contains data-related keywords like 'filter', 'sort', 'plot', 'train', 'predict', 'rename', assume it's a data task and generate code.

    **Code Generation Rules (Data Analyst Mode Only):**
    - You have access to a dictionary of DataFrames named `df_dict`. Available DataFrames: {df_names}.
    - The primary DataFrame is `df = df_dict['{primary_df_name}']` if it exists.
    - The output MUST be a single line of code. Do NOT use markdown or comments.
    - To train a model, generate: `message = train_exit_model(df)`
    - To predict with a saved model, generate: `df, message = predict_with_saved_model(df)`

    **PANDAS EXAMPLES:**
    User: "filter to only show rows where Last Funding Type is Series A, Series B, or Seed"
    Generated code: `df = df[df['Last Funding Type'].isin(['Series A', 'Series B', 'Seed'])]`

    User: "sort by Total Funding Amount descending"
    Generated code: `df = df.sort_values(by='Total Funding Amount', ascending=False)`
    """
    try:
        response = model.generate_content([system_prompt, prompt])
        return response.text.strip()
    except Exception as e:
        return f"ERROR: AI generation failed: {e}"

# --- Machine Learning Functions ---

def clean_monetary_columns(df):
    """A robust function to clean known monetary columns."""
    df = df.copy()
    monetary_cols = [
        'Last Funding Amount', 'Total Equity Funding Amount', 'Total Funding Amount',
        'Amount', 'Valuation'
    ]
    
    for col in monetary_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                errors='coerce'
            ).fillna(0)
    return df

def train_exit_model(df, target_col='Exit'):
    """Trains a model and returns a success message."""
    if target_col not in df.columns:
        return f"Error: Training data must have a '{target_col}' column."
    
    feature_cols = [col for col in df.columns if col not in [target_col, 'Organization Name', 'Description', 'Top 5 Investors', 'Exit Date', 'Founded Date', 'Last Funding Date']]
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    df_for_ml = pd.get_dummies(df[feature_cols], dummy_na=True).fillna(0)
    st.session_state.trained_features = df_for_ml.columns.tolist()

    X = df_for_ml
    y = df[target_col].fillna(0)
    
    y, X = y.align(X, join='inner', axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    st.session_state.trained_model = model

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return f"✅ RandomForest model trained with an accuracy of {accuracy:.2%}. It is now saved and ready for predictions."

def predict_with_saved_model(df):
    """Uses the saved model to make predictions, create a score, and sort."""
    if 'trained_model' not in st.session_state or st.session_state.trained_model is None:
        return df, "Error: You must train a model first before making predictions."
    
    model = st.session_state.trained_model
    trained_features = st.session_state.trained_features
    
    df_processed = pd.get_dummies(df).fillna(0)
    df_processed = df_processed.reindex(columns=trained_features, fill_value=0)
    
    df['Exit_Probability'] = model.predict_proba(df_processed)[:, 1]
    df['Exit Score (1-100)'] = (df['Exit_Probability'] * 100).round().astype(int)
    
    df = df.sort_values(by='Exit Score (1-100)', ascending=False)
    
    message = f"✅ Predictions made and scored. Displaying top potential exits."
    return df, message

# --- Streamlit App UI and Logic ---

st.set_page_config(layout="wide")
st.title("🧠 AI Predictive Analyst")
st.caption("Your conversational partner for data analysis and prediction.")

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
            df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.df_dict[file_key] = clean_monetary_columns(df_raw)
            st.success(f"Loaded and cleaned '{uploaded_file.name}' as '{file_key}'.")
    
    st.header("Analysis Status")
    st.write(f"**Datasets Loaded:** {', '.join(st.session_state.df_dict.keys()) or 'None'}")
    st.write(f"**Model Trained:** {'Yes' if st.session_state.trained_model else 'No'}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if message.get("data") is not None:
            st.caption("Displaying the first 5 rows as a preview. The full dataset has been updated in memory.")
            st.dataframe(message["data"])

if prompt := st.chat_input("Ask a question or give a command..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "data": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.df_dict and "load" not in prompt.lower():
            st.warning("Please upload at least one data file first.")
        else:
            with st.spinner("🧠 Thinking..."):
                primary_df_name = list(st.session_state.df_dict.keys())[0] if st.session_state.df_dict else None
                df_copy = st.session_state.df_dict[primary_df_name].copy() if primary_df_name else None
                
                parts = prompt.split()
                command = parts[0].lower() if parts else ""

                if command == "save":
                    if df_copy is None:
                        st.warning("Please upload a file first.")
                    else:
                        try:
                            output_path = parts[1]
                            output_buffer = io.BytesIO()
                            with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                                df_copy.to_excel(writer, index=False, sheet_name='Sheet1')
                            
                            st.download_button(
                                label=f"📥 Download {output_path}",
                                data=output_buffer.getvalue(),
                                file_name=output_path,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            response_content = f"Your file is ready. Click the button above to download."
                            st.markdown(response_content)
                        except Exception as e:
                            response_content = f"❌ Error creating download link: {e}"
                            st.error(response_content)
                    st.session_state.messages.append({"role": "assistant", "content": response_content, "data": None})
                else:
                    ai_response = get_ai_response(model, prompt, st.session_state.df_dict)
                    
                    # --- FINAL FIX IS HERE: Clean the AI response ---
                    cleaned_response = ai_response.strip().strip('`')

                    code_keywords = ['df =', 'fig =', 'message =', 'df, message =']
                    is_code = any(keyword in cleaned_response for keyword in code_keywords)

                    if cleaned_response.startswith("ERROR:"):
                        response_content = f"❌ {cleaned_response}"
                        st.error(response_content)
                    elif is_code:
                        st.code(cleaned_response, language="python")
                        local_vars = {"df": df_copy, "pd": pd, "px": px, "train_exit_model": train_exit_model, "predict_with_saved_model": predict_with_saved_model, "df_dict": st.session_state.df_dict}
                        response_content, response_data = "An unknown action occurred.", None
                        
                        try:
                            exec(cleaned_response, globals(), local_vars)
                            
                            df_result = local_vars.get('df')
                            
                            if 'message' in local_vars:
                                response_content = local_vars['message']
                                if isinstance(df_result, pd.DataFrame):
                                    st.session_state.df_dict[primary_df_name] = df_result
                                    response_data = df_result.head()
                            else:
                                if isinstance(df_result, pd.DataFrame):
                                    st.session_state.df_dict[primary_df_name] = df_result
                                    response_content = "✅ Command executed successfully."
                                    response_data = df_result.head()
                                else:
                                    response_content = f"✅ Command executed. Result: {df_result}"

                            st.markdown(response_content)
                            if response_data is not None:
                                st.caption("Displaying the first 5 rows as a preview. The full dataset has been updated in memory.")
                                st.dataframe(response_data)

                        except Exception as e:
                            response_content = f"❌ Error executing code: {e}"
                            st.error(response_content)
                    else: # It's a conversational response
                        response_content = cleaned_response
                        st.markdown(response_content)
                
                    st.session_state.messages.append({"role": "assistant", "content": response_content, "data": response_data if 'response_data' in locals() else None})
