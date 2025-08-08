Why it still happens:
Even if you cast numeric columns, some columns in your transformed row are still object dtype, likely:

Categorical columns with missing or unexpected values

Text columns or columns that your pipeline expects to be numeric but aren't clean

Or the SimpleImputer or transformers inside the pipeline aren't properly handling missing/strange values in this single-row input

What to do:
1. Add explicit data cleaning before transformation inside explain_single_prediction:
Right before you do preprocessor.transform(company_row_prepared), run:

python
Copy
Edit
for col in model_columns:
    if col in model_features['numeric']:
        company_row_prepared[col] = pd.to_numeric(company_row_prepared[col], errors='coerce')
    elif col in model_features['categorical']:
        # Fill missing and convert to string to avoid object issues
        company_row_prepared[col] = company_row_prepared[col].fillna('Unknown').astype(str)
    elif col in model_features['text']:
        company_row_prepared[col] = company_row_prepared[col].fillna('').astype(str)
Make sure this is exactly how you prepare the data every time you pass data to the pipeline.

2. Check the dtypes right before transformation:
Add this debug print or st.write() to see what types you have:

python
Copy
Edit
st.write(company_row_prepared.dtypes)
st.write(company_row_prepared)
Look specifically for any columns that still show dtype('O') but should be numeric.

3. Check the full pipeline preprocessor input expectations:
SimpleImputer expects numeric columns to be numeric, not objects.

TfidfVectorizer expects text columns as strings.

OneHotEncoder expects categorical columns as strings with no nulls.

Make sure no missing values or NaNs in categorical/text columns and numeric columns are strictly floats/ints or NaN (not strings).

4. Possible workaround: convert entire row dataframe to a consistent dtype before transforming:
Try adding:

python
Copy
Edit
company_row_prepared = company_row_prepared.astype(object)
before transforming, but usually this does not help if the underlying problem is mixed types.

5. Trace error origin:
The error comes from a NumPy isnan call on object data.

Try isolating which column triggers this by applying np.isnan() on each numeric column manually:

python
Copy
Edit
for col in model_features['numeric']:
    try:
        np.isnan(company_row_prepared[col].values)
    except Exception as e:
        st.write(f"Column '{col}' triggers isnan error: {e}")
This will help you identify exactly which column is causing the issue.

Summary checklist:
Fully clean numeric columns with pd.to_numeric(..., errors='coerce') before transform

Fill missing categorical/text columns and convert to str

Check and debug column dtypes just before transform

Identify the exact column causing the isnan failure

Ensure your input row matches the training data feature expectations exactly
