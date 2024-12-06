import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()  # This will load environment variables from the .env file

api_key = os.getenv("api_key")
# Initialize the Google Generative AI LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # Specify the correct model
    temperature=0.5,
    timeout=None,
    max_retries=2,
    api_key=api_key,  # Ensure the API key is set
)

# Initialize an empty list to store DataFrame info
data_info = []

# File uploader in Streamlit (allow multiple CSV files)
uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

# Check if files are uploaded
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            df_name = uploaded_file.name.split('.')[0]  # Use file name (without extension) as variable name
            #uploaded_files[df_name] = pd.read_csv(uploaded_file) 
            globals()[df_name]=pd.read_csv(uploaded_file)
            # Store the DataFrame info in the data_info list
            data_info.append({df_name: dict(globals()[df_name].head(5))})
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")

# Streamlit widget for entering metadata and purpose of analysis
metadata = st.text_area("Enter DataFrame Metadata")
purpose = st.text_area("Enter Purpose of Analysis")

# Define the prompt template for Langchain
prompt_template = """
You are a data analysis expert who provides insights based on data. The user will provide metadata about the dataframes in text format, describing the columns and data_info containing actual columns info. Your job is to provide visualizations that serve the analysis purpose given by the user. 

### Instructions:
1) Do **NOT** tamper with original dataframes for calculations, use a copy instead.
2) Always use `df.info()` to get the column names and df.dtypes for checking datatypes before suggesting visualizations.
3) Do **NOT** suggest charts for columns that are not in the provided dataframe. Ensure both `Column 1` and `Column 2` are from the dataframe columns. Do not hallucinate column names.
4) Recommend only the most suitable chart types and their corresponding columns based on the user’s analysis purpose.
5) Do **NOT** include any preambles, explanations, or supplementary suggestions. Only provide the visualization code directly.
6) If merging data is required:
    - Merging should be done directly in the `sns` line (e.g., `sns.barplot(data=fact_events.merge(dim_stores), x="city", y="quantity_sold(after_promo)", hue="campaign_id")`) and not in any other part of the code.
    - Any calculations needed should be done **before** plotting (e.g., creating new columns in the dataframe), not in the `x` or `y` arguments of the plot function.
) **Data Processing**:
    - Ensure that any necessary calculations are performed **before** creating the plot.
    - Example: Calculating `quantity_diff` should be done as a new column in the dataframe, not in the `sns.barplot()` function.

### Example of expected output:
# Compare promo effect across cities for each campaign
sns.barplot(data=fact_events.merge(dim_stores), x="city", y="quantity_sold(after_promo)", hue="campaign_id")

### DataFrame Metadata:
{metadata}

### Purpose of Analysis:
{purpose}

### data_frame_info
{data_info}
"""

# Initialize the PromptTemplate with input variables
prompt = PromptTemplate(
    input_variables=["metadata", "purpose", "data_info"],
    template=prompt_template
)

# Format the prompt with actual values
prompt_input = prompt.format(metadata=metadata, purpose=purpose, data_info=data_info)

# Create an LLMChain using the prompt and the LLM
chain = LLMChain(llm=llm, prompt=prompt)

# Get the response (visualization code) from the LLM
response = chain.run({"metadata": metadata, "purpose": purpose, "data_info": data_info})

# Display the recommended visualizations (code) in the Streamlit app
st.write("Recommended Visualizations based on Purpose:")
st.text(response)

# Execute the generated code and display the plots
for i in response.split("\n"):
    try:
        if '=' in i and 'sns.' not in i:  # Likely a data processing step
            exec(i)
        if 'sns.' in i:
            plt.figure(figsize=(10, 6))
            exec(i)
            st.pyplot()  # Display the plot in the Streamlit app
    except Exception as e:
        st.error(f"Error executing code: {e}")
