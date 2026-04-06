import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# =========================================================
# LOAD PIPELINE
# =========================================================
pipeline = joblib.load("model_pipeline.pkl")

# =========================================================
# MODEL COLUMNS (must match training exactly)
# =========================================================
model_columns = [
    'Income', 'Recency', 'NumDealsPurchases', 'NumWebPurchases',
    'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
    'Complain', 'Total_Accepted_Campaigns', 'Age', 'TotalSpend',
    'FamilySize', 'TotalPurchases', 'Customer_Tenure',
    'Education_Basic', 'Education_Graduation', 'Education_Master', 'Education_PhD',
    'Marital_Status_Partner', 'Marital_Status_Previously_Married'
]

# =========================================================
# SEGMENT MAP (based on your cluster means)
# 0 -> Premium
# 1 -> Low Value
# 2 -> Mid Value
# =========================================================
segment_map = {
    0: "Premium Customers",
    1: "Low Value Customers",
    2: "Mid Value Customers"
}

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
    .stApp {
        background-color: #0f1117;
    }
    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: white !important;
    }
    .about-box {
        background-color: #1f2937;
        padding: 12px;
        border-radius: 10px;
        font-size: 14px;
        color: white;
    }
    .result-box {
        padding: 14px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        margin-top: 12px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# FEATURE ENGINEERING
# =========================================================
def feature_engineering(df):
    df = df.copy()

    # Age
    if 'Year_Birth' in df.columns and 'Age' not in df.columns:
        df['Age'] = 2024 - df['Year_Birth']

    # FamilySize
    if 'Kidhome' in df.columns and 'Teenhome' in df.columns and 'FamilySize' not in df.columns:
        df['FamilySize'] = df['Kidhome'] + df['Teenhome'] + 1

    # TotalSpend
    spend_cols = [
        'MntWines', 'MntFruits', 'MntMeatProducts',
        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
    ]
    if all(col in df.columns for col in spend_cols) and 'TotalSpend' not in df.columns:
        df['TotalSpend'] = df[spend_cols].sum(axis=1)

    # TotalPurchases
    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    if all(col in df.columns for col in purchase_cols) and 'TotalPurchases' not in df.columns:
        df['TotalPurchases'] = df[purchase_cols].sum(axis=1)

    # Total Accepted Campaigns
    campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
    if all(col in df.columns for col in campaign_cols) and 'Total_Accepted_Campaigns' not in df.columns:
        df['Total_Accepted_Campaigns'] = df[campaign_cols].sum(axis=1)

    # Customer Tenure
    if 'Dt_Customer' in df.columns and 'Customer_Tenure' not in df.columns:
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
        df['Customer_Tenure'] = (
            pd.Timestamp.today().normalize() - df['Dt_Customer']
        ).dt.days.fillna(0)

    return df

# =========================================================
# PREPROCESS INPUT
# =========================================================
def preprocess_input(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    df = feature_engineering(df)

    if 'Education' in df.columns:
        df['Education'] = df['Education'].astype(str).str.strip()

    if 'Marital_Status' in df.columns:
        df['Marital_Status'] = df['Marital_Status'].astype(str).str.strip()

    processed = pd.DataFrame(0, index=df.index, columns=model_columns)

    base_cols = [
        'Income', 'Recency', 'NumDealsPurchases', 'NumWebPurchases',
        'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
        'Complain', 'Total_Accepted_Campaigns', 'Age', 'TotalSpend',
        'FamilySize', 'TotalPurchases', 'Customer_Tenure'
    ]

    for col in base_cols:
        if col in df.columns:
            processed[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    if 'Education' in df.columns:
        processed['Education_Basic'] = (df['Education'] == 'Basic').astype(int)
        processed['Education_Graduation'] = (df['Education'] == 'Graduation').astype(int)
        processed['Education_Master'] = (df['Education'] == 'Master').astype(int)
        processed['Education_PhD'] = (df['Education'] == 'PhD').astype(int)

    if 'Marital_Status' in df.columns:
        processed['Marital_Status_Partner'] = (df['Marital_Status'] == 'Partner').astype(int)
        processed['Marital_Status_Previously_Married'] = (
            df['Marital_Status'] == 'Previously_Married'
        ).astype(int)

    return processed

# =========================================================
# PREDICTION
# =========================================================
def predict_segments(input_df: pd.DataFrame):
    processed_df = preprocess_input(input_df)
    clusters = pipeline.predict(processed_df)

    result_df = input_df.copy()
    result_df["Predicted_Cluster"] = clusters
    result_df["Customer_Segment"] = result_df["Predicted_Cluster"].map(segment_map)

    return processed_df, result_df

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## About")
    st.markdown(
        """
        <div class="about-box">
        This app predicts customer segments based on customer demographics,
        spending behavior, and purchase activity.
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# APP HEADER
# =========================================================
st.markdown("## Customer Segmentation Dashboard")

mode = st.radio(
    "Choose Prediction Mode",
    ["Single Entry Prediction", "Bulk Prediction via File Upload"]
)

# =========================================================
# SINGLE ENTRY MODE
# =========================================================
if mode == "Single Entry Prediction":
    st.markdown("### Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        Income = st.number_input("Income", min_value=0.0, value=80000.0)
        Recency = st.number_input("Recency", min_value=0, value=25)
        NumDealsPurchases = st.number_input("NumDealsPurchases", min_value=0, value=2)
        NumWebPurchases = st.number_input("NumWebPurchases", min_value=0, value=6)
        NumCatalogPurchases = st.number_input("NumCatalogPurchases", min_value=0, value=7)
        NumStorePurchases = st.number_input("NumStorePurchases", min_value=0, value=8)
        NumWebVisitsMonth = st.number_input("NumWebVisitsMonth", min_value=0, value=5)

    with col2:
        Complain = st.selectbox("Complain", [0, 1])
        Total_Accepted_Campaigns = st.number_input("Total_Accepted_Campaigns", min_value=0, value=1)
        Age = st.number_input("Age", min_value=0, value=34)
        TotalSpend = st.number_input("TotalSpend", min_value=0.0, value=1000.0)
        FamilySize = st.number_input("FamilySize", min_value=1, value=3)
        TotalPurchases = st.number_input("TotalPurchases", min_value=0, value=21)
        Customer_Tenure = st.number_input("Customer_Tenure", min_value=0, value=10)

    col3, col4 = st.columns(2)

    with col3:
        Education = st.selectbox("Education", ["Basic", "Graduation", "Master", "PhD"])

    with col4:
        Marital_Status = st.selectbox("Marital Status", ["Single", "Partner", "Previously_Married"])

    single_df = pd.DataFrame([{
        'Income': Income,
        'Recency': Recency,
        'NumDealsPurchases': NumDealsPurchases,
        'NumWebPurchases': NumWebPurchases,
        'NumCatalogPurchases': NumCatalogPurchases,
        'NumStorePurchases': NumStorePurchases,
        'NumWebVisitsMonth': NumWebVisitsMonth,
        'Complain': Complain,
        'Total_Accepted_Campaigns': Total_Accepted_Campaigns,
        'Age': Age,
        'TotalSpend': TotalSpend,
        'FamilySize': FamilySize,
        'TotalPurchases': TotalPurchases,
        'Customer_Tenure': Customer_Tenure,
        'Education': Education,
        'Marital_Status': Marital_Status
    }])

    if st.button("Predict Segment"):
        processed_df, result_df = predict_segments(single_df)

        cluster = int(result_df.loc[0, "Predicted_Cluster"])
        segment = result_df.loc[0, "Customer_Segment"]

        color_map = {
            "Premium Customers": "#f59e0b",
            "Low Value Customers": "#ef4444",
            "Mid Value Customers": "#3b82f6"
        }

        box_color = color_map.get(segment, "#16a34a")

        st.markdown(
            f"""
            <div class="result-box" style="background-color:{box_color};">
                Predicted Segment: {segment}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Prediction Details")
        st.dataframe(result_df)

# =========================================================
# BULK MODE
# =========================================================
else:
    st.markdown("### Upload File for Bulk Prediction")

    st.write("You can upload either:")
    st.markdown("""
    - a **raw file** with original dataset columns, or
    - a **processed file** with final model-ready columns
    """)

    st.markdown("#### Example raw columns supported")
    st.code("""
ID
Year_Birth
Education
Marital_Status
Income
Kidhome
Teenhome
Dt_Customer
Recency
MntWines
MntFruits
MntMeatProducts
MntFishProducts
MntSweetProducts
MntGoldProds
NumDealsPurchases
AcceptedCmp1
AcceptedCmp2
AcceptedCmp3
AcceptedCmp4
AcceptedCmp5
NumWebPurchases
NumCatalogPurchases
NumStorePurchases
NumWebVisitsMonth
Complain
""")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                bulk_df = pd.read_csv(uploaded_file)
            else:
                bulk_df = pd.read_excel(uploaded_file)

            st.markdown("### Uploaded Data Preview")
            st.dataframe(bulk_df.head())

            if st.button("Run Bulk Prediction"):
                processed_df, result_df = predict_segments(bulk_df)

                st.markdown("### Prediction Results")
                st.dataframe(result_df.head())

                st.markdown("### Predicted Segment Counts")
                segment_counts = (
                    result_df["Customer_Segment"]
                    .value_counts()
                    .rename_axis("Segment")
                    .reset_index(name="Count")
                )
                st.dataframe(segment_counts)

                st.markdown("### Predicted Cluster Counts")
                cluster_counts = (
                    result_df["Predicted_Cluster"]
                    .value_counts()
                    .rename_axis("Cluster")
                    .reset_index(name="Count")
                )
                st.dataframe(cluster_counts)

                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results CSV",
                    data=csv,
                    file_name="customer_segmentation_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error while processing file: {e}")