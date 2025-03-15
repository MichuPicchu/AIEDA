import streamlit as st
import pandas as pd
import os
from utils.calc import (
    get_table_shape,
    get_column_datatypes,
    get_missing_values,
    get_unique_values,
    get_numeric_stats,
    detect_outliers,
    get_example_rows,
    get_duplicate_and_na_counts
)
import plotly.express as px
import plotly.graph_objects as go
from utils.helper import (
    gen_preprocessing_tips,
    gen_feature_tips,
    split_data,
)
import zipfile
import io

# Constants
EXAMPLE_FILES = {
    "Example Data 1": "data/laptopData.csv",
    "Example Data 2 (with pre-generated tips)": "data/bike_sales.xlsx",
    "Example Data 3": "data/placementdata.csv",
    "Example Data 4": "data/statistics_on_daily_passenger_traffic.csv",
}
DATA_PURPOSE = {
    "None": None,
    "Regression": "",
    "Classification": "",
}

st.set_page_config(layout="wide")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analyses' not in st.session_state:
    st.session_state.analyses = {}
if "preprocessing_tips" not in st.session_state:
    st.session_state.preprocessing_tips = ""
if "feature_tips" not in st.session_state:
    st.session_state.feature_tips = ""
if "loaded_file_path" not in st.session_state:
    st.session_state.loaded_file_path = ""


def load_data(file):
    """Load data from either a file path or uploaded file object."""
    if isinstance(file, str):  # Example file (path string)
        ext = os.path.splitext(file)[1].lower()
    else:  # Uploaded file (BytesIO)
        ext = os.path.splitext(file.name)[1].lower()
    
    if ext == '.csv':
        try:
            return pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            # Try a different encoding if UTF-8 fails
            return pd.read_csv(file, encoding='ISO-8859-1')
    if ext == '.xlsx':
        return pd.read_excel(file)
    raise ValueError(f"Unsupported file format: {ext}")

def compute_analyses(df):
    """Compute and return all data analyses in a dictionary."""
    df.columns = df.columns.str.strip()
    return {
        "Table_Shape": get_table_shape(df),
        "Overall_duplicates_nas": get_duplicate_and_na_counts(df),
        "Column_Data_Types": get_column_datatypes(df),
        "Missing_Values": get_missing_values(df),
        "Unique_Values": get_unique_values(df, max_unique=10),
        "Numeric_Statistics": get_numeric_stats(df),
        "Outliers": detect_outliers(df),
        "Example_Rows": get_example_rows(df, n=3),
    }

def main():
    st.title("Data File Analyzer")

    st.sidebar.title("AI Exploratory Data Analysis (AIEDA)")
    st.sidebar.write("🌿 Made by Kinto")


    st.sidebar.write("-----")


    
    # Sidebar components
    st.sidebar.header("Example file")
    selected_example = st.sidebar.selectbox(
        "Choose example data",
        options=list(EXAMPLE_FILES.keys()),
        key="selected_example"
    )
    st.sidebar.write("or")

    st.sidebar.header("Custom file")
    st.session_state.uploaded_file = st.sidebar.file_uploader(
        "Choose a csv or xlsx file",
        type=["csv", "xlsx"],
        key="file_uploader"
    )

    # Determine which file to load
    file_to_load = None
    if st.session_state.uploaded_file:
        file_to_load = st.session_state.uploaded_file
    elif selected_example and EXAMPLE_FILES[selected_example]:
        file_to_load = EXAMPLE_FILES[selected_example]

    # Load data and compute analyses
    if file_to_load:
        try:
            st.session_state.df = load_data(file_to_load)
            st.session_state.analyses = compute_analyses(st.session_state.df)

        except ValueError as e:
            st.error(f"Error loading file: {e}")
            st.session_state.df = None
            st.session_state.analyses = {}

    if st.session_state.loaded_file_path != file_to_load:
        st.session_state.loaded_file_path = file_to_load
        st.session_state.preprocessing_tips = ""
        st.session_state.feature_tips = ""

    if selected_example=="Example Data 2 (with pre-generated tips)" and file_to_load == EXAMPLE_FILES["Example Data 2 (with pre-generated tips)"]:
        st.session_state.preprocessing_tips = "The data is relatively clean with only 4 missing values and no duplicate rows, but there are some inconsistencies in the 'Country' and 'Month' columns.\n\n- Remove leading/trailing whitespaces from 'Country' column values to ensure consistency.\n- Correct the typo in 'Month' column value 'Decmber' to 'December'.\n- Replace the missing value in 'Age_Group' column with a suitable value or impute it based on other columns.\n- Handle the outliers in 'Customer_Age', 'Unit_Price', 'Profit', 'Cost', and 'Revenue' columns by either removing or imputing them.\n- Consider encoding categorical columns like 'Month', 'Age_Group', 'Customer_Gender', 'Country', 'State', 'Product_Category', and 'Sub_Category' for better analysis."
        st.session_state.feature_tips = "- Extract seasonality features from the 'Date' column to capture periodic patterns in sales.\n- Create a new feature 'Average Order Value' by dividing 'Revenue' by 'Order_Quantity' to analyze customer spending habits.\n- Engineer a 'Customer Age Group' feature by binning 'Customer_Age' into distinct groups to identify age-related trends.\n- Develop a 'Product Complexity' feature by extracting relevant information from 'Product_Description', such as bike type or size, to analyze product-specific sales.\n- Calculate a 'Profit Margin' feature by dividing 'Profit' by 'Revenue' to evaluate the profitability of each sale."


    st.sidebar.write("-----")

    if st.session_state.df is not None:

        st.sidebar.header("Feature Engineering")
        selected_data_goal = st.sidebar.selectbox(
            "Choose data objective",
            options=list(DATA_PURPOSE.keys()),
            key="selected_goal"
        )
        # if selected_data_goal:
        selected_data_goal_column = st.sidebar.selectbox(
            "Choose column",
            options=["None"]+list(st.session_state.analyses["Column_Data_Types"]["data_types"].keys()),
            key="selected_goal_column"
        )

        generate_tips = st.sidebar.button("Generate Tips", use_container_width=True)
        hide_tips = st.sidebar.button("Remove Tips", use_container_width=True)

    st.sidebar.write("-----")

    st.sidebar.header("Split Data")


    # Input for percentages
    random_seed = st.sidebar.number_input("Random Seed (Ignore if unknown)", min_value=0, value=0)
    train_size = st.sidebar.number_input("Training set percentage (1-100)", min_value=1, max_value=100, value=80)
    val_size = st.sidebar.number_input("Validation set percentage (0-100, leave 0 for no validation)", min_value=0, max_value=100, value=0)
    test_size = st.sidebar.number_input("Test set percentage (1-100)", min_value=1, max_value=100, value=20)

    # Input for random seed

    total_size = train_size + val_size + test_size
    if total_size > 100:
        st.sidebar.error("The total percentage exceeds 100%. Please adjust the values.")
    if total_size < 100:
        st.sidebar.error("The total percentage is less than 100%. Please adjust the values.")
    else:
        if st.sidebar.button("Split Data", use_container_width=True):
            # Convert percentages to proportions
            train_size /= 100
            val_size /= 100
            test_size /= 100
            
            # Split the data
            train_data, val_data, test_data = split_data(st.session_state.df, train_size, val_size, test_size, random_seed)

            # Create a zip file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                zip_file.writestr("train_data.csv", train_data.to_csv(index=False).encode())
                if not val_data.empty:
                    zip_file.writestr("val_data.csv", val_data.to_csv(index=False).encode())
                zip_file.writestr("test_data.csv", test_data.to_csv(index=False).encode())
            
            zip_buffer.seek(0)

            # Download button for the zip file
            st.sidebar.download_button("Download All Datasets as ZIP", zip_buffer, "datasets.zip", "application/zip", use_container_width=True)

    st.sidebar.write("-----")

    st.sidebar.header("🌟 Support me")

    st.sidebar.link_button("Buy me a coffee :)", "https://buymeacoffee.com/kinto", type="primary", help="Support me", icon="☕",use_container_width=True)



    # Display results
    if st.session_state.df is not None:

        # Section 1 ==============

        # Collecting data for the one-row DataFrame
        data = {
            "Number of Columns": [len(st.session_state.analyses["Column_Data_Types"]["data_types"])],
            "Number of Rows": [st.session_state.analyses["Table_Shape"]["rows"]],
            "Numeric Columns": [len(st.session_state.analyses["Numeric_Statistics"]["numeric_statistics"])],
            "Categorical Columns": [len(st.session_state.analyses["Unique_Values"]["unique_values"])],
            "Total N/A Values": [st.session_state.analyses["Missing_Values"]["total_missing"]],
            "Number of duplicate rows": [st.session_state.analyses["Overall_duplicates_nas"]["duplicate_count"]]
        }

        # Basic dataset statistics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Number of Columns", len(st.session_state.analyses["Column_Data_Types"]["data_types"]), border=True)
        col2.metric("Number of Rows", st.session_state.analyses["Table_Shape"]["rows"], border=True)
        col3.metric("Numeric Columns", len(st.session_state.analyses["Numeric_Statistics"]["numeric_statistics"]), border=True)
        col4.metric("Categorical Columns", len(st.session_state.analyses["Unique_Values"]["unique_values"]), border=True)
        col5.metric("Total N/A Values", st.session_state.analyses["Missing_Values"]["total_missing"], border=True)
        col6.metric("Number of duplicate rows", st.session_state.analyses["Overall_duplicates_nas"]["duplicate_count"], border=True)


        # Show examples ==============

        st.dataframe(st.session_state.df.head(1000), use_container_width=True)
    
        # LLM ==============

        if generate_tips:
            input = str(st.session_state.analyses)
            print(input)
            print(f"\nLength: {len(input)}\n")
            with st.spinner("generating"):
                try:
                    st.session_state.preprocessing_tips = gen_preprocessing_tips(input)
                    st.session_state.feature_tips = gen_feature_tips(input, goal=selected_data_goal, column=selected_data_goal_column)
                except:
                    print("\nBug Error: Generating file format.\n")
                    st.session_state.preprocessing_tips = ""
                    st.session_state.feature_tips = ""

        if hide_tips:
            st.session_state.preprocessing_tips = ""
            st.session_state.feature_tips = ""


        if st.session_state.preprocessing_tips:
            sec_1, sec_2 = st.columns(2)

            with sec_1:
                st.subheader("Data Preprocessing Tips")
                with st.spinner(""):
                    st.info(st.session_state.preprocessing_tips)


            with sec_2:
                st.subheader("Feature Engineering Tips")
                with st.spinner(""):
                    st.info(st.session_state.feature_tips)


        # Section 2 ==============

        st.subheader("Further Insight")

        with st.expander("Column Type Analysis", expanded=False):
            st.subheader("Column Type Analysis")

            col2, col1 = st.columns(2)
            with col1:
                try:
                    data_types = pd.Series(st.session_state.analyses["Column_Data_Types"]["data_types"])
                    fig = px.pie(names=data_types.value_counts().index, 
                                values=data_types.value_counts().values,
                                title="Distribution of Data Types",
                                template = 'seaborn')
                    st.plotly_chart(fig, theme="streamlit")
                except:
                    ...
            
            with col2:
                st.dataframe(pd.DataFrame({"Column": data_types.index, "Type": data_types.values}), use_container_width=True)


        # Section 3 ==============

        with st.expander("Missing Values Analysis"):
            st.subheader("Missing Values Analysis")
            sec_4, sec_3 = st.columns(2)
            with sec_3:
                try:
                    missing_data = pd.Series(st.session_state.analyses["Missing_Values"]["missing_values"])
                    fig = px.line(x=missing_data.index, y=missing_data.values,
                                title="Missing Values by Column",
                                labels={"x": "Column", "y": "Number of Missing Values"},
                                template = 'seaborn')
                    st.plotly_chart(fig, theme='streamlit')
                except:
                    ...

            with sec_4:
                outliers_df = pd.DataFrame.from_dict(st.session_state.analyses["Missing_Values"]["missing_values"], orient='index')
                st.write("Missing Values Summary:")
                st.dataframe(outliers_df.reset_index().rename(columns={'index': 'Column Name', 0:'# Missing Values'}), use_container_width=True, hide_index=True)
            

        # Section 4 ==============

        with st.expander("Unique Values Analysis"):
            st.subheader("Unique Categorical Values Analysis")
            unique_data = pd.Series(st.session_state.analyses["Unique_Values"]["unique_values"])
            y_uniques = [i['count'] for i in unique_data]

            try:
                fig = px.line(x=unique_data.index, y=y_uniques,
                    title="Unique Categorical Values by Column",
                    labels={"x": "Column", "y": "Number of unique values"},
                    template = 'seaborn')
                st.plotly_chart(fig, theme='streamlit')
            except:
                ...

            # with sec_2:
            # Assuming your data is structured in the session state
            data = []
            for col, info in st.session_state.analyses["Unique_Values"]["unique_values"].items():
                data.append({
                    'Column': col,
                    'Unique Values Count': info['count'],
                    'Values (n=10)': info['values'],
                })
            st.table(pd.DataFrame(data))  # or use st.dataframe(df) for an interactive table


        # Section 5 ==============
        with st.expander("Numeric Statistics"):
            st.header("Numeric Statistics")
            col_a,col_b = st.columns(2)
            try:
                # Select column for detailed view
                selected_column = st.selectbox(
                    "Select column for detailed statistics",
                    list(st.session_state.analyses["Numeric_Statistics"]["numeric_statistics"].keys())
                )
                stats = st.session_state.analyses["Numeric_Statistics"]["numeric_statistics"][selected_column]
            except:
                ...

            with col_a:
                try:
                    # Display statistics - Added formatting and delta colors
                    st.metric("Mean", f"{stats['mean']:.2f}")#, delta=f"Median: {stats['median']:.2f}", delta_color="off")
                    st.metric("Median", f"{stats['median']:.2f}")
                    st.metric("Std", f"{stats['std']:.2f}")
                    st.metric("Skew", f"{stats['skew']:.2f}")
                    st.metric("Range", f"{stats['min']:.2f} - {stats['max']:.2f}")
                except:
                    ...

            with col_b:
                try:
                    # Create enhanced box plot
                    fig = go.Figure()
                    # Add proper box plot with actual data points
                    fig.add_trace(go.Box(
                        y=st.session_state.df[selected_column],
                        name=selected_column,
                        boxpoints='outliers',  # Show outliers
                        marker_color='#1f77b4',  # Custom color
                        line_color='darkblue',
                        fillcolor='lightblue',
                        hoverinfo='y'
                    ))

                    # Add styling and layout improvements
                    fig.update_layout(
                        title=f"{selected_column} Distribution",
                        title_font_size=20,
                        title_x=0.3,
                        yaxis_title=selected_column,
                        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=40, r=40, t=60, b=40),
                        height=500,
                        width=800,
                        showlegend=False
                    )

                    # Add mean line annotation
                    fig.add_vline(x=0.5, line_dash="dot", 
                                line_color="red", 
                                annotation_text=f"Mean: {stats['mean']:.2f}",
                                annotation_position="top right")

                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                except:
                    ...

        # Section 6 ==============

        with st.expander("Outliers Analysis"):
            st.header("Outliers Analysis")
            # Create bar chart of missing values
            outliers_data = pd.Series(st.session_state.analyses["Outliers"]["outliers"])

            try:
                y_outliers = [i['num_outlier_values'] for i in outliers_data]
                fig = px.line(x=outliers_data.index, y=y_outliers,
                    title="Outlier Values by Column",
                    labels={"x": "Column", "y": "Number of outlier values"},
                    template = 'seaborn')
                st.plotly_chart(fig, theme='streamlit')
            except:
                ...

            outliers_df = pd.DataFrame.from_dict(st.session_state.analyses["Outliers"]["outliers"], orient='index')
            st.write("Outliers Summary:")
            st.dataframe(outliers_df.reset_index().rename(columns={'index': 'Column'}), use_container_width=True)

    else:
        st.info("Please upload a file or select an example to begin analysis.")

if __name__ == "__main__":
    main()
