import pandas as pd


def load_data(file_path):
    """Load a CSV or Excel file."""
    if file_path.lower().endswith(".csv"):
        return pd.read_csv(file_path)

    if file_path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)

    raise ValueError("Unsupported file format. Please use CSV or Excel.")


def get_dataset_summary(df):
    """Return basic information about the dataset."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
    }


def get_numeric_summary(df):
    """Return statistical summary for numeric columns."""
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return pd.DataFrame()

    return numeric_df.describe()


def search_data(df, keyword):
    """Search for a keyword across text columns."""
    keyword = str(keyword).lower()

    text_columns = df.select_dtypes(include="object").columns

    if len(text_columns) == 0:
        return pd.DataFrame()

    mask = df[text_columns].apply(
        lambda column: column.astype(str).str.lower().str.contains(
            keyword, na=False
        )
    ).any(axis=1)

    return df[mask]