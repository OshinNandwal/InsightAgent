import pandas as pd


def calculate_total_revenue(df):
    """Calculate total revenue."""
    return float(df["Revenue"].sum())


def calculate_total_profit(df):
    """Calculate total profit."""
    return float(df["Profit"].sum())


def calculate_total_units(df):
    """Calculate total units sold."""
    return int(df["Units_Sold"].sum())


def revenue_by_region(df):
    """Calculate revenue by region."""
    return (
        df.groupby("Region")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )


def profit_by_region(df):
    """Calculate profit by region."""
    return (
        df.groupby("Region")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )


def revenue_by_product(df):
    """Calculate revenue by product."""
    return (
        df.groupby("Product")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )


def profit_by_product(df):
    """Calculate profit by product."""
    return (
        df.groupby("Product")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )


def profit_margin_by_product(df):
    """Calculate profit margin for each product."""
    grouped = df.groupby("Product")[["Revenue", "Profit"]].sum()

    grouped["Profit_Margin"] = (
        grouped["Profit"] / grouped["Revenue"] * 100
    )

    return grouped.sort_values(
        "Profit_Margin",
        ascending=False
    )


def monthly_revenue(df):
    """Calculate revenue by month."""
    data = df.copy()

    data["Date"] = pd.to_datetime(data["Date"])

    result = (
        data.groupby(data["Date"].dt.to_period("M"))["Revenue"]
        .sum()
    )

    result.index = result.index.astype(str)

    return result.sort_index()