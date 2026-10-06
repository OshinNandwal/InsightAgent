import os

import pandas as pd
import streamlit as st

from agent.agent import InsightAgent
from agent.charts import (
    create_bar_chart,
    create_line_chart,
)

from agent.tools import (
    profit_by_product,
    profit_by_region,
    profit_margin_by_product,
    revenue_by_product,
    revenue_by_region,
    monthly_revenue,
)

from utils.data_tools import get_dataset_summary


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="InsightAgent",
    page_icon="🤖",
    layout="wide",
)


# =========================================================
# HEADER
# =========================================================

st.title("🤖 InsightAgent")

st.caption(
    "AI-powered Business Intelligence & Analytics Agent"
)


# =========================================================
# LOAD AGENT
# =========================================================

KNOWLEDGE_PATH = "documents/business_knowledge.txt"


@st.cache_resource
def load_agent():
    return InsightAgent(KNOWLEDGE_PATH)


agent = load_agent()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx"],
)


# =========================================================
# LOAD DATASET
# =========================================================

if uploaded_file is not None:

    if uploaded_file.name.lower().endswith(".csv"):

        df = pd.read_csv(uploaded_file)

    else:

        df = pd.read_excel(uploaded_file)

else:

    default_path = "data/sales_data.csv"

    if os.path.exists(default_path):

        df = pd.read_csv(default_path)

    else:

        df = None


# =========================================================
# MAIN APPLICATION
# =========================================================

if df is None:

    st.info(
        "Upload a business dataset from the sidebar to begin."
    )

else:

    # =====================================================
    # DATASET OVERVIEW
    # =====================================================

    st.subheader("📊 Dataset Overview")

    summary = get_dataset_summary(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Rows",
            summary["rows"]
        )

    with col2:

        st.metric(
            "Columns",
            summary["columns"]
        )

    with col3:

        st.metric(
            "Missing Values",
            summary["missing_values"]
        )

    with col4:

        st.metric(
            "Dataset Size",
            f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
        )

    st.divider()


    # =====================================================
    # DATASET PREVIEW
    # =====================================================

    with st.expander("🔍 Preview Dataset"):

        st.dataframe(
            df.head(10),
            width="stretch"
        )


    # =====================================================
    # AI ANALYST
    # =====================================================

    st.subheader("🤖 Ask InsightAgent")

    question = st.text_input(
        "Ask a business question",
        placeholder=(
            "Example: Which region generated "
            "the highest revenue?"
        )
    )


    # =====================================================
    # ANALYZE
    # =====================================================

    if st.button(
        "Analyze",
        type="primary",
        width="stretch"
    ):

        if not question.strip():

            st.warning(
                "Please enter a business question."
            )

        else:

            with st.spinner(
                "InsightAgent is analyzing your data..."
            ):

                try:

                    # =====================================
                    # RUN AGENT
                    # =====================================

                    agent_result = agent.analyze(
                        df,
                        question
                    )

                    st.success(
                        "Analysis complete"
                    )


                    # =====================================
                    # AGENT DECISION
                    # =====================================

                    st.markdown(
                        "### 🧠 Agent Decision"
                    )

                    st.info(
                        f"InsightAgent selected the "
                        f"`{agent_result['tool_used']}` "
                        "analytics tool."
                    )


                    # =====================================
                    # AGENT WORKFLOW
                    # =====================================

                    st.markdown(
                        "#### 🔄 Agent Workflow"
                    )

                    st.caption(
                        "Question → Tool Selection → "
                        "Data Analysis → RAG Context → "
                        "AI Interpretation"
                    )


                    # =====================================
                    # RETRIEVED RAG KNOWLEDGE
                    # =====================================

                    with st.expander(
                        "📚 Retrieved Business Knowledge"
                    ):

                        for item in agent_result["knowledge"]:

                            st.markdown(
                                f"**Relevant knowledge:** "
                                f"{item['text']}"
                            )


                    # =====================================
                    # ANALYTICAL RESULT
                    # =====================================

                    st.markdown(
                        "### 📊 Analytical Result"
                    )

                    result = agent_result["result"]


                    # =====================================
                    # REVENUE BY REGION
                    # =====================================

                    if (
                        agent_result["tool_used"]
                        == "revenue_by_region"
                    ):

                        top_region = result.idxmax()

                        top_revenue = result.max()

                        metric_col1, metric_col2 = (
                            st.columns(2)
                        )

                        with metric_col1:

                            st.metric(
                                "🏆 Top Region",
                                top_region
                            )

                        with metric_col2:

                            st.metric(
                                "💰 Revenue",
                                f"₹{top_revenue:,.0f}"
                            )

                        with st.expander(
                            "View regional breakdown"
                        ):

                            st.dataframe(
                                result.rename(
                                    "Revenue"
                                ),
                                width="stretch"
                            )


                    # =====================================
                    # PROFIT BY PRODUCT
                    # =====================================

                    elif (
                        agent_result["tool_used"]
                        == "profit_by_product"
                    ):

                        top_product = result.idxmax()

                        top_profit = result.max()

                        metric_col1, metric_col2 = (
                            st.columns(2)
                        )

                        with metric_col1:

                            st.metric(
                                "🏆 Top Product",
                                top_product
                            )

                        with metric_col2:

                            st.metric(
                                "💰 Profit",
                                f"₹{top_profit:,.0f}"
                            )

                        with st.expander(
                            "View product breakdown"
                        ):

                            st.dataframe(
                                result.rename(
                                    "Profit"
                                ),
                                width="stretch"
                            )


                    # =====================================
                    # REVENUE BY PRODUCT
                    # =====================================

                    elif (
                        agent_result["tool_used"]
                        == "revenue_by_product"
                    ):

                        top_product = result.idxmax()

                        top_revenue = result.max()

                        metric_col1, metric_col2 = (
                            st.columns(2)
                        )

                        with metric_col1:

                            st.metric(
                                "🏆 Top Product",
                                top_product
                            )

                        with metric_col2:

                            st.metric(
                                "💰 Revenue",
                                f"₹{top_revenue:,.0f}"
                            )

                        with st.expander(
                            "View product breakdown"
                        ):

                            st.dataframe(
                                result.rename(
                                    "Revenue"
                                ),
                                width="stretch"
                            )


                    # =====================================
                    # PROFIT BY REGION
                    # =====================================

                    elif (
                        agent_result["tool_used"]
                        == "profit_by_region"
                    ):

                        top_region = result.idxmax()

                        top_profit = result.max()

                        metric_col1, metric_col2 = (
                            st.columns(2)
                        )

                        with metric_col1:

                            st.metric(
                                "🏆 Top Region",
                                top_region
                            )

                        with metric_col2:

                            st.metric(
                                "💰 Profit",
                                f"₹{top_profit:,.0f}"
                            )

                        with st.expander(
                            "View regional breakdown"
                        ):

                            st.dataframe(
                                result.rename(
                                    "Profit"
                                ),
                                width="stretch"
                            )


                    # =====================================
                    # PROFIT MARGIN
                    # =====================================

                    elif (
                        agent_result["tool_used"]
                        == "profit_margin_by_product"
                    ):

                        top_product = (
                            result[
                                "Profit_Margin"
                            ].idxmax()
                        )

                        top_margin = (
                            result[
                                "Profit_Margin"
                            ].max()
                        )

                        metric_col1, metric_col2 = (
                            st.columns(2)
                        )

                        with metric_col1:

                            st.metric(
                                "🏆 Highest Margin",
                                top_product
                            )

                        with metric_col2:

                            st.metric(
                                "📈 Profit Margin",
                                f"{top_margin:.2f}%"
                            )

                        with st.expander(
                            "View margin breakdown"
                        ):

                            st.dataframe(
                                result,
                                width="stretch"
                            )


                    # =====================================
                    # MONTHLY REVENUE
                    # =====================================

                    elif (
                        agent_result["tool_used"]
                        == "monthly_revenue"
                    ):

                        highest_month = result.idxmax()

                        highest_revenue = result.max()

                        metric_col1, metric_col2 = (
                            st.columns(2)
                        )

                        with metric_col1:

                            st.metric(
                                "🏆 Best Month",
                                highest_month
                            )

                        with metric_col2:

                            st.metric(
                                "💰 Revenue",
                                f"₹{highest_revenue:,.0f}"
                            )

                        with st.expander(
                            "View monthly breakdown"
                        ):

                            st.dataframe(
                                result.rename(
                                    "Revenue"
                                ),
                                width="stretch"
                            )


                    # =====================================
                    # OTHER RESULTS
                    # =====================================

                    else:

                        if isinstance(
                            result,
                            pd.Series
                        ):

                            st.dataframe(
                                result.rename(
                                    "Value"
                                ),
                                width="stretch"
                            )

                        elif isinstance(
                            result,
                            pd.DataFrame
                        ):

                            st.dataframe(
                                result,
                                width="stretch"
                            )

                        else:

                            st.metric(
                                "Calculated Result",
                                str(result)
                            )


                    # =====================================
                    # AI INSIGHT
                    # =====================================

                    st.markdown(
                        "### 💡 AI Insight"
                    )

                    st.markdown(
                        agent_result["answer"]
                    )


                    # =====================================
                    # VISUALIZATION
                    # =====================================

                    question_lower = (
                        question.lower()
                    )

                    chart = None


                    if (
                        "highest revenue"
                        in question_lower
                        or "revenue by region"
                        in question_lower
                    ):

                        chart = create_bar_chart(
                            revenue_by_region(df),
                            "Revenue by Region",
                            "Region",
                            "Revenue"
                        )


                    elif (
                        "highest profit"
                        in question_lower
                        or "profit by product"
                        in question_lower
                    ):

                        chart = create_bar_chart(
                            profit_by_product(df),
                            "Profit by Product",
                            "Product",
                            "Profit"
                        )


                    elif (
                        "revenue by product"
                        in question_lower
                    ):

                        chart = create_bar_chart(
                            revenue_by_product(df),
                            "Revenue by Product",
                            "Product",
                            "Revenue"
                        )


                    elif (
                        "profit by region"
                        in question_lower
                    ):

                        chart = create_bar_chart(
                            profit_by_region(df),
                            "Profit by Region",
                            "Region",
                            "Profit"
                        )


                    elif (
                        "profit margin"
                        in question_lower
                    ):

                        chart = create_bar_chart(
                            profit_margin_by_product(df)[
                                "Profit_Margin"
                            ],
                            "Profit Margin by Product",
                            "Product",
                            "Profit Margin (%)"
                        )


                    elif (
                        "monthly revenue"
                        in question_lower
                        or "month"
                        in question_lower
                    ):

                        chart = create_line_chart(
                            monthly_revenue(df),
                            "Monthly Revenue Trend",
                            "Month",
                            "Revenue"
                        )


                    if chart is not None:

                        st.markdown(
                            "### 📈 Visual Analysis"
                        )

                        st.pyplot(chart)


                except Exception as e:

                    st.error(
                        "Something went wrong while "
                        f"analyzing the question: {e}"
                    )