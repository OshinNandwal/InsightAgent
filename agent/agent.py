import pandas as pd

from agent.tools import (
    calculate_total_revenue,
    calculate_total_profit,
    calculate_total_units,
    revenue_by_region,
    profit_by_region,
    revenue_by_product,
    profit_by_product,
    profit_margin_by_product,
    monthly_revenue,
)

from rag.retriever import BusinessKnowledgeRetriever
from utils.llm import generate_response


class InsightAgent:
    """AI business intelligence agent."""

    def __init__(self, knowledge_path):
        self.retriever = BusinessKnowledgeRetriever(
            knowledge_path
        )

    def select_tool(self, question):
        """Select the appropriate analytics tool."""

        question_lower = question.lower()

        if (
            "highest revenue" in question_lower
            or "revenue by region" in question_lower
        ):
            return "revenue_by_region"

        if (
            "highest profit" in question_lower
            or "profit by product" in question_lower
        ):
            return "profit_by_product"

        if "revenue by product" in question_lower:
            return "revenue_by_product"

        if "profit by region" in question_lower:
            return "profit_by_region"

        if "profit margin" in question_lower:
            return "profit_margin_by_product"

        if (
            "monthly revenue" in question_lower
            or "month" in question_lower
        ):
            return "monthly_revenue"

        if "total revenue" in question_lower:
            return "calculate_total_revenue"

        if "total profit" in question_lower:
            return "calculate_total_profit"

        if "units" in question_lower:
            return "calculate_total_units"

        return "dataset_summary"

    def run_tool(self, df, tool_name):
        """Execute the selected analytics tool."""

        tools = {
            "revenue_by_region": revenue_by_region,
            "profit_by_product": profit_by_product,
            "revenue_by_product": revenue_by_product,
            "profit_by_region": profit_by_region,
            "profit_margin_by_product": profit_margin_by_product,
            "monthly_revenue": monthly_revenue,
            "calculate_total_revenue": calculate_total_revenue,
            "calculate_total_profit": calculate_total_profit,
            "calculate_total_units": calculate_total_units,
        }

        if tool_name in tools:

            return tools[tool_name](df)

        return df.describe(include="all")

    def analyze(self, df, question):
        """Analyze a business question."""

        # =====================================================
        # 1. SELECT ANALYTICS TOOL
        # =====================================================

        tool_name = self.select_tool(question)


        # =====================================================
        # 2. EXECUTE ANALYTICS TOOL
        # =====================================================

        result = self.run_tool(
            df,
            tool_name,
        )


        # =====================================================
        # 3. RETRIEVE RELEVANT KNOWLEDGE USING RAG
        # =====================================================

        knowledge_results = self.retriever.search(
            question,
            top_k=2,
        )


        knowledge = "\n\n".join(
            item["text"]
            for item in knowledge_results
        )


        # =====================================================
        # 4. FORMAT ANALYTICAL RESULT
        # =====================================================

        if isinstance(
            result,
            pd.Series,
        ):

            result_text = result.to_string()

        elif isinstance(
            result,
            pd.DataFrame,
        ):

            result_text = result.to_string()

        else:

            result_text = str(result)


        # =====================================================
        # 5. GENERATE AI INTERPRETATION
        # =====================================================

        prompt = f"""
You are InsightAgent, an AI business intelligence analyst.

Answer the user's business question using ONLY the
provided analytical result and business knowledge.

User question:
{question}

Analytics tool used:
{tool_name}

Analytical result:
{result_text}

Retrieved business knowledge:
{knowledge}

Instructions:
1. Give a clear direct answer.
2. Use the numerical evidence provided.
3. Do not invent data.
4. Explain the business meaning.
5. Give one practical recommendation when appropriate.
6. Clearly distinguish facts from possible explanations.

Answer:
"""

        answer = generate_response(prompt)


        # =====================================================
        # 6. RETURN STRUCTURED AGENT RESPONSE
        # =====================================================

        return {
            "answer": answer,
            "tool_used": tool_name,
            "result": result,
            "knowledge": knowledge_results,
        }