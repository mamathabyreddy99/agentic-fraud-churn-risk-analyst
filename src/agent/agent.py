from __future__ import annotations
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from .tools import (
    explain_flagged_case,
    get_flagged_cases,
    get_model_performance,
    get_top_risk_factors,
)

SYSTEM_PROMPT = """
You are an AI risk-analysis assistant inside a fraud/churn ML application.

Use tools for every quantitative claim about the user's uploaded data or trained model.
Never invent metrics, scores, or explanations.
Treat a risk score as a decision-support signal, not proof of fraud, misconduct, or customer intent.
Explain results in clear business language and mention relevant uncertainty or metric tradeoffs.
"""

def build_agent():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        temperature=0,
    )

    return create_agent(
        model=model,
        tools=[
            get_model_performance,
            get_top_risk_factors,
            explain_flagged_case,
            get_flagged_cases,
        ],
        system_prompt=SYSTEM_PROMPT,
    )

def ask_agent(question: str) -> str:
    agent = build_agent()
    response = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    final_message = response["messages"][-1]
    content = final_message.content
    return content if isinstance(content, str) else str(content)
