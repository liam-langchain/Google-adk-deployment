"""
LangGraph wrapper around the Google ADK research pipeline for LangSmith deployment.

The LangGraph StateGraph provides the deployment interface; the Google ADK
SequentialAgent pipeline runs inside the single node.
"""

import asyncio
import os
from typing import TypedDict

from dotenv import load_dotenv

load_dotenv(override=True)

from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from langsmith.integrations.google_adk import configure_google_adk
from langgraph.graph import END, START, StateGraph
from tavily import TavilyClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def web_search(query: str) -> dict:
    """Search the web for up-to-date information on a topic.

    Args:
        query: The search query string.

    Returns:
        A dict with 'answer' (quick summary) and 'results'
        (list of {title, url, content}).
    """
    response = tavily.search(query=query, max_results=5, include_answer=True)
    return {
        "answer": response.get("answer", ""),
        "results": [
            {"title": r["title"], "url": r["url"], "content": r["content"]}
            for r in response.get("results", [])
        ],
    }


def extract_page_content(url: str) -> dict:
    """Fetch and read the full text content of a specific webpage.

    Args:
        url: The URL to extract content from.

    Returns:
        A dict with 'url' and 'content' (raw text of the page).
    """
    response = tavily.extract(urls=[url])
    results = response.get("results", [])
    if results:
        return {"url": url, "content": results[0].get("raw_content", "")}
    return {"url": url, "content": "No content extracted."}


# ---------------------------------------------------------------------------
# Google ADK agents
# ---------------------------------------------------------------------------

query_planner = Agent(
    name="query_planner",
    model=MODEL,
    description="Decomposes a research topic into 3 focused, diverse search queries.",
    instruction=(
        "The user's research topic is in the conversation. "
        "Output exactly 3 distinct search queries that together give comprehensive coverage. "
        "Vary the angle: one news-focused, one technical, one trends/future. "
        "Return them as a plain numbered list with no extra commentary."
    ),
    output_key="search_queries",
)

web_researcher = Agent(
    name="web_researcher",
    model=MODEL,
    description="Executes web searches and extracts full content from the most relevant pages.",
    instruction=(
        "The search queries to execute are:\n{search_queries}\n\n"
        "For each query:\n"
        "1. Call web_search with that query.\n"
        "2. From all results combined, pick the 2 most relevant URLs and call "
        "extract_page_content on each.\n"
        "Return all raw findings clearly structured: search summaries followed "
        "by extracted page contents."
    ),
    tools=[web_search, extract_page_content],
    output_key="raw_findings",
)

fact_checker = Agent(
    name="fact_checker",
    model=MODEL,
    description="Identifies and verifies the top 3 key claims from research findings.",
    instruction=(
        "Here are the raw research findings:\n{raw_findings}\n\n"
        "Identify the 3 most important factual claims. "
        "For each claim, run a web_search to find corroborating or contradicting evidence. "
        "Return each claim with: the claim text, your web_search query, and a verdict "
        "(Verified / Unverified / Disputed) with a one-sentence justification."
    ),
    tools=[web_search],
    output_key="fact_check_results",
)

synthesizer = Agent(
    name="synthesizer",
    model=MODEL,
    description="Synthesizes research findings and fact-check results into a final structured report.",
    instruction=(
        "You have the following inputs:\n\n"
        "RAW RESEARCH FINDINGS:\n{raw_findings}\n\n"
        "FACT CHECK RESULTS:\n{fact_check_results}\n\n"
        "Write a final research report with:\n"
        "1. **Executive Summary** (2-3 sentences)\n"
        "2. **Key Findings** (bullet points with inline source citations)\n"
        "3. **Verified vs. Disputed Claims** (from the fact check)\n"
        "4. **Sources** (all URLs referenced)\n\n"
        "Be precise, professional, and well-structured."
    ),
)

research_pipeline = SequentialAgent(
    name="research_pipeline",
    description=(
        "4-step research pipeline: "
        "plan queries → search & extract → fact check → synthesize."
    ),
    sub_agents=[query_planner, web_researcher, fact_checker, synthesizer],
)

# ---------------------------------------------------------------------------
# LangGraph state and node
# ---------------------------------------------------------------------------


class State(TypedDict):
    question: str
    report: str


def research_node(state: State) -> State:
    """Run the Google ADK research pipeline for a given question."""
    configure_google_adk()

    session_service = InMemorySessionService()
    runner = Runner(
        agent=research_pipeline,
        app_name="research_app",
        session_service=session_service,
    )

    async def _run() -> str:
        session = await session_service.create_session(
            app_name="research_app",
            user_id="user_1",
            session_id="session_0",
        )
        report = ""
        async for event in runner.run_async(
            user_id="user_1",
            session_id=session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=state["question"])],
            ),
        ):
            if event.is_final_response():
                report = event.content.parts[0].text
        return report

    report = asyncio.run(_run())
    return {"report": report}


# ---------------------------------------------------------------------------
# Compiled graph — referenced by langgraph.json
# ---------------------------------------------------------------------------

_builder = StateGraph(State)
_builder.add_node("research", research_node)
_builder.add_edge(START, "research")
_builder.add_edge("research", END)

graph = _builder.compile()
