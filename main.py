"""
Multi-Agent Research Pipeline using Google ADK + LangSmith Tracing

Architecture (SequentialAgent pipeline):
  1. query_planner   — breaks the topic into 3 focused search queries
  2. web_researcher  — runs searches and deep-reads the most relevant pages
  3. fact_checker    — cross-verifies the top 3 claims with additional searches
  4. synthesizer     — produces a structured report with sources

Tracing: every agent invocation, tool call, and LLM interaction is
automatically captured in LangSmith via configure_google_adk().
"""

import asyncio
import os
from dotenv import load_dotenv

# Must load env vars before any SDK imports so LangSmith
# picks up LANGSMITH_API_KEY / LANGSMITH_PROJECT at init time.
load_dotenv(override=True)

from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from langsmith.integrations.google_adk import configure_google_adk
from tavily import TavilyClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gemini-3-flash-preview"

RESEARCH_QUESTIONS = [
    "What are the latest AI breakthroughs and model releases in 2026?",
    "What are the biggest trends in renewable energy and climate tech in 2026?",
    "What are the top programming languages and frameworks developers are using in 2026?",
    "What are the latest developments in quantum computing in 2026?",
    "How is AI being used in drug discovery and healthcare in 2026?",
    "What are the major breakthroughs in space exploration in 2026?",
]

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
# Agents
# ---------------------------------------------------------------------------

# Step 1 — Query Planner
# Decomposes the research topic into 3 targeted search queries and saves
# them to session state under the key "search_queries".
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

# Step 2 — Web Researcher
# Executes each planned query, deep-reads the 2 most relevant pages, and
# saves all raw findings to session state under "raw_findings".
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

# Step 3 — Fact Checker
# Identifies the top 3 claims from the raw findings, runs a verification
# search for each, and saves verdicts to "fact_check_results".
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

# Step 4 — Synthesizer
# Combines raw findings and fact-check results into a final structured report.
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

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

research_pipeline = SequentialAgent(
    name="research_pipeline",
    description=(
        "4-step research pipeline: "
        "plan queries → search & extract → fact check → synthesize."
    ),
    sub_agents=[query_planner, web_researcher, fact_checker, synthesizer],
)

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


async def run_research(question: str, session_id: str, runner: Runner) -> None:
    print(f"\n{'=' * 60}")
    print(f"Researching: {question}")
    print(f"{'=' * 60}")

    async for event in runner.run_async(
        user_id="user_1",
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=question)],
        ),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text)


async def main() -> None:
    configure_google_adk()

    session_service = InMemorySessionService()
    runner = Runner(
        agent=research_pipeline,
        app_name="research_app",
        session_service=session_service,
    )

    for i, question in enumerate(RESEARCH_QUESTIONS):
        session = await session_service.create_session(
            app_name="research_app",
            user_id="user_1",
            session_id=f"session_{i}",
        )
        await run_research(question, session.id, runner)

    print(f"\n{'=' * 60}")
    print(f"All {len(RESEARCH_QUESTIONS)} research topics complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
