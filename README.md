# Google ADK Research Agent

A production-style multi-agent research pipeline built with [Google ADK](https://google.github.io/adk-docs/) and traced end-to-end with [LangSmith](https://smith.langchain.com).

Give it a topic. It plans queries, searches the web, fact-checks key claims, and delivers a structured report — automatically, with every step fully observable in LangSmith.

---

## What This Demo Shows

Multi-agent systems are powerful, but they are notoriously hard to debug and optimize. This demo pairs two complementary technologies to solve that:

- **Google ADK** handles agent orchestration — sequential execution, tool use, and state handoff between agents via session state.
- **LangSmith** provides full-stack observability — every LLM call, tool invocation, token count, and latency is captured in a nested trace, automatically.

Together, they demonstrate what production-quality agentic AI looks like: modular, inspectable, and cost-aware.

---

## Tracing with LangSmith — Set Env Vars + One Function Call

Google ADK has **native LangSmith support built in**. Getting full observability is as simple as setting your environment variables and calling one function at startup:

**Step 1 — Set env vars** (in `.env` or your shell):

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_api_key
LANGSMITH_PROJECT=your_project_name
```

**Step 2 — Call once before creating agents:**

```python
from langsmith.integrations.google_adk import configure_google_adk

configure_google_adk()
```

That's it. No decorators. No callbacks. No manual span creation. No wrapping your LLM client. Because ADK has native LangSmith support, the integration hooks directly into the framework internals — from that point, LangSmith **automatically captures everything**: every agent invocation, LLM call, tool call, input, output, token count, latency, and cost across the entire multi-agent pipeline, in a fully nested trace.

---

## Why LangSmith?

When a multi-agent pipeline produces a bad output, the question is never "did it fail?" — it is "which step failed, why, and how much did it cost to get there?"

LangSmith answers that without any extra instrumentation work:

**Deep trace visibility.** Every run produces a fully nested trace: `session → pipeline → agent → LLM call / tool call`. Click into any node to inspect the exact prompt sent, the model's raw response, and the latency at that step.

**Tool-level debugging.** Web searches and page extractions are captured as individual spans. When the fact-checker reaches a wrong conclusion, you can see exactly which URLs it read and what it found.

**Cost and token tracking.** Token usage is recorded at every LLM call. Across a batch of research questions, you get a clear picture of where tokens are spent and where prompts can be tightened.

**Multi-agent state inspection.** ADK passes data between agents via session state. LangSmith makes that state visible at each handoff, so you can verify that the query planner's output actually reached the web researcher intact.

**Replay and comparison.** Once traces are captured, you can rerun individual steps against prompt edits or model swaps and compare results side by side in the LangSmith UI.

---

## Architecture

The pipeline is a `SequentialAgent` composed of four specialized sub-agents. Each agent reads from and writes to a shared ADK session state, passing its output forward as the next agent's input.

```
 Research Topic (input)
        │
        ▼
┌───────────────────┐
│  1. query_planner │  Breaks the topic into 3 focused search queries
└────────┬──────────┘
         │  session_state["search_queries"]
         ▼
┌────────────────────┐
│  2. web_researcher │  Runs searches + deep-reads top pages (Tavily)
└────────┬───────────┘
         │  session_state["raw_findings"]
         ▼
┌──────────────────┐
│  3. fact_checker │  Verifies the top 3 claims with additional searches
└────────┬─────────┘
         │  session_state["fact_check_results"]
         ▼
┌─────────────────┐
│  4. synthesizer │  Produces a structured report with cited sources
└─────────────────┘
        │
        ▼
 Structured Report (output)
```

**State flow.** ADK's `output_key` mechanism writes each agent's result into the session state dictionary under a named key. The next agent receives that key as part of its context. No manual wiring required.

**Tool use.** The web researcher and fact checker both have access to two Tavily-backed tools: `web_search` (returns ranked results with a summary) and `extract_page_content` (fetches and parses the full text of a URL). A typical run generates 3 searches and 2 page extractions in the research step, and 3 additional verification searches in the fact-check step.

---

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- API keys for [Google AI Studio](https://aistudio.google.com), [Tavily](https://tavily.com), and [LangSmith](https://smith.langchain.com)

---

## Setup

**1. Clone the repo and install dependencies:**

```bash
git clone https://github.com/liam-langchain/Google-ADK-Demo-Agent.git
cd Google-ADK-Demo-Agent
uv sync
```

**2. Copy the env template and fill in your keys:**

```bash
cp .env.example .env
```

```
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=lsv2_pt_...        # https://smith.langchain.com
LANGSMITH_PROJECT=google-adk-research-agent

GOOGLE_API_KEY=...                   # https://aistudio.google.com
TAVILY_API_KEY=tvly-...              # https://tavily.com
```

**3. Create the LangSmith project** (first time only):

```bash
curl -X POST "https://api.smith.langchain.com/api/v1/sessions" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $LANGSMITH_API_KEY" \
  -d '{"name": "google-adk-research-agent"}'
```

**4. Run the pipeline:**

```bash
uv run main.py
```

---

## Customization

**Change the research topics.** Edit `RESEARCH_QUESTIONS` in `main.py`:

```python
RESEARCH_QUESTIONS = [
    "Your first research question here",
    "Your second research question here",
]
```

**Change the model.** Update the `MODEL` constant in `main.py`:

```python
MODEL = "gemini-3-flash-preview"
```

---

## Viewing Traces in LangSmith

After running the pipeline, open your LangSmith project at [smith.langchain.com](https://smith.langchain.com). Each research question produces one top-level trace with the following structure:

```
google_adk.session
└── research_pipeline
    ├── query_planner      LLM call
    ├── web_researcher     LLM call → web_search × 3 → extract_page_content × 2
    ├── fact_checker       LLM call → web_search × 3
    └── synthesizer        LLM call
```

Click any span to inspect the full input, output, token usage, and latency for that step. Use the **Comparison** view to diff results across multiple runs when iterating on prompts or models.

---

## Learn More

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [LangSmith Tracing with Google ADK](https://docs.langchain.com/langsmith/trace-with-google-adk)
- [LangSmith Documentation](https://docs.smith.langchain.com)
- [Tavily API](https://docs.tavily.com)
