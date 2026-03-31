# Google ADK Research Agent

A multi-agent research pipeline built with [Google ADK](https://google.github.io/adk-docs/) and traced end-to-end with [LangSmith](https://smith.langchain.com).

Given a research topic, the pipeline automatically plans queries, searches the web, fact-checks key claims, and synthesizes a structured report — all with full observability in LangSmith.

## Architecture

```
research_pipeline (SequentialAgent)
├── 1. query_planner    → breaks topic into 3 focused search queries
├── 2. web_researcher   → runs searches + deep-reads top pages (Tavily)
├── 3. fact_checker     → verifies top 3 claims with additional searches
└── 4. synthesizer      → produces a structured report with sources
```

Each agent passes its output to the next via ADK session state (`output_key`). Every step — agent calls, tool calls, LLM completions — is automatically traced to LangSmith.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)
- API keys for Google AI Studio, Tavily, and LangSmith

## Setup

**1. Clone the repo and install dependencies:**

```bash
git clone <repo-url>
cd google-adk-research-agent
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

**4. Run:**

```bash
uv run main.py
```

## Customisation

Edit `RESEARCH_QUESTIONS` in `main.py` to change the topics:

```python
RESEARCH_QUESTIONS = [
    "Your first research question here",
    "Your second research question here",
]
```

Change the model by updating `MODEL`:

```python
MODEL = "gemini-3-flash-preview"
```

## Viewing Traces

Open your LangSmith project to see the full nested trace for each research topic:

```
google_adk.session
└── research_pipeline
    ├── query_planner      (LLM call)
    ├── web_researcher     (LLM call + web_search × 3 + extract_page_content × 2)
    ├── fact_checker       (LLM call + web_search × 3)
    └── synthesizer        (LLM call)
```
