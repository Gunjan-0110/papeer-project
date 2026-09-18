# Papeer — Research Paper Assistant

A conversational AI assistant for students and researchers to upload, explore, and verify academic papers through natural language chat.

---

## Project Description

Papeer is a Retrieval-Augmented Generation (RAG) application built with LangGraph, LangChain, and Streamlit. Users upload research papers (PDF, TXT, Markdown, web URL, or ArXiv ID) into isolated sessions, then ask questions about them. The system routes each query intelligently — answering directly from paper content, searching the web for current developments, or verifying whether a claim from a paper has been superseded by newer research.

---

## Target Users

- **Students** reading and trying to understand dense academic papers
- **Researchers** who want to quickly cross-reference claims across multiple papers
- **Literature reviewers** checking whether findings or methods from older papers still hold today
- **Anyone** who wants a conversational interface to a set of documents without manual reading

---

## Features & Capabilities

| Feature | Description |
|---|---|
| **Paper Q&A** | Ask questions about uploaded papers; the system retrieves relevant chunks and generates grounded answers |
| **Claim Verification** | Ask the assistant to verify a claim — it searches the web and ArXiv to determine if the claim is current or superseded, and returns links to newer papers if applicable |
| **Web Search** | For questions about current developments or explicit search requests, live Tavily results are incorporated |
| **Direct Answers** | General knowledge questions are answered without retrieval or web calls |
| **`/btw` Command** | A side-channel for off-topic questions outside the session context. The LLM decides to answer directly or search the web. These exchanges are **not stored in session history** |
| **Multi-session UI** | Open multiple independent sessions simultaneously, each with its own paper collection and conversation history |
| **Auto Session Naming** | Session titles are automatically generated (3–5 words) from the first message using the LLM |
| **Multiple Paper Sources** | Load papers via file upload (PDF, TXT, MD), web URL, or ArXiv ID/title search |
| **Graph State Inspector** | Each assistant turn exposes an expandable JSON view of the LangGraph state for debugging |
| **Streaming Responses** | Assistant responses stream token-by-token with a cursor animation |

---

## How to Use

### 1. Start a Session
Launch the app and a default session is created automatically. Use **New Chat** in the sidebar to start additional sessions.

### 2. Upload Papers
In the sidebar, choose one of three loading methods:
- **File Upload** — drag and drop a PDF, TXT, or MD file
- **Web URL** — paste one or more URLs (one per line)
- **ArXiv** — enter a paper title or ArXiv ID (e.g. `2303.08774`)

Loaded papers are listed under "Loaded Papers" in the sidebar.

### 3. Ask Questions
Type in the chat input. Example queries:
- *"What methodology does the paper use for evaluation?"*
- *"Verify the claim that encoder-decoder models are the best approach for translation."*
- *"What are the latest developments in diffusion models?"*

### 4. Use `/btw` for Off-Topic Questions
Prefix any message with `/btw` to ask a question outside the current paper context. These exchanges are not saved to the session:
```text
/btw What is the difference between attention mechanisms and RNNs?
```

---

## Installation & Setup

Papeer uses [uv](https://github.com/astral-sh/uv) for high-performance dependency management.

```bash
# Clone the repository
git clone <repo-url>
cd rag-papeer-project

# Install all dependencies
uv sync

# Copy the example env file and fill in your keys
cp .env.example .env

# Run the Streamlit app
uv run streamlit run app.py
```

To add a new dependency:
```bash
uv add <package-name>
```

To run a backend module directly (useful during development):
```bash
uv run python -m backend.<module_name>
```

---

## Required API Keys (Free Tier Compatible)

All keys are loaded from a `.env` file in the project root via `python-dotenv`.

| Variable | Purpose | Where to Get It |
|---|---|---|
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | LLM inference and embeddings (Free via Google AI Studio) | [aistudio.google.com](https://aistudio.google.com) |
| `TAVILY_API_KEY` | Web search for current developments and claim verification | [tavily.com](https://tavily.com) |
| `QDRANT_URL` | Qdrant Cloud endpoint for the vector store (Free Forever 1GB tier) | [cloud.qdrant.io](https://cloud.qdrant.io) |
| `QDRANT_API_KEY` | Authentication for Qdrant Cloud | [cloud.qdrant.io](https://cloud.qdrant.io) |

`.env` file format template:
```env
GOOGLE_API_KEY=your-gemini-api-key-here
TAVILY_API_KEY=tvly-...
QDRANT_URL=[https://your-cluster.qdrant.io](https://your-cluster.qdrant.io)
QDRANT_API_KEY=your-qdrant-api-key
```

---

## Architecture & Code Structure

```text
app.py (Streamlit UI)
│
├── backend/rag_graph.py       — LangGraph RAG workflow (router → retrieve/verify/direct → answer)
├── backend/btw_handler.py     — Off-topic /btw handler (streaming, not stored in history)
├── backend/vector_store.py    — Qdrant Cloud vector store with cached embeddings
├── backend/paper_loader.py    — Multi-source paper loader (PDF, TXT, MD, URL, ArXiv)
└── backend/models.py          — Pydantic models for routing and structured LLM outputs
```

### RAG Graph Decision Flow Visualization

```text
User Query
    │
    ▼
 Router (LLM)
    │
    ├── direct_answer ──────────────────────────► Generate Answer
    │
    ├── retrieve ──► Agent (retriever + web tools) ──► Relevancy Check
    │                        │                              │
    │                        │◄── Query Rewrite (max 3) ────┘
    │                        └──────────────────────────────► Generate Answer
    │
    └── verify_claim ──► Web Search + ArXiv Search ──► Verdict + Paper Links
```

---

## How the Project Is Production Optimized

| Optimization Strategy | Implementation Details |
|---|---|
| **Embedding Cache** | `CacheBackedEmbeddings` writes to `./embedding_cache/` so identical text is never re-embedded across sessions, reducing API call volume and latency. |
| **Session Isolation** | Each session gets its own Qdrant collection (`papeer_{session_id}`) and a separate LangGraph SQLite checkpointer thread to prevent cross-session data leakage. |
| **Graph Caching** | The LangGraph execution graph is compiled once using `@st.cache_resource` and cleanly reused across Streamlit reruns. |
| **Streaming Responses** | Leverages `graph.stream()` with message mode to render assistant responses token-by-token with an active cursor animation. |
| **Session Persistence** | `sessions.json` maps session metadata while SQLite saves complete state, enabling seamless restoration after app reboots. |
| **Temp File Cleanup** | Uploaded user files are written to a temporary path, processed immediately, and safely purged regardless of success or failure. |
| **Async Evaluation** | The evaluation pipeline throttles concurrency (3 workers, 5-second throttle delays) to stay safely within free API rate limits. |
| **ArXiv Reliability** | Claim verification implements two targeted Tavily searches (general web + `site:arxiv.org`) instead of flaky Python wrapper packages. |

---

## Constraints and Engineering Choices

| Constraint Rule | Engineering Rationale |
|---|---|
| **Max 3 Query Rewrites** | Caps retry loops before falling back to a direct LLM response, preventing infinite loops on ambiguous or unanswerable queries. |
| **Chunk Size 1000 / Overlap 200** | Balances high-precision retrieval against context continuity across text boundaries. The 200-char overlap catches sentences split across split vectors. |
| **Tavily Max 3 Results for `/btw`** | Limits token consumption and keeps context windows small for out-of-scope queries. |
| **`/btw` Exchanges Not Stored** | Prevents off-topic noise from polluting the permanent session vector history or confusing subsequent RAG turns. |
| **Session-Scoped Qdrant Collections** | Namespaces every collection by session UUID to guarantee complete data segregation. |
| **Dual Search for Claim Verification** | General web searches find blogs/news while `arxiv.org`-targeted queries fetch peer-reviewed papers. One alone would miss half the picture. |
| **`k=4` Default Retrieval Chunks** | Strikes the ideal balance between comprehensive source inclusion and token efficiency. |

---

## Evaluation

Papeer features an automated RAG evaluation framework (`evaluate.py`) driven by [DeepEval](https://github.com/confident-ai/deepeval).

### Quality Metrics (Passing Threshold: 0.7)

| Metric Name | What It Measures |
|---|---|
| **Contextual Precision** | Are the retrieved document chunks relevant to the user's query? |
| **Contextual Recall** | Does the retrieved context encompass all required ground-truth information? |
| **Contextual Relevancy** | Is the overall context appropriately scoped to the input and goal? |
| **Answer Relevancy** | Does the generated response directly answer the user prompt? |
| **Faithfulness** | Is the response strictly grounded in the retrieved text without hallucinations? |

### Execution Command

```bash
uv run python evaluate.py
```

- **Golden Generation**: On first launch, synthetic test cases are compiled from `documents/Openclaw_Research_Report.pdf` and stored in `goldens.json`.
- **Results Logging**: Execution metrics, assertion pass/fail flags, and diagnostics write directly to `eval_results.json`.
- **Caching**: Future runs read from cached goldens unless `goldens.json` is cleared.
