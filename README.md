# Just-in-Time Tools: Tool Search with Embeddings

A walkthrough of the Claude Cookbook recipe
[Tool search with embeddings](https://platform.claude.com/cookbook/tool-use-tool-search-with-embeddings)
in four small Python scripts, run with **uv**.

Claude starts with just **one** tool, `tool_search`. Every other tool is sent with
`defer_loading: true`, which keeps it out of Claude's context. When Claude needs a capability, it
searches, our embedding search returns `tool_reference` blocks, and the API loads only those tools.

## Project layout

```
jit-tools/
├── pyproject.toml            # uv project + dependencies
├── .env.example              # copy to .env and add your key
├── common.py                 # tool library, embeddings index, mock tools, client
├── demo_1_the_problem.py     # token cost: all tools vs just-in-time
├── demo_2_embeddings.py      # semantic tool search (local, no API key)
├── demo_3_search_loop.py     # Claude discovers + calls tools
└── demo_4_scale.py           # 208 tools, multi-domain question
```

## Setup

1. Install [uv](https://docs.astral.sh/uv/) if you don't have it:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Add your API key:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and set `ANTHROPIC_API_KEY=sk-ant-...`. You can also set `CLAUDE_MODEL` (default `claude-opus-5-5`).
3. Install dependencies. uv downloads a native Python 3.12 and creates `.venv`:
   ```bash
   uv sync
   ```

> **Conda users:** if a conda env is active (`(base)` in your prompt), uv may pick conda's Python,
> which can be the wrong architecture on Apple Silicon and has no `torch` wheel. Run
> `conda deactivate` first, or prefix the commands with `env -u CONDA_PREFIX`.

The first run of any embedding demo downloads the `all-MiniLM-L6-v2` model (~90 MB) from Hugging Face.

## Run the demos

| # | Command | API key? | What you'll see |
|---|---|---|---|
| 1 | `uv run demo_1_the_problem.py` | yes (token count only) | Table of input tokens for 8 / 208 tools, loaded vs. deferred, plus the % saved |
| 2 | `uv run demo_2_embeddings.py` | **no** | A tool as text, embedding shape `(8, 384)`, ranked matches for 5 queries |
| 3 | `uv run demo_3_search_loop.py` | yes | Turn-by-turn log: `[search]` → `tool_reference`s → `[call]` → Claude's answer, for a weather and a finance question |
| 4 | `uv run demo_4_scale.py` | yes | Multiple searches + parallel calls over 208 tools; "Tools ever loaded into context: N of 208" |

Demo 3 also takes your own question:

```bash
uv run demo_3_search_loop.py "What time is it in Tokyo, and how is AAPL doing?"
```

### Example output (Demo 2, deterministic)

```
Embeddings shape: (8, 384)  -> 8 tools x 384 dims

  Query: 'how much will my savings grow'
   1. calculate_compound_interest    0.327 #############
   2. convert_currency               0.140 #####
   3. get_market_news                0.102 ####
```

### Real results (claude-haiku-5-5)

| Demo | Result |
|---|---|
| 1 | 8 tools: 1,399 → 584 tokens (58% less). 208 tools: 33,679 → 584 tokens (98% less) |
| 3 | Weather: search → `get_weather` → answer in 3 turns. Interest: search → `calculate_compound_interest` → $16,470.09 |
| 4 | 3 searches in one turn, 3 parallel tool calls, **4 of 208** tools ever loaded, peak 2,315 input tokens (without the score cutoff: 6 tools, 2,686 tokens) |

Demos 3 and 4 use **mock** tool results (random weather, random rates), so the numbers change on every run,
and Claude's wording varies too. The compound-interest answer is real maths: **$16,470.09**.

## How it works (the 3 lines that matter)

```python
tools = [TOOL_SEARCH_DEFINITION] + [{**t, "defer_loading": True} for t in TOOL_LIBRARY]
# ...when Claude calls tool_search:
content = [{"type": "tool_reference", "tool_name": t["name"]} for t, _ in index.search(query)]
tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
```

- Send the **same** `tools` array on every request, deferred tools included.
- Keep at least one tool non-deferred, or the API returns a 400.
- Drop weak matches: hits scoring below `MIN_SCORE` (0.3, in `common.py`) are never returned, so irrelevant tools don't use up context.
- Append Claude's full `response.content` to the history each turn.

## Cost

Demo 1 uses the free `count_tokens` endpoint. Demos 3 and 4 make a few short Messages API calls each
(a few cents at most on Claude Opus 5.5). Set `CLAUDE_MODEL=claude-haiku-5-5` in `.env` to make them even cheaper.

## Differences from the cookbook notebook

- Library tools are marked `defer_loading: true`, so Claude really starts with only `tool_search`, as the
  [tool search docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool) describe.
- The default model is `claude-opus-5-5` (configurable).
- Two extra demos: a token-cost comparison (1) and a 208-tool scale test (4).
- A minimum similarity score (`MIN_SCORE = 0.3`), so weak matches aren't loaded into context.

---

From **TechieTalksAI** by [Schogini](http://schogini.com). Want an AI agent for your business? Book a free demo at http://schogini.com.
