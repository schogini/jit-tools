"""Shared pieces for every demo: the tool library, mock tool runner,
embedding helpers and the Claude client.

Adapted from the Claude Cookbook recipe "Tool search with embeddings".
"""

import json
import os
import random
from datetime import datetime, timedelta
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY (and optional CLAUDE_MODEL) from .env

# Keep demo output clean: silence tokenizer/transformers chatter
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, local, 384 dims
MIN_SCORE = 0.3  # drop search hits less similar than this (weak matches just waste context)


# ---------------------------------------------------------------------------
# 1. The tool library: 8 tools across two domains (Weather and Finance)
# ---------------------------------------------------------------------------
TOOL_LIBRARY: list[dict[str, Any]] = [
    # Weather tools
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature"},
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_forecast",
        "description": "Get the weather forecast for multiple days ahead",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state"},
                "days": {"type": "number", "description": "Number of days to forecast (1-10)"},
            },
            "required": ["location", "days"],
        },
    },
    {
        "name": "get_timezone",
        "description": "Get the current timezone and time for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name or timezone identifier"},
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_air_quality",
        "description": "Get current air quality index and pollutant levels for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name or coordinates"},
            },
            "required": ["location"],
        },
    },
    # Finance tools
    {
        "name": "get_stock_price",
        "description": "Get the current stock price and market data for a given ticker symbol",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL, GOOGL)"},
                "include_history": {"type": "boolean", "description": "Include historical data"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another using current exchange rates",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to convert"},
                "from_currency": {"type": "string", "description": "Source currency code (e.g., USD)"},
                "to_currency": {"type": "string", "description": "Target currency code (e.g., EUR)"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "calculate_compound_interest",
        "description": "Calculate compound interest for investments over time",
        "input_schema": {
            "type": "object",
            "properties": {
                "principal": {"type": "number", "description": "Initial investment amount"},
                "rate": {"type": "number", "description": "Annual interest rate (as percentage)"},
                "years": {"type": "number", "description": "Number of years"},
                "frequency": {
                    "type": "string",
                    "enum": ["daily", "monthly", "quarterly", "annually"],
                    "description": "Compounding frequency",
                },
            },
            "required": ["principal", "rate", "years"],
        },
    },
    {
        "name": "get_market_news",
        "description": "Get recent financial news and market updates for a specific company or sector",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Company name, ticker symbol, or sector"},
                "limit": {"type": "number", "description": "Maximum number of news articles to return"},
            },
            "required": ["query"],
        },
    },
]


def make_synthetic_tools(n: int) -> list[dict[str, Any]]:
    """Generate n plausible-looking extra tools, to simulate a big company catalogue."""
    services = ["crm", "billing", "github", "slack", "jira", "hr", "inventory", "analytics", "email", "calendar"]
    actions = ["create", "update", "delete", "list", "get", "search", "archive", "export", "assign", "sync"]
    objects = ["record", "ticket", "invoice", "user", "report", "message", "event", "item", "project", "file"]
    tools = []
    for i in range(n):
        svc, act, obj = services[i % 10], actions[(i // 10) % 10], objects[(i // 100) % 10]
        tools.append(
            {
                "name": f"{svc}_{act}_{obj}_{i}",
                "description": f"{act.capitalize()} a {obj} in the {svc} system. "
                f"Use this when the user wants to {act} {obj}s stored in {svc}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": f"The {obj} identifier"},
                        "payload": {"type": "object", "description": f"Fields for the {obj}"},
                        "dry_run": {"type": "boolean", "description": "Validate without applying"},
                    },
                    "required": ["id"],
                },
            }
        )
    return tools


# ---------------------------------------------------------------------------
# 2. The single meta-tool Claude starts with
# ---------------------------------------------------------------------------
TOOL_SEARCH_DEFINITION = {
    "name": "tool_search",
    "description": (
        "Search for available tools that can help with a task. Returns tool definitions "
        "for matching tools. Use this when you need a tool but don't have it available yet."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language description of the kind of tool you need "
                "(e.g., 'weather information', 'currency conversion', 'stock prices')",
            },
            "top_k": {"type": "number", "description": "Number of tools to return (default: 5)"},
        },
        "required": ["query"],
    },
}


def deferred(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return copies of the tools marked defer_loading=True (sent, but hidden until referenced)."""
    return [{**tool, "defer_loading": True} for tool in tools]


# ---------------------------------------------------------------------------
# 3. Embeddings: turn each tool into text, then into a vector
# ---------------------------------------------------------------------------
def tool_to_text(tool: dict[str, Any]) -> str:
    """Flatten a tool definition into readable text (name + description + params)."""
    parts = [f"Tool: {tool['name']}", f"Description: {tool['description']}"]
    params = tool.get("input_schema", {}).get("properties", {})
    if params:
        parts.append(
            "Parameters: "
            + ", ".join(f"{name} ({info.get('type', '')}): {info.get('description', '')}" for name, info in params.items())
        )
    return "\n".join(parts)


class ToolIndex:
    """A tiny in-memory vector index over a tool library."""

    def __init__(self, tools: list[dict[str, Any]]):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.tools = tools
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        # normalize_embeddings=True makes dot product == cosine similarity
        self.vectors = self.model.encode(
            [tool_to_text(t) for t in tools], convert_to_numpy=True, normalize_embeddings=True
        )

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[tuple[dict[str, Any], float]]:
        q = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        scores = self.vectors @ q
        best = self.np.argsort(scores)[::-1][:top_k]
        return [(self.tools[i], float(scores[i])) for i in best if scores[i] >= min_score]


# ---------------------------------------------------------------------------
# 4. Mock tool execution (a real app would call real APIs here)
# ---------------------------------------------------------------------------
def mock_tool_execution(tool_name: str, tool_input: dict[str, Any]) -> str:
    if tool_name == "get_weather":
        unit = tool_input.get("unit", "fahrenheit")
        temp = random.randint(15, 30) if unit == "celsius" else random.randint(60, 85)
        return json.dumps(
            {
                "location": tool_input.get("location"),
                "temperature": temp,
                "unit": unit,
                "conditions": random.choice(["sunny", "partly cloudy", "cloudy", "rainy"]),
                "humidity": random.randint(40, 80),
            }
        )
    if tool_name == "get_forecast":
        days = int(tool_input.get("days", 3))
        forecast = [
            {
                "date": (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
                "high": random.randint(20, 30),
                "low": random.randint(10, 20),
                "conditions": random.choice(["sunny", "cloudy", "rainy"]),
            }
            for i in range(days)
        ]
        return json.dumps({"location": tool_input.get("location"), "forecast": forecast})
    if tool_name == "get_timezone":
        return json.dumps({"location": tool_input.get("location"), "timezone": "UTC+9", "utc_offset": "+09:00"})
    if tool_name == "get_air_quality":
        aqi = random.randint(20, 150)
        return json.dumps({"location": tool_input.get("location"), "aqi": aqi, "category": "Good" if aqi <= 50 else "Moderate"})
    if tool_name == "get_stock_price":
        return json.dumps(
            {"ticker": tool_input.get("ticker"), "price": round(random.uniform(100, 500), 2), "change_percent": round(random.uniform(-2, 2), 2)}
        )
    if tool_name == "convert_currency":
        rate = round(random.uniform(0.8, 1.2), 4) if tool_input.get("to_currency") != "JPY" else round(random.uniform(140, 155), 2)
        amount = tool_input.get("amount", 0)
        return json.dumps({**tool_input, "exchange_rate": rate, "converted_amount": round(amount * rate, 2)})
    if tool_name == "calculate_compound_interest":
        p, r, y = tool_input.get("principal", 0), tool_input.get("rate", 0), tool_input.get("years", 0)
        n = {"daily": 365, "monthly": 12, "quarterly": 4, "annually": 1}.get(tool_input.get("frequency", "monthly"), 12)
        final = p * (1 + r / 100 / n) ** (n * y)
        return json.dumps({"final_amount": round(final, 2), "interest_earned": round(final - p, 2)})
    if tool_name == "get_market_news":
        q = tool_input.get("query", "")
        return json.dumps({"query": q, "articles": [{"title": f"{q} - headline {i + 1}", "source": "Reuters"} for i in range(3)]})
    return json.dumps({"status": "executed", "tool": tool_name, "input": tool_input})


def get_client():
    import anthropic

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is missing. Copy .env.example to .env and add your key.")
    return anthropic.Anthropic()
