"""Demo 4 - Same pattern, bigger haystack, harder question.

The catalogue now holds 208 tools (8 real + 200 synthetic). The question needs
tools from TWO domains, so Claude must search more than once.
"""

from common import TOOL_LIBRARY, ToolIndex, get_client, make_synthetic_tools
from demo_3_search_loop import run_conversation

client = get_client()
library = TOOL_LIBRARY + make_synthetic_tools(200)
print(f"Indexing {len(library)} tools...")
index = ToolIndex(library)

question = (
    "I'm flying to Tokyo next week. What's the 3-day forecast there, what is the "
    "air quality like, and how many yen will I get for 500 USD?"
)
messages = run_conversation(client, index, question, library)

discovered = sorted(
    {
        part["tool_name"]
        for m in messages
        if m["role"] == "user" and isinstance(m["content"], list)
        for r in m["content"]
        if isinstance(r.get("content"), list)
        for part in r["content"]
    }
)
print(f"\nTools ever loaded into context: {len(discovered)} of {len(library)}")
print("  " + ", ".join(discovered))
