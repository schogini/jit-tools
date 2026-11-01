"""Demo 1 - The problem: every tool definition costs context tokens.

We ask the API to COUNT tokens (no model reply is generated) for four setups:
  A. 8 tools loaded up front
  B. tool_search + the same 8 tools deferred
  C. 208 tools loaded up front (a "real company" catalogue)
  D. tool_search + the same 208 tools deferred
"""

from common import MODEL, TOOL_LIBRARY, TOOL_SEARCH_DEFINITION, deferred, get_client, make_synthetic_tools

client = get_client()
messages = [{"role": "user", "content": "What's the weather like in Tokyo?"}]


def count(tools):
    return client.messages.count_tokens(model=MODEL, messages=messages, tools=tools).input_tokens


big_library = TOOL_LIBRARY + make_synthetic_tools(200)

setups = {
    "A. 8 tools, all loaded": TOOL_LIBRARY,
    "B. tool_search + 8 deferred": [TOOL_SEARCH_DEFINITION] + deferred(TOOL_LIBRARY),
    "C. 208 tools, all loaded": big_library,
    "D. tool_search + 208 deferred": [TOOL_SEARCH_DEFINITION] + deferred(big_library),
}

print(f"Model: {MODEL}\n")
print(f"{'Setup':<34}{'Input tokens':>14}")
print("-" * 48)
results = {}
for label, tools in setups.items():
    results[label] = count(tools)
    print(f"{label:<34}{results[label]:>14,}")

a, b = results["A. 8 tools, all loaded"], results["B. tool_search + 8 deferred"]
c, d = results["C. 208 tools, all loaded"], results["D. tool_search + 208 deferred"]
print("-" * 48)
print(f"Saving with   8 tools: {a - b:>7,} tokens ({(a - b) / a:.0%})")
print(f"Saving with 208 tools: {c - d:>7,} tokens ({(c - d) / c:.0%})")
print("\nThe more tools you have, the more 'just-in-time' loading pays off.")
