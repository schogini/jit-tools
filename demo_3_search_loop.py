"""Demo 3 - Claude discovers tools just in time.

Claude starts with ONE visible tool: tool_search. The 8 real tools are sent with
defer_loading=True, so they stay out of Claude's context until our search
returns a `tool_reference` to them.
"""

import json
import sys

from common import MIN_SCORE, MODEL, TOOL_LIBRARY, TOOL_SEARCH_DEFINITION, ToolIndex, deferred, get_client, mock_tool_execution

SYSTEM = (
    "You only see a tool_search tool at first. Use it to discover the tools you need "
    "(weather, time, air quality, stocks, currency, interest, market news), then call them. "
    "Prefer a tool over your own knowledge or mental math whenever one might exist."
)


def run_conversation(client, index, user_message, tool_library, max_turns=8):
    tools = [TOOL_SEARCH_DEFINITION] + deferred(tool_library)  # same array on every request
    messages = [{"role": "user", "content": user_message}]
    print("=" * 78 + f"\nUSER: {user_message}\n" + "=" * 78)

    for turn in range(1, max_turns + 1):
        response = client.messages.create(
            model=MODEL, max_tokens=4096, system=SYSTEM, tools=tools, messages=messages
        )
        print(f"\n--- Turn {turn}  (input tokens: {response.usage.input_tokens:,}) ---")
        messages.append({"role": "assistant", "content": response.content})  # keep every block

        if response.stop_reason != "tool_use":
            if response.stop_reason not in ("end_turn", "stop_sequence"):
                print(f"Stopped early: {response.stop_reason}")
            for block in response.content:
                if block.type == "text":
                    print(f"\nCLAUDE: {block.text}")
            return messages

        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"CLAUDE: {block.text}")
            if block.type != "tool_use":
                continue

            if block.name == "tool_search":
                query, top_k = block.input["query"], int(block.input.get("top_k", 3))
                hits = index.search(query, top_k=top_k)  # no cutoff here, so we can show what gets dropped
                print(f"[search] '{query}'")
                for tool, score in hits:
                    note = "" if score >= MIN_SCORE else f"  x dropped (< {MIN_SCORE})"
                    print(f"         -> {tool['name']} ({score:.3f}){note}")
                kept = [tool for tool, score in hits if score >= MIN_SCORE]
                if kept:
                    # The magic: return references, the API expands them into full definitions
                    content = [{"type": "tool_reference", "tool_name": tool["name"]} for tool in kept]
                else:
                    content = "No matching tools found. Try a different description."
            else:
                content = mock_tool_execution(block.name, block.input)
                print(f"[call]   {block.name}({json.dumps(block.input)})")
                print(f"         <- {content[:110]}{'...' if len(content) > 110 else ''}")

            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})

        messages.append({"role": "user", "content": tool_results})  # all results in ONE message
    print("Reached max turns.")
    return messages


if __name__ == "__main__":
    client = get_client()
    print("Loading embedding model and indexing tools...")
    index = ToolIndex(TOOL_LIBRARY)

    questions = sys.argv[1:] or [
        "What's the weather like in Tokyo?",
        "If I invest $10,000 at 5% annual interest for 10 years with monthly compounding, how much will I have?",
    ]
    for q in questions:
        run_conversation(client, index, q, TOOL_LIBRARY)
        print()
