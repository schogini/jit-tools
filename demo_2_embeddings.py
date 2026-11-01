"""Demo 2 - Semantic search over tools with embeddings (runs 100% locally, no API key).

Steps:
  1. Turn each tool definition into plain text
  2. Embed every tool into a 384-number vector
  3. Embed a query and rank tools by cosine similarity
"""

from common import MIN_SCORE, TOOL_LIBRARY, ToolIndex, tool_to_text

print("Step 1 - a tool, flattened to text:\n")
print(tool_to_text(TOOL_LIBRARY[0]))

print("\nStep 2 - embedding all tools (first run downloads a ~90 MB model)...")
index = ToolIndex(TOOL_LIBRARY)
print(f"Embeddings shape: {index.vectors.shape}  -> {index.vectors.shape[0]} tools x {index.vectors.shape[1]} dims")

print("\nStep 3 - search by meaning, not keywords:")
queries = [
    "I need to check the weather",
    "stock market data",
    "how much will my savings grow",  # no word overlap with 'compound interest'
    "is it safe to go jogging outside",  # should lean to air quality / weather
    "turn dollars into yen",
]
for q in queries:
    print(f"\n  Query: '{q}'")
    for rank, (tool, score) in enumerate(index.search(q, top_k=3), 1):
        bar = "#" * int(max(score, 0) * 40)
        note = "" if score >= MIN_SCORE else "  (below cutoff)"
        print(f"   {rank}. {tool['name']:<30} {score:.3f} {bar}{note}")

print(f"\nMatches below MIN_SCORE={MIN_SCORE} are dropped before Claude sees them (Demos 3 and 4).")
