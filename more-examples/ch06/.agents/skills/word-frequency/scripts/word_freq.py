"""Print the top-10 most frequent words in the bundled passage.

Output is a markdown table.
"""

import re
from collections import Counter
from pathlib import Path

passage_path = Path(__file__).parent.parent / "assets" / "passage.txt"
text = passage_path.read_text(encoding="utf-8").lower()
words = re.findall(r"[a-z']+", text)
top10 = Counter(words).most_common(10)

print(f"**Top-10 words in `{passage_path.name}`**\n")
print("| Rank | Word | Count |")
print("|------|------|-------|")
for rank, (word, count) in enumerate(top10, 1):
    print(f"| {rank} | {word} | {count} |")
