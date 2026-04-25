"""Read text from stdin and print the top-10 most frequent words.

Output is a markdown table.
"""

import re
import sys
from collections import Counter

text = sys.stdin.read().lower()
words = re.findall(r"[a-z']+", text)
top10 = Counter(words).most_common(10)

print("**Top-10 word frequencies**\n")
print("| Rank | Word | Count |")
print("|------|------|-------|")
for rank, (word, count) in enumerate(top10, 1):
    print(f"| {rank} | {word} | {count} |")
