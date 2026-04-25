"""Fetch a popular public-domain text and print the top-10 most frequent words.

Output is a markdown table.
"""

import re
import urllib.request
from collections import Counter

URL = "https://www.gutenberg.org/files/11/11-0.txt"  # Alice in Wonderland
TITLE = "Alice's Adventures in Wonderland"

with urllib.request.urlopen(URL) as response:  # noqa: S310
    text = response.read().decode("utf-8").lower()

words = re.findall(r"[a-z']+", text)
top10 = Counter(words).most_common(10)

print(f"**Top-10 words in _{TITLE}_**\n")
print("| Rank | Word | Count |")
print("|------|------|-------|")
for rank, (word, count) in enumerate(top10, 1):
    print(f"| {rank} | {word} | {count} |")
