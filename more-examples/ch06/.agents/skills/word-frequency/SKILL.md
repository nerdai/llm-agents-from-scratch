---
name: word-frequency
description: Compute the top-10 most frequent words in a public-domain text fetched from the web using a Python script.
---

# Word Frequency

This skill counts word frequencies in a popular public-domain text and reports the top-10 results. The text is fetched from the web at runtime and the computation is performed by a Python script to ensure deterministic, accurate counts.

## Steps

### 1. Read the script

Inspect the bundled script before running it:

```
from_scratch__read_file(path="<skill_dir>/scripts/word_freq.py")
```

Replace `<skill_dir>` with the value of **Skill directory** shown at the bottom of this skill content.

### 2. Execute the script

Run the script using the Python interpreter tool:

```
from_scratch__python_interpreter(path="<skill_dir>/scripts/word_freq.py")
```

### 3. Report the result

Report the markdown table printed by the script exactly as-is.
