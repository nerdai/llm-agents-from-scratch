---
name: word-frequency
description: Compute the top-10 most frequent words in any text passage supplied by the user.
---

# Word Frequency

This skill counts word frequencies in a text passage provided by the user and reports the top-10 results as a markdown table. The computation is performed by a Python script to ensure deterministic, accurate counts.

## Steps

### 1. Read the script

Inspect the bundled script before running it:

```
from_scratch__read_file(path="<skill_dir>/scripts/word_freq.py")
```

Replace `<skill_dir>` with the value of **Skill directory** shown at the bottom of this skill content.

### 2. Execute the script

Run the script and pass the user's text passage as the `stdin` argument. The script reads from stdin, so the `stdin` argument **must** contain the full text provided by the user:

```
from_scratch__python_interpreter(path="<skill_dir>/scripts/word_freq.py", stdin="<user_text>")
```

Replace `<user_text>` with the **complete, verbatim** text passage supplied by the user in their request.

### 3. Report the result

Report the markdown table printed by the script exactly as-is.
