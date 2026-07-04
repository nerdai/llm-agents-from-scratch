---
name: meeting-notes-summarizer
description: Summarise a meeting transcript by first asking the human operator for their preferred format and level of detail, then producing the summary accordingly.
---

# Meeting Notes Summarizer

Produce a summary of a meeting transcript provided by the user.

## Steps

### 1. Ask for format preference

Call `from_scratch__human_input` with:
- `prompt`: "What format would you like the summary in?"
- `choices`: `["bullet-points", "prose"]`

Record the response as `format_preference`.

### 2. Ask for detail level

Call `from_scratch__human_input` with:
- `prompt`: "What level of detail?"
- `choices`: `["brief", "detailed"]`

Record the response as `detail_preference`.

### 3. Summarise

Produce a summary of the transcript according to the chosen format and detail level:

- **bullet-points + brief**: 3–5 top-level bullet points covering key decisions and action items only.
- **bullet-points + detailed**: Structured bullet points with sub-bullets covering all discussion topics, decisions, and action items.
- **prose + brief**: Two to three sentences capturing the meeting's purpose, key outcomes, and next steps.
- **prose + detailed**: Multiple paragraphs covering all discussion topics in narrative form, ending with action items and owners.
