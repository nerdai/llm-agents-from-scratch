---
name: create-issue
description: Create a GitHub issue in nerdai/llm-agents-from-scratch and add it to project #11. Use when asked to create a ticket, issue, or task. Supports labels and issue kinds like book-diagrams.
disable-model-invocation: true
---

# Create GitHub Issue

## Repo and project

- **Repo:** `nerdai/llm-agents-from-scratch`
- **Project:** `#11` (owner: `nerdai`)

## Workflow

0. **Before doing anything**, confirm you have enough information. If any of
   the following are unclear, ask the user before proceeding:
   - What is the issue title or topic?
   - What kind of issue is it? (standard, book-diagram, or other)
   - Which chapter is it scoped to, if any?
   - Any specific labels to apply?

   Do not create the issue until you have a clear title and kind.

1. Identify issue details from the user's request:
   - **Title** — concise, action-oriented
   - **Body** — see templates below based on kind
   - **Labels** — from user request, or infer from kind
   - **Chapter label** — add `Chapter N` label if the issue is scoped to a chapter

2. Create the issue:
   ```bash
   gh issue create \
     --repo nerdai/llm-agents-from-scratch \
     --title "<title>" \
     --body "<body>" \
     --label "<label>"
   ```

3. Add to project #11:
   ```bash
   gh project item-add 11 --owner nerdai --url <issue-url>
   ```

4. If the `Chapter N` label doesn't exist yet, create it first:
   ```bash
   gh label create "Chapter N" \
     --repo nerdai/llm-agents-from-scratch \
     --color 0075ca \
     --description "Chapter N — <title>"
   ```

5. Confirm with the issue URL.

---

## Issue kinds

### Standard issue

```markdown
## Overview

<1-2 sentence summary of what needs to be done>

## Acceptance Criteria

- [ ] <criterion>
- [ ] <criterion>

## Related

<chapter, dependency, or context>
```

### Book diagram

Title format: `Diagram: <description>`
Label: `diagram` (create if it doesn't exist)

```markdown
## Overview

<What this diagram communicates and where it appears in the book>

## Chapter

Chapter N — <chapter title>

## Diagram type

<UML class diagram / sequence diagram / flowchart / table / other>

## Content to show

- <element>
- <element>

## Notes

<Any layout hints, style notes, or references>
```

---

## Label reference

| Label | When to use |
|---|---|
| `Chapter N` | Issue scoped to a specific chapter |
| `diagram` | Book diagram to create |
| `enhancement` | New feature or capability |
| `bug` | Bug fix |
| `documentation` | Docs, notebooks, or book content |
