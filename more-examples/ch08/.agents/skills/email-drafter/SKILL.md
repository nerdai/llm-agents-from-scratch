---
name: email-drafter
description: Draft a reply to an email provided by the user, then ask the operator to confirm before sending.
---

# Email Drafter

Draft a professional reply to the email provided by the user, then seek
operator confirmation before the reply is sent.

## Steps

### 1. Draft the reply

Read the email and compose a clear, professional reply that addresses all
points raised.

### 2. Ask for confirmation

Call `from_scratch__human_input` with:
- `prompt`: "Draft ready — send or discard?"
- `choices`: `["send", "discard"]`

Record the response as `decision`.

### 3. Act on the decision

- If `decision` is `send`: report that the email has been sent and include the
  full draft in your final response.
- If `decision` is `discard`: report that the draft has been discarded without
  sending.
