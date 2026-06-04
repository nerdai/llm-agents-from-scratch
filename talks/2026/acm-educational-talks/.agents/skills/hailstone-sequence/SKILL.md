---
name: hailstone-sequence
description: Compute the complete Hailstone sequence for a given starting number by repeatedly calling the next_number tool until the value 1 is reached.
---

# Hailstone Sequence

Compute the full Hailstone (Collatz) sequence from a starting number down to 1
using the `next_number` tool.

## Arguments

The user must provide a **starting number** (a positive integer greater than 1).

If no starting number is given, ask the user before proceeding.

## Steps

### 1. Initialize

Set the current number `x` to the starting number provided by the user.

Begin tracking the sequence as a list: `[x]`.

### 2. Call the tool

Call `next_number` with the current value of `x`.

```
next_number(x=<current_number>)
```

**STOP and WAIT** for the tool result before continuing.

### 3. Record the result

Append the returned value to the sequence list.

Set `x` to the returned value.

### 4. Check termination

- If `x == 1`, proceed to Step 5.
- Otherwise, go back to Step 2.

**Important rules:**
- NEVER fabricate or simulate tool call results.
- NEVER make multiple tool calls in a single response.
- ALWAYS wait for the actual tool response before deciding next steps.

### 5. Report the result

Print the complete sequence from start to finish, for example:

```
6 → 3 → 10 → 5 → 16 → 8 → 4 → 2 → 1
```

Also report:
- **Starting number**
- **Total steps taken** (number of tool calls made)
- **Maximum value reached** in the sequence
