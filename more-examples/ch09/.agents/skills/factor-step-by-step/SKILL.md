---
name: factor-step-by-step
description: Fully factor a number into primes by repeatedly calling the factor_step tool, one prime factor at a time, until the remainder is 1.
---

# Factor Step by Step

Fully factor a starting number into its prime factors using the
`factor_step` tool.

## Arguments

The user must provide a **starting number** (an integer greater than 1).

If no starting number is given, ask the user before proceeding.

## Steps

### 1. Initialize

Set the current remainder `n` to the starting number provided.

Begin tracking the factors as a list: `[]`.

### 2. Call the tool

Call `factor_step` with the current value of `n`.

```
factor_step(n=<current_remainder>)
```

**STOP and WAIT** for the tool result before continuing.

### 3. Record the result

Append the returned `factor` to the factors list.

Set `n` to the returned `remaining`.

### 4. Check termination

- If `n == 1`, proceed to Step 5.
- Otherwise, go back to Step 2.

**Important rules:**
- NEVER fabricate or simulate tool call results.
- NEVER make multiple tool calls in a single response.
- ALWAYS wait for the actual tool response before deciding next steps.

### 5. Report the result

Print the complete list of prime factors, for example:

```
1024 = 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2 × 2
```

Also report:
- **Starting number**
- **Total steps taken** (number of tool calls made)
