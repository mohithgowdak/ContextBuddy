# Demand Validation — pre-April 20

Three 1-hour experiments. Run in parallel with light doc polish.
Goal: decide whether post-launch backlog items (playground, LangGraph
node, GitHub Action, doctor CLI) actually have demand before building.

---

## Experiment 1 — Problem-first social post + CTA (60 min)

**Goal:** collect ~5 qualified DMs from agent / LLM engineers.

**Where:** LinkedIn + X (same post, minor tweak per platform).

**Template (copy, edit, post):**

> Shipping ContextBuddy next week: a zero-dep Python library that cuts
> LLM context by 60–80% while guaranteeing entity survival (IDs, dates,
> amounts never get dropped). I want 5 design partners who’ll run it on
> real agent traces and tell me what breaks.
>
> Reply **“compress”** or DM me — I’ll send early access.

**Signal to collect (write down, don’t vibe-check):**

- replies asking for repo link
- DMs mentioning a real stack (LangGraph, CrewAI, custom agent, FastAPI)
- people asking “does it work with X vector DB / embedder / model?”
- ignored / zero engagement → means the hook is wrong, not the product

**Kill/ship rule:** if ≥ 3 qualified replies → keep playground + LangGraph
node in post-launch backlog. If 0 → kill playground, rewrite the hook.

---

## Experiment 2 — 15 cold DMs to agent engineers (60 min)

**Goal:** 3 yes-I-have-this-pain responses.

**Audience:** LinkedIn 2nd-degree connections at AI-native startups
(agent platforms, devtools, vertical AI). Prioritize people whose bio
mentions “agents”, “RAG”, “LLM pipeline”, or “FastAPI + LLM”.

**Template (3 lines, no fluff):**

> Hey {first_name}, seeing you build {agent-thing} at {company}.
> Quick q: how do you handle context cost today — summarize, trim,
> RAG, or just eat the tokens? I’m open-sourcing a zero-dep compressor
> next week; happy to benchmark it on one of your traces if useful.

**Signal to collect:**

- replies at all (raw response rate)
- yes-send-link vs no-thanks vs “we already solve this with X”
- what tool they named as the current solution (summarize? tiktoken? LangChain?)
- how many asked “does it have a LangGraph node?” vs “does it have an MCP server?”

**Kill/ship rule:** the named-solution column decides the positioning
(“drop-in replacement for manual truncation” vs “compression layer
before your LangChain call”). More than 2 asks for MCP = MCP server
ships Apr 19 no matter what.

---

## Experiment 3 — README A/B hook test (60 min)

**Goal:** find out which hook actually converts to “I’ll try it in 5 min”.

**Variant A — “Entity survival guarantee”:**

> ContextBuddy is a zero-dep Python library that compresses LLM context
> 60–80% **while guaranteeing no critical entity (IDs, dates, money,
> tickets) ever gets dropped.** Built for agent pipelines where losing
> one invoice ID ruins the answer.

**Variant B — “ROI report + benchmarks”:**

> ContextBuddy is a zero-dep Python library that cuts LLM context 60–80%
> and prints the exact dollar savings on every call. Reproducible
> benchmarks, deterministic core, no new dependencies. 3 lines to
> integrate.

**How to run:**

1. Create two gists (or branches) with the different first screens.
2. Send to 10 target users split 5/5 via DM: “which makes you try it
   in 5 minutes — A or B? why?”
3. Log the verbatim reasons, not just the A/B score.

**Signal to collect:**

- which variant wins outright
- the *reason* people give (trust vs curiosity vs job-relevance)
- any phrase that gets repeated → that is the new tagline

**Kill/ship rule:** winning hook becomes the new README H1. Loser copy
goes into the FAQ as “when NOT to use”.

---

## What to do with the data (30 min, after all 3)

- If **MCP** is mentioned ≥ 2× → ship MCP server before Apr 20. (Locked anyway.)
- If **LangGraph node** is mentioned ≥ 2× → promote it from Backlog to Apr 21 post-launch day-1.
- If **playground** is mentioned 0× → leave it in Backlog, do not build pre-launch.
- If **GUI dashboard / cloud hosted** mentioned → politely ignore. (See: Kill list.)

Record everything in `docs/validation-results.md` so you can cite it
when recruiters ask “how did you prioritize?”.
