# WP-I: LLM-agent interface and benchmark  (L1-L4)

**Goal.** Let a language-model agent play the same world through text and tool calls, and measure it on the same footing as
every other agent. This turns the environment into a benchmark for LLM agents.

## Read first
`src/emberpath/world.py` (`Observation`, `step`), `reward.py`, `evaluate.py`, `render.py` (`render_ascii`), `docs/REWARD.md`.
Better with WP-J (difficulty tiers), but works on `studio` and `ranch` without it.

## Deliverables
1. `src/emberpath/llm_env.py`*:
   * `describe(world_or_obs) -> str`: an ASCII map with a legend, the rescuer's and dependents' status, nearby fire in words, the
     time step, and narration of the last tick's events ("the door at (5, 3) did not open"). **No leaking of hidden rules:** a jammed
     door must only be discoverable by trying it. A test must assert that the true `Rules` values never appear in the text.
   * A tool schema with two tools: `move(direction: up|down|left|right)` and `wait()`. A parser that tolerates sloppy model
     output, counts invalid actions, and substitutes `wait` for them (with a logged penalty), never crashing the episode.
   * `run_llm_episode(client, ...)` returning the same `EpisodeResult` shape as `evaluate.run_episode`, plus the transcript.
2. A **scripted fake client** (`ScriptedClient`) that replays canned actions, so every test is offline and deterministic (L2).
3. An optional **Anthropic client** behind the `llm` extra (add `llm = ["anthropic>=0.34"]` if not present). It reads
   `ANTHROPIC_API_KEY` from the environment only, and has a hard cap on steps and on tokens per episode (L3).
4. `scripts/eval_llm.py`*: run N episodes per tier / novelty / household, write per-episode JSONL (transcript, actions, reward
   components, outcome) and a summary table: success rate, mean return, efficiency, **invalid-action rate**, steps to first
   rescue (L4).
5. A short "How good are LLM agents at this?" note in `docs/` with whatever the run shows, including failure modes you read in
   the transcripts. If you cannot run a real model, say so; do not fabricate numbers.

## Done when
```bash
python -m pytest -q -k llm         # all offline, using ScriptedClient
python scripts/eval_llm.py --client scripted --episodes 3 --blueprint studio     # runs end to end with no network
```

## Constraints
Tests never touch the network. Never write a key into any file. Cost guard: default to a small episode count and print the cap
before starting. The interface must expose the same information a non-LLM agent gets, nothing more.

## Docs
`README.md` section on the LLM interface; flip L1-L4 in `docs/SPEC.md`.
