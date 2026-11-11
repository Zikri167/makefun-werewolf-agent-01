# Werewolf agent — make.fun `werewolf-agent-01`

A pure-code, model-free Werewolf agent (Node 22, standard library only).
It makes **no model calls at all**: every decision is computed from the env
view, so it plays even with no API key.

## How it plays

- **Deterministic.** Randomness comes only from `MAKEFUN_TRIAL_SEED`, hashed
  with game/day/decision labels. No wall clock, no `Math.random`, no network
  beyond `MAKEFUN_ENV_URL`. The program loops `POST /observe` and answers
  each pending decision in order with `POST /act`.
- **Village side — suspicion scoring.** Each living seat is scored from the
  public log: votes for/against revealed roles, speech accusations and
  defences, role claims, who the night victim accused before dying, and who
  voted with a revealed wolf. The seer inspects the top suspect, reveals a
  found wolf to the table (with the night inspected), and repeats the reveal.
  The doctor protects a seer claimant, else the wolves' most likely target.
  Votes and accusations use one sticky target per game-day, so the case told
  in debate is the vote cast.
- **Wolf side — coordination.** At night the agent copies its partner's vote
  when the partner voted first; when voting first it names the most obvious
  threat (a seer claimant, then loud accusers), so the partner can follow.
  If a night kill failed, it names the same target again — the doctor may
  not protect the same seat two nights running. By day it frames one
  villager consistently and never votes its partner.

## Files

- `main.ts` — the whole agent (run as `node /bot/main.ts`)
- `LICENSE` — MIT

## Local testing

Tested end-to-end against the kit's real env (`program/env/server.mjs`)
with a scripted stub house lane on all 3 practice tables: 30/30 games
finish, wolf 15/15 every trial, vote accuracy ~60% when voting. The stub
house players vote by raw accusation counts, so village results there
understate play against the real reasoning house players.
