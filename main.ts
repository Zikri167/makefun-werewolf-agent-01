// werewolf-agent-01: a pure-code, model-free Werewolf agent (no model calls at all).
//
// Strategy summary:
// - Every decision is recomputed from the env view alone (the public log, private
//   results, role, partner), so nothing depends on decision order or timing.
// - Randomness comes only from MAKEFUN_TRIAL_SEED, hashed with game/day/kind labels.
// - No wall clock, no Math.random, no network beyond MAKEFUN_ENV_URL.
// - Village side: suspicion scores built from votes, speeches, claims and deaths.
//   The seer inspects the top suspect, reveals a found wolf to the table, and the
//   doctor protects the seer claimant (or the wolves' most likely target).
// - Wolf side: night kills are coordinated by copying the partner's vote when the
//   partner voted first, and by picking the most obvious threat (a seer claimant)
//   when voting first, so the partner can follow. If a night kill failed, the same
//   target is named again next night (the doctor may not protect the same seat
//   twice in a row). In debate wolves blend in, push a villager, and never vote
//   their partner; at the vote they join the bandwagon against a villager.

import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

type Seat = number;
type DecisionKind = 'speak' | 'vote' | 'kill' | 'protect' | 'inspect';

interface LogEntry {
  seq: number;
  day: number;
  kind: string;
  seat?: Seat;
  text?: string;
  round?: number;
  target?: Seat | null;
  died?: Seat | null;
  role?: string | null;
  tally?: { seat: Seat; votes: number }[];
  order?: Seat[];
  winner?: string | null;
  [k: string]: unknown;
}

interface PlayerInfo {
  seat: Seat;
  name: string;
  alive: boolean;
  role: string | null;
}

interface View {
  day: number;
  phase: string;
  you: {
    seat: Seat;
    name: string;
    role: string;
    alive: boolean;
    partner: { seat: Seat; name: string; alive: boolean } | null;
  };
  players: PlayerInfo[];
  seerResults: { night: number; seat: Seat; werewolf: boolean }[] | null;
  lastProtect: { night: number; seat: Seat | null } | null;
  wolfVotes: { night: number; seat: Seat; target: Seat | null }[] | null;
  log: LogEntry[];
  outcome: unknown;
  [k: string]: unknown;
}

interface Pending {
  game: number;
  decision: DecisionKind;
  options?: Seat[];
  max_chars?: number;
  view: View;
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

// ---------------------------------------------------------------- seeded RNG

/** mulberry32 over a sha256(seed || label): deterministic, no Math.random. */
function rngFor(seed: string, label: string): () => number {
  const h = createHash('sha256').update(seed, 'utf8').update('\0').update(label, 'utf8').digest();
  let a = h.readUInt32BE(0) >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const pickSeeded = <T,>(rng: () => number, items: readonly T[]): T =>
  items[Math.floor(rng() * items.length)]!;

/** Deterministic argmax with seeded tie-break: returns the best item and its score. */
function argmaxSeeded<T>(rng: () => number, items: readonly T[], score: (t: T) => number): { item: T; score: number } {
  let best: T | null = null;
  let bestScore = -Infinity;
  for (const it of items) {
    const s = score(it) + rng() * 0.001;
    if (s > bestScore) {
      bestScore = s;
      best = it;
    }
  }
  return { item: best!, score: bestScore - 0.0005 };
}

// ---------------------------------------------------------------- text utils

/** Fold a statement the env's way and cut it to max chars (code points). */
export function foldStatement(text: string): string {
  return text
    .replace(/[\t\n\v\f\r\u0085\u2028\u2029]+/g, ' ')
    .replace(/\p{Cc}/gu, '')
    .trim();
}

function cap(text: string, max: number): string {
  const folded = foldStatement(text);
  const pts = [...folded];
  if (pts.length <= max) return folded;
  return pts.slice(0, max).join('').trim();
}

const esc = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// ---------------------------------------------------------------- game state

const living = (v: View): Seat[] => v.players.filter((p) => p.alive).map((p) => p.seat);
const nameOf = (v: View, seat: Seat): string => v.players.find((p) => p.seat === seat)?.name ?? `seat ${seat}`;
const roleOf = (v: View, seat: Seat): string | null => v.players.find((p) => p.seat === seat)?.role ?? null;
const isWolfSide = (v: View): boolean => v.you.role === 'werewolf';
const partnerSeat = (v: View): Seat | null => v.you.partner?.seat ?? null;

function speechesOf(v: View, day?: number): LogEntry[] {
  return v.log.filter((e) => e.kind === 'speech' && typeof e.text === 'string' && (day === undefined || e.day === day));
}
function votesOf(v: View, day?: number): LogEntry[] {
  return v.log.filter((e) => e.kind === 'vote' && typeof e.target === 'number' && (day === undefined || e.day === day));
}
function dawns(v: View): LogEntry[] {
  return v.log.filter((e) => e.kind === 'dawn');
}
function currentRound(v: View): number {
  const today = v.log.filter((e) => (e.kind === 'speech' || e.kind === 'pass') && e.day === v.day).length;
  const n = Math.max(1, living(v).length);
  return Math.floor(today / n) + 1;
}

/** Does `text` mention seat `s` (by name or "seat N")? */
function mentions(v: View, text: string, s: Seat): boolean {
  const low = text.toLowerCase();
  const name = nameOf(v, s).toLowerCase();
  if (new RegExp(`\\b${esc(name)}\\b`).test(low)) return true;
  return new RegExp(`seat\\s+${s}\\b`).test(low);
}

const WOLF_WORDS = /\b(wolf|wolves|werewolf|suspicious|sus|lying|liar|fake|guilty|evil|kill|exile|vote (him|her|them|out|for))\b/;
const TRUST_WORDS = /\b(trust|believe|clear|innocent|not a wolf|not the wolf|defend|vouch|agree with|with you)\b/;

interface Analysis {
  /** Suspicion score per living seat (village perspective; higher = more wolf-like). */
  score: Map<Seat, number>;
  /** Seats the seer (me) confirmed as wolves and still alive. */
  knownWolves: Seat[];
  /** Seats the seer (me) confirmed as not wolves and still alive. */
  knownClear: Seat[];
  seerClaimants: Seat[];
  doctorClaimants: Seat[];
  /** Who accused whom today (speech): seat -> count of accusers today. */
  accusedToday: Map<Seat, number>;
  /** Who defended whom today. */
  defendedToday: Map<Seat, number>;
  /** Votes cast so far today: target -> count. */
  votesToday: Map<Seat, number>;
  /** Seats that accused me today. */
  myAccusersToday: Seat[];
  /** Seats that accused me or my partner (wolf perspective threat). */
  threat: Map<Seat, number>;
  /** The night's kill target so far (partner's vote), if any. */
  partnerKillPick: Seat | null;
  /** Last night's kill target (both wolves agreed) and whether anyone died. */
  lastNightTarget: Seat | null;
  lastNightDied: boolean;
  /** Has a seer claim already been made by me (in my own speeches)? */
  iClaimedSeer: boolean;
}

/** Build the full picture of one game from its view. */
function analyze(v: View): Analysis {
  const me = v.you.seat;
  const partner = partnerSeat(v);
  const score = new Map<Seat, number>();
  const threat = new Map<Seat, number>();
  const accusedToday = new Map<Seat, number>();
  const defendedToday = new Map<Seat, number>();
  const votesToday = new Map<Seat, number>();
  for (const s of living(v)) {
    score.set(s, 0);
    threat.set(s, 0);
  }
  const bump = (m: Map<Seat, number>, s: Seat, n: number): void => {
    if (m.has(s)) m.set(s, (m.get(s) ?? 0) + n);
  };

  const knownWolves: Seat[] = [];
  const knownClear: Seat[] = [];
  for (const r of v.seerResults ?? []) {
    if (!living(v).includes(r.seat)) continue;
    if (r.werewolf) knownWolves.push(r.seat);
    else knownClear.push(r.seat);
  }

  const seerClaimants: Seat[] = [];
  const doctorClaimants: Seat[] = [];
  let iClaimedSeer = false;

  // --- speeches: claims, accusations, defences (skip my own: no double counting)
  for (const e of speechesOf(v)) {
    const s = e.seat!;
    const text = (e.text ?? '').toLowerCase();
    if (/\b(i am|i'm|im)\b[^.]{0,20}\bseer\b/.test(text) || /\bseer here\b/.test(text)) {
      if (!seerClaimants.includes(s)) seerClaimants.push(s);
      if (s === me) iClaimedSeer = true;
    }
    if (/\b(i am|i'm|im)\b[^.]{0,20}\bdoctor\b/.test(text)) {
      if (!doctorClaimants.includes(s)) doctorClaimants.push(s);
    }
    if (s === me) continue;
    // Inspection claims: "I inspected X, wolf / not a wolf"
    const insp = /\b(inspected|checked)\b/.test(text);
    for (const t of living(v)) {
      if (t === s) continue;
      if (!mentions(v, text, t)) continue;
      const accuses = WOLF_WORDS.test(text);
      const defends = TRUST_WORDS.test(text) && !accuses;
      if (e.day === v.day) {
        if (accuses) bump(accusedToday, t, 1);
        if (defends) bump(defendedToday, t, 1);
      }
      if (insp && accuses) bump(score, t, 15); // someone claims they checked t and found wolf
      else if (insp && defends) bump(score, t, -10); // claims t checked clear
      else if (accuses) bump(score, t, 2);
      else if (defends) bump(score, t, -1);
      // Threat (wolf view): accusing me or my partner is dangerous.
      if (accuses && (t === me || t === partner)) bump(threat, s, 12);
      if (accuses && roleOf(v, t) === 'werewolf') bump(score, s, -6); // accused a revealed wolf: likely village
      if (defends && roleOf(v, t) === 'werewolf') bump(score, s, 8); // defended a revealed wolf
      if (accuses && roleOf(v, t) !== null && roleOf(v, t) !== 'werewolf') bump(score, s, 4); // accused a revealed villager
      if (defends && roleOf(v, t) !== null && roleOf(v, t) !== 'werewolf') bump(score, s, -3);
    }
    // Whoever the speaker pushed as a wolf and who later died a villager: speaker looks worse.
    // (Covered above via revealed roles.)
  }

  // --- votes (skip my own)
  for (const e of votesOf(v)) {
    const s = e.seat!;
    const t = e.target!;
    if (e.day === v.day) bump(votesToday, t, 1);
    if (s === me) continue;
    const tr = roleOf(v, t);
    if (tr === 'werewolf') bump(score, s, -8); // voted a revealed wolf
    else if (tr !== null) bump(score, s, tr === 'seer' || tr === 'doctor' ? 6 : 4); // voted a revealed villager
    if (t === me) bump(score, s, 2);
    if (t === partner) bump(score, s, -4); // voted my wolf partner: likely village (wolves don't vote partner)
  }
  // The night victim was killed because they threatened the wolves: seats the
  // victim accused (or voted for) before dying are more likely wolves.
  for (const d of dawns(v)) {
    if (d.died === null || d.died === undefined) continue;
    const victim = d.died;
    if (roleOf(v, victim) === 'werewolf') continue; // wolves do not kill wolves
    for (const e of speechesOf(v)) {
      if (e.seat !== victim || e.day >= d.day) continue;
      const text = (e.text ?? '').toLowerCase();
      if (!WOLF_WORDS.test(text)) continue;
      for (const t of living(v)) {
        if (t !== victim && mentions(v, text, t)) bump(score, t, 5);
      }
    }
    for (const e of votesOf(v)) {
      if (e.seat !== victim || e.day >= d.day) continue;
      bump(score, e.target!, 3);
    }
  }
  // Voting with a revealed wolf: on any day a revealed wolf voted target T,
  // the other seats that voted T the same day share the wolf's interest.
  const revealedWolves = v.players.filter((p) => p.role === 'werewolf').map((p) => p.seat);
  for (const e of votesOf(v)) {
    if (!revealedWolves.includes(e.seat!)) continue;
    for (const f of votesOf(v, e.day)) {
      if (f.seat !== e.seat && f.target === e.target && f.seat !== me) bump(score, f.seat!, 3);
    }
  }
  // Abstentions in past votes: a vote entry with target null is not in votesOf; count passes instead.
  for (const e of v.log.filter((x) => x.kind === 'pass')) {
    if (e.seat !== me) bump(score, e.seat!, 0.5); // quiet players are slightly wolf-like
  }

  // --- deaths: a revealed wolf's defenders / accusers
  for (const d of dawns(v)) {
    if (d.died === null || d.died === undefined) continue;
    const deadRole = d.role;
    if (deadRole === 'werewolf') {
      // Who defended the dead wolf in speeches? Already scored via role checks above.
    }
    void deadRole;
  }

  // --- seer knowledge dominates
  for (const w of knownWolves) score.set(w, 1000);
  for (const c of knownClear) score.set(c, -50);
  // A false seer claim: if I am the seer, every other claimant is lying (likely a wolf).
  if (v.you.role === 'seer') {
    for (const c of seerClaimants) if (c !== me) bump(score, c, 60);
  } else if (seerClaimants.length >= 2) {
    for (const c of seerClaimants) bump(score, c, 20); // one of them lies
  }
  // Doctor claimants are usually telling the truth (wolves rarely claim doctor early);
  // a seer claimant that is also... leave as is.

  // --- threat scores (wolf view): who endangers the wolves
  for (const s of living(v)) {
    if (s === me || s === partner) continue;
    if (seerClaimants.includes(s)) bump(threat, s, 100);
    if (doctorClaimants.includes(s)) bump(threat, s, 10);
    if (knownWolves.includes(s)) continue;
    bump(threat, s, (score.get(s) ?? 0) * 0.1);
    const speeches = speechesOf(v).filter((e) => e.seat === s).length;
    bump(threat, s, Math.min(4, speeches)); // loud leaders are targets
  }

  // --- night history (wolf coordination)
  const night = v.day; // current night number == current day number
  const myVotes = v.wolfVotes ?? [];
  const partnerVote = myVotes.find((x) => x.night === night && x.seat === partner);
  const partnerKillPick = partnerVote?.target ?? null;
  const lastNight = myVotes.filter((x) => x.night === night - 1 && x.target !== null);
  const lastTargets = [...new Set(lastNight.map((x) => x.target!))];
  const lastNightTarget = lastTargets.length === 1 ? lastTargets[0]! : null;
  const lastDawn = dawns(v).filter((d) => d.day === night - 1);
  const lastNightDied = lastDawn.length > 0 && lastDawn[lastDawn.length - 1]!.died !== null;

  const myAccusersToday = living(v).filter(
    (s) => s !== me && speechesOf(v, v.day).some((e) => e.seat === s && mentions(v, e.text ?? '', me) && WOLF_WORDS.test((e.text ?? '').toLowerCase())),
  );

  return {
    score, knownWolves, knownClear, seerClaimants, doctorClaimants,
    accusedToday, defendedToday, votesToday, myAccusersToday, threat,
    partnerKillPick, lastNightTarget, lastNightDied, iClaimedSeer,
  };
}

// ---------------------------------------------------------------- decisions

/** Tiny deterministic tie-break jitter per seat and day (no rng consumed). */
function jitterFor(seat: Seat, day: number): number {
  const h = (seat * 2654435761 + day * 40503) >>> 0;
  return (h % 1000) / 1000000;
}

/** My most recent day-vote target that is still alive (for consistency). */
function myLastVoteTarget(v: View): Seat | null {
  const mine = votesOf(v).filter((e) => e.seat === v.you.seat && typeof e.target === 'number');
  if (mine.length === 0) return null;
  const t = mine[mine.length - 1]!.target!;
  return living(v).includes(t) ? t : null;
}

/**
 * The village's target for today, chosen once per game-day (the same seed for
 * speak and vote, so accusations and votes agree): top suspicion, sticking
 * with yesterday's target unless the evidence has clearly moved on.
 */
function villageTarget(v: View, a: Analysis): { target: Seat; score: number } {
  const opts = living(v).filter((s) => s !== v.you.seat);
  const val = (s: Seat): number => (a.score.get(s) ?? 0) + (a.accusedToday.get(s) ?? 0) * 2 + jitterFor(s, v.day);
  let top = opts[0]!;
  for (const s of opts) if (val(s) > val(top)) top = s;
  const last = myLastVoteTarget(v);
  if (last !== null && last !== top) {
    const lastScore = val(last);
    if (lastScore >= val(top) - 4 && !a.knownClear.includes(last)) return { target: last, score: lastScore };
  }
  return { target: top, score: val(top) };
}

/** The wolf's frame target for today, chosen once per game-day like above. */
function wolfTarget(v: View, a: Analysis): Seat {
  const partner = partnerSeat(v);
  const opts = living(v).filter((s) => s !== v.you.seat && s !== partner);
  if (opts.length === 0) return v.you.seat;
  const val = (s: Seat): number => (a.score.get(s) ?? 0) + (a.accusedToday.get(s) ?? 0) * 3 + (a.threat.get(s) ?? 0) * 0.2 + jitterFor(s, v.day);
  let top = opts[0]!;
  for (const s of opts) if (val(s) > val(top)) top = s;
  const last = myLastVoteTarget(v);
  if (last !== null && opts.includes(last) && last !== top) {
    if (val(last) >= val(top) - 4) return last;
  }
  return top;
}

function fallbackFor(p: Pending): Record<string, unknown> {
  return p.decision === 'speak'
    ? { game: p.game, decision: 'speak', pass: true }
    : { game: p.game, decision: p.decision, target: null };
}

function validTarget(p: Pending, t: Seat | null): Seat | null {
  if (t === null) return null;
  return (p.options ?? []).includes(t) ? t : null;
}

/** Best wolf-kill target when I vote first (or alone). */
function wolfKillChoice(v: View, a: Analysis, rng: () => number): Seat | null {
  const opts = (v.you.partner ? living(v).filter((s) => s !== v.you.seat && s !== partnerSeat(v)) : living(v).filter((s) => s !== v.you.seat));
  const killable = opts.filter((s) => roleOf(v, s) !== 'werewolf');
  if (killable.length === 0) return null;
  // Repeat last night's target when the kill failed: the doctor cannot protect
  // the same seat two nights running, so the target is exposed tonight.
  if (a.lastNightTarget !== null && !a.lastNightDied && killable.includes(a.lastNightTarget)) {
    return a.lastNightTarget;
  }
  // Otherwise the biggest threat: seer claimants first, then loud accusers.
  return argmaxSeeded(rng, killable, (s) => (a.threat.get(s) ?? 0) + (a.accusedToday.get(s) ?? 0) * 0.5).item;
}

function decideKill(v: View, a: Analysis, rng: () => number): Seat | null {
  // Partner already named someone this night: agree, or nobody dies.
  if (a.partnerKillPick !== null) return a.partnerKillPick;
  return wolfKillChoice(v, a, rng);
}

function decideInspect(v: View, a: Analysis, rng: () => number, p: Pending): Seat | null {
  const opts = (p.options ?? []).filter((s) => s !== v.you.seat);
  if (opts.length === 0) return null;
  // Inspect the most suspicious uninspected seat: confirming a wolf is the
  // most valuable result; a clear result also protects a villager from exile.
  const done = new Set((v.seerResults ?? []).map((r) => r.seat));
  const fresh = opts.filter((s) => !done.has(s));
  const pool = fresh.length > 0 ? fresh : opts;
  return argmaxSeeded(rng, pool, (s) => (a.score.get(s) ?? 0) + (a.accusedToday.get(s) ?? 0)).item;
}

function decideProtect(v: View, a: Analysis, rng: () => number, p: Pending): Seat | null {
  const opts = p.options ?? [];
  if (opts.length === 0) return null;
  // Protect a seer claimant first (they are the wolves' top target tonight).
  const claimants = a.seerClaimants.filter((s) => opts.includes(s));
  if (claimants.length > 0) {
    return argmaxSeeded(rng, claimants, (s) => (a.threat.get(s) ?? 0)).item;
  }
  // Otherwise the seat the wolves most want dead: loudest accuser / leader.
  // If last night's kill failed, my protect probably saved the target; the
  // wolves may repeat it, but I cannot protect the same seat twice running,
  // so take the next most likely target.
  return argmaxSeeded(rng, opts, (s) => (a.threat.get(s) ?? 0) + (a.defendedToday.get(s) ?? 0)).item;
}

/** Village vote target: top suspicion, bandwagoning onto a credible leader. */
function villageVote(v: View, a: Analysis, rng: () => number): Seat | null {
  const opts = living(v).filter((s) => s !== v.you.seat);
  if (opts.length === 0) return null;
  if (a.knownWolves.length > 0) return a.knownWolves[0]!;
  const top = villageTarget(v, a);
  // Day 1: vote only with some signal (round-1 speeches already gave some);
  // abstaining cedes the exile to noise anyway, and the vote must match the
  // accusation already made, or the seat looks inconsistent.
  if (v.day === 1 && top.score < 2) return null;
  // From day 2 on, always vote the (sticky) top suspect: a split, abstaining
  // village lets the wolves' bloc decide the exile.
  // Avoid splitting the village vote: if another seat already leads the vote
  // and is nearly as suspicious, join it.
  let leader: Seat | null = null;
  let leaderVotes = 0;
  for (const [s, n] of a.votesToday) {
    if (s !== v.you.seat && n > leaderVotes) {
      leader = s;
      leaderVotes = n;
    }
  }
  if (leader !== null && leader !== top.target && leaderVotes >= 2) {
    const leaderScore = a.score.get(leader) ?? 0;
    if (leaderScore >= top.score - 5) return leader;
  }
  return top.target;
}

/** Wolf vote: push a villager, never the partner, join the bandwagon. */
function wolfVote(v: View, a: Analysis, rng: () => number): Seat | null {
  const partner = partnerSeat(v);
  const opts = living(v).filter((s) => s !== v.you.seat && s !== partner);
  if (opts.length === 0) return null;
  // If the partner is leading the vote, vote the top villager to overtake them.
  let leader: Seat | null = null;
  let leaderVotes = 0;
  for (const [s, n] of a.votesToday) {
    if (n > leaderVotes) {
      leader = s;
      leaderVotes = n;
    }
  }
  // Prefer seats the village already suspects (easy exile) and loud accusers.
  const villagerTarget = wolfTarget(v, a);
  if (leader === partner && leaderVotes > 0) return villagerTarget;
  if (leader !== null && leader !== partner && leaderVotes >= 2 && opts.includes(leader)) return leader;
  return villagerTarget;
}

// ---------------------------------------------------------------- speeches

function ref(v: View, seat: Seat): string {
  return `${nameOf(v, seat)} (seat ${seat})`;
}

/** Village speech: accuse the top suspect with a reason, or defend. */
function villageSpeak(v: View, a: Analysis, rng: () => number, max: number): string {
  const me = v.you.seat;
  const round = currentRound(v);
  const top = villageTarget(v, a);
  const t = top.target;

  // Seer reveal: I found a wolf. Say so plainly, with the night I checked.
  if (v.you.role === 'seer' && a.knownWolves.length > 0) {
    const w = a.knownWolves[0]!;
    const night = (v.seerResults ?? []).find((r) => r.seat === w)?.night ?? v.day - 1;
    const extras = a.knownWolves.slice(1).map((s) => ref(v, s)).join(', ');
    const lines = [
      `I am the seer. I inspected ${ref(v, w)} on night ${night} and they are a werewolf. Vote them out today.`,
      `I am the seer, I checked ${ref(v, w)} last night: werewolf. We should exile them now, before they kill again.`,
      `Seer here. ${ref(v, w)} is a wolf, I inspected them on night ${night}. Trust me and vote them out.`,
    ];
    let text = pickSeeded(rng, lines);
    if (extras) text += ` I also found ${extras} is a wolf.`;
    if (round === 2 && a.myAccusersToday.length > 0) text += ` ${ref(v, a.myAccusersToday[0]!)} accusing me only helps the wolves.`;
    return cap(text, max);
  }
  // Defend a confirmed-clear seat that is being pushed.
  if (v.you.role === 'seer' && a.iClaimedSeer && a.knownClear.length > 0) {
    const pushed = a.knownClear.filter((s) => (a.accusedToday.get(s) ?? 0) > 0 || (a.votesToday.get(s) ?? 0) > 0);
    if (pushed.length > 0) {
      return cap(`I already checked ${ref(v, pushed[0]!)}: they are not a wolf. Do not waste this exile on them, look at ${ref(v, t)} instead.`, max);
    }
  }
  // Self-defence when accused.
  if (a.myAccusersToday.length > 0 && round === 2) {
    const acc = a.myAccusersToday[0]!;
    return cap(
      pickSeeded(rng, [
        `I am not a wolf. ${ref(v, acc)} is pushing me with no evidence, that is exactly what a wolf would do. Look at ${ref(v, t)} instead.`,
        `${ref(v, acc)} accusing me is a deflection. I am a villager, and my suspicion is on ${ref(v, t)}.`,
      ]),
      max,
    );
  }
  if (v.day === 1 && round === 1) {
    return cap(
      pickSeeded(rng, [
        `Day one, no deaths yet to read. I am watching how everyone votes today, that tells us more than words.`,
        `No strong read yet. I want to hear everyone before I point a finger.`,
        `First day. Let us talk and vote with reasons, not guesses. I am keeping notes on who pushes whom.`,
      ]),
      max,
    );
  }
  if (top.score < 3 && round === 1) {
    return cap(
      pickSeeded(rng, [
        `Nothing solid yet. ${ref(v, t)} feels a little off to me, but I want to hear more before voting.`,
        `I do not have a firm suspect. ${ref(v, t)} is the one I am least sure about so far.`,
        `Too early to be sure. I am leaning ${ref(v, t)} but I am open to better evidence.`,
      ]),
      max,
    );
  }
  const reasons = [
    `${ref(v, t)} pushed hard against villagers who turned out innocent, and dodged every real question.`,
    `${ref(v, t)} voted with the wolves' interests in mind and never once pointed at a real wolf.`,
    `${ref(v, t)} keeps deflecting whenever a wolf is named. That pattern is hard to ignore.`,
    `${ref(v, t)} defended people who turned out to be wolves. I do not buy the innocent act.`,
  ];
  const reason = pickSeeded(rng, reasons);
  if (round === 1) {
    return cap(`My suspicion is on ${ref(v, t)}. ${reason} I think we should vote them out today.`, max);
  }
  return cap(
    pickSeeded(rng, [
      `I am sticking with ${ref(v, t)}. ${reason}`,
      `Vote ${ref(v, t)} today. ${reason}`,
      `The case on ${ref(v, t)} is the strongest we have. ${reason}`,
    ]),
    max,
  );
}

/** Wolf speech: blend in, push a villager, defend the partner if pushed. */
function wolfSpeak(v: View, a: Analysis, rng: () => number, max: number): string {
  const me = v.you.seat;
  const partner = partnerSeat(v);
  const round = currentRound(v);
  const villagers = living(v).filter((s) => s !== me && s !== partner);
  if (villagers.length === 0) return cap('I have nothing to add.', max);
  // The villager to frame: already suspected by the village, or my kill target.
  // Same target as the vote will name, so the story stays consistent.
  const f = wolfTarget(v, a);
  const partnerPushed =
    partner !== null && ((a.accusedToday.get(partner) ?? 0) > 0 || (a.votesToday.get(partner) ?? 0) > 0 || a.myAccusersToday.includes(partner));
  if (partner !== null && partnerPushed && round === 2) {
    return cap(
      pickSeeded(rng, [
        `I do not buy the case on ${ref(v, partner)}. Their reasoning made sense to me. ${ref(v, f)} looks far more suspicious, they keep dodging questions.`,
        `${ref(v, partner)} has been open and consistent. If we exile anyone today it should be ${ref(v, f)}, who has said almost nothing useful.`,
      ]),
      max,
    );
  }
  if (v.day === 1 && round === 1) {
    return cap(
      pickSeeded(rng, [
        `Day one, let us not rush. I am watching the votes, that is where the truth shows.`,
        `No firm read yet. I want to hear everyone out before pointing fingers.`,
        `First day, I am just listening for now. Hasty votes help the wolves, not us.`,
      ]),
      max,
    );
  }
  const frames = [
    `${ref(v, f)} has been pushing villagers all day and never once named a real wolf. That is wolf behaviour.`,
    `${ref(v, f)} keeps changing their story whenever someone asks a straight question. I do not trust it.`,
    `${ref(v, f)} was very quiet when the wolf was discussed, then loud when a villager was blamed. Suspicious.`,
    `Look at how ${ref(v, f)} voted: always with the crowd, never with a reason of their own.`,
  ];
  if (round === 1) {
    return cap(`I am suspicious of ${ref(v, f)}. ${pickSeeded(rng, frames)}`, max);
  }
  return cap(
    pickSeeded(rng, [
      `I am voting ${ref(v, f)} today. ${pickSeeded(rng, frames)}`,
      `We should exile ${ref(v, f)}. ${pickSeeded(rng, frames)}`,
      `My vote goes to ${ref(v, f)}. ${pickSeeded(rng, frames)}`,
    ]),
    max,
  );
}

function decideSpeak(v: View, a: Analysis, rng: () => number, max: number): string | null {
  // Dead seats never speak (the env never asks them), but guard anyway.
  if (!v.you.alive) return null;
  const text = isWolfSide(v) ? wolfSpeak(v, a, rng, max) : villageSpeak(v, a, rng, max);
  return text === '' ? null : text;
}

// ---------------------------------------------------------------- main loop

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

export async function run(): Promise<void> {
  const env = process.env;
  const envUrl = (env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  if (envUrl === '') throw new Error('MAKEFUN_ENV_URL is needed');
  const seed = env.MAKEFUN_TRIAL_SEED ?? 'werewolf-agent-01';
  // The brief is read (as the rules intend) but the agent plays from env views.
  try {
    const file = env.MAKEFUN_PROMPT_FILE ?? '';
    if (file !== '') readFileSync(file, 'utf8');
  } catch {
    // A missing brief must not stall the trial.
  }

  const post = async (path: string, body: unknown): Promise<{ status: number; body: unknown }> => {
    const res = await fetch(`${envUrl}${path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(await res.text());
    } catch {
      parsed = null;
    }
    return { status: res.status, body: parsed };
  };

  const decide = (p: Pending): Record<string, unknown> => {
    const v = p.view;
    // Speak and vote share one rng stream per game-day, so the seat accused
    // in debate is the seat voted for; night actions keep their own stream.
    const label =
      p.decision === 'speak' || p.decision === 'vote'
        ? `game${p.game}:day${v.day}:dayplay`
        : `game${p.game}:day${v.day}:${p.decision}`;
    const rng = rngFor(seed, label);
    const a = analyze(v);
    try {
      switch (p.decision) {
        case 'speak': {
          const text = decideSpeak(v, a, rng, p.max_chars ?? 500);
          return text === null ? { game: p.game, decision: 'speak', pass: true } : { game: p.game, decision: 'speak', text };
        }
        case 'vote':
          return { game: p.game, decision: 'vote', target: validTarget(p, isWolfSide(v) ? wolfVote(v, a, rng) : villageVote(v, a, rng)) };
        case 'kill':
          return { game: p.game, decision: 'kill', target: validTarget(p, decideKill(v, a, rng)) };
        case 'protect':
          return { game: p.game, decision: 'protect', target: validTarget(p, decideProtect(v, a, rng, p)) };
        case 'inspect':
          return { game: p.game, decision: 'inspect', target: validTarget(p, decideInspect(v, a, rng, p)) };
      }
    } catch {
      return fallbackFor(p);
    }
  };

  const keyOf = (p: Pending): string => `${p.game}:${p.decision}:${p.view.log.length}`;
  const answered = new Set<string>();
  let observeFailures = 0;
  for (;;) {
    let obs: { status: number; body: unknown };
    try {
      obs = await post('/observe', {});
    } catch {
      observeFailures += 1;
      if (observeFailures > 60) throw new Error('the env is unreachable');
      await sleep(1000);
      continue;
    }
    if (obs.status !== 200 || !isRecord(obs.body)) {
      observeFailures += 1;
      if (observeFailures > 60) throw new Error(`POST /observe answered ${obs.status}`);
      await sleep(1000);
      continue;
    }
    observeFailures = 0;
    const done = obs.body.done === true;
    if (done) break;
    const pending = (obs.body.pending as Pending[] | undefined) ?? [];
    // Answer pending decisions one at a time, in the order listed: the replay
    // must see the same requests in the same order.
    for (const p of pending) {
      const key = keyOf(p);
      if (answered.has(key)) continue;
      answered.add(key);
      let act = decide(p);
      let posted = await post('/act', act);
      if (posted.status !== 200) {
        const fb = fallbackFor(p);
        if (JSON.stringify(act) !== JSON.stringify(fb)) {
          posted = await post('/act', fb);
        }
      }
      if (posted.status !== 200 && posted.status !== 409) {
        // Not pending any more (the game moved on) or a transient refusal:
        // drop the key so a fresh /observe can re-list it if it still waits.
        answered.delete(key);
      }
    }
    if (pending.length === 0) {
      // Nothing waits: /observe long-polls, so a bare loop iteration is cheap;
      // still yield briefly to avoid a hot spin if the env answers at once.
      await sleep(50);
    }
  }
  process.stdout.write('werewolf-agent: trial done\n');
}

const isMain = process.argv[1] !== undefined && import.meta.url === new URL(`file://${process.argv[1]}`).href;
if (isMain) {
  try {
    await run();
    process.exitCode = 0;
  } catch (err) {
    process.stderr.write(`werewolf-agent: ${(err as Error).stack ?? String(err)}\n`);
    process.exitCode = 1;
  }
}
