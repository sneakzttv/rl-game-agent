// Browser equivalent of gridworld.py: coordinates are [row, column].
// No network, pretrained weights, or pathfinding is involved in learning.
export const config = Object.freeze({
  size: 6,
  walls: [[1, 1], [1, 2], [2, 4], [3, 1], [4, 3]],
  pits: [[1, 4], [3, 3], [4, 1]],
  goal: [5, 5], start: [0, 0],
  actions: ["up", "right", "down", "left"], max_steps: 60,
});
const key = (state) => state.join(",");
const walls = new Set(config.walls.map(key));
const pits = new Set(config.pits.map(key));
const deltas = [[-1, 0], [0, 1], [1, 0], [0, -1]];
export const states = [];
for (let row = 0; row < config.size; row += 1) {
  for (let col = 0; col < config.size; col += 1) {
    const state = [row, col];
    if (!walls.has(key(state)) && !pits.has(key(state)) && key(state) !== key(config.goal)) {
      states.push(state);
    }
  }
}
const safe = new Set(states.map(key));
function integer(value, name, low, high) {
  if (!Number.isInteger(value) || value < low || value > high) {
    throw new Error(`${name} must be an integer from ${low} to ${high}.`);
  }
}
function validateState(state) {
  if (!Array.isArray(state) || state.length !== 2 || !state.every(Number.isInteger) || !safe.has(key(state))) {
    throw new Error("State must be a nonterminal, walkable [row, column].");
  }
}

// Mulberry32 uses explicit 32-bit arithmetic, not Math.random. This is seeded
// and reproducible within this engine, but is NOT Python's random.Random.
export function mulberry32(seed) {
  let value = seed >>> 0;
  return () => {
    value = (value + 0x6D2B79F5) >>> 0;
    let mixed = Math.imul(value ^ (value >>> 15), value | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}
const choose = (items, rng) => items[Math.floor(rng() * items.length)];

export function transition(state, action) {
  validateState(state);
  integer(action, "action", 0, 3);
  const [dr, dc] = deltas[action];
  const next = [state[0] + dr, state[1] + dc];
  const [row, col] = next;
  // Blocked actions still consume a step and incur a larger penalty.
  if (row < 0 || row >= config.size || col < 0 || col >= config.size || walls.has(key(next))) {
    return { state: [...state], reward: -0.06, terminal: false, event: "wall" };
  }
  if (pits.has(key(next))) return { state: next, reward: -1, terminal: true, event: "pit" };
  if (key(next) === key(config.goal)) return { state: next, reward: 1, terminal: true, event: "goal" };
  return { state: next, reward: -0.02, terminal: false, event: "move" };
}

export function greedyAction(table, state) {
  const values = table[key(state)];
  // indexOf selects the first maximum: up/right/down/left breaks ties.
  return values.indexOf(Math.max(...values));
}

export function evaluate(table = null, { episodes = 200, seed = 9001 } = {}) {
  integer(episodes, "episodes", 1, 1000);
  integer(seed, "seed", 0, 2147483647);
  // Separate RNG streams preserve identical starts for learned and random agents,
  // regardless of how many action draws the baseline consumes in each rollout.
  const starts = mulberry32(seed);
  const actions = mulberry32(seed + 1);
  let wins = 0, pitsCount = 0, timeouts = 0, returns = 0, lengths = 0;
  for (let episode = 0; episode < episodes; episode += 1) {
    let state = choose(states, starts);
    for (let step = 1; step <= config.max_steps; step += 1) {
      const action = table === null ? Math.floor(actions() * 4) : greedyAction(table, state);
      const result = transition(state, action);
      returns += result.reward;
      lengths += 1;
      state = result.state;
      if (result.terminal) {
        wins += Number(result.event === "goal");
        pitsCount += Number(result.event === "pit");
        break;
      }
      if (step === config.max_steps) timeouts += 1;
    }
  }
  return { episodes, seed, wins, pits: pitsCount, timeouts,
    success_rate: wins / episodes, mean_return: returns / episodes, mean_steps: lengths / episodes };
}

export async function train({ episodes = 3000, seed = 42, alpha = 0.2, gamma = 0.95,
  onProgress = () => {} } = {}) {
  integer(episodes, "episodes", 1, 10000);
  integer(seed, "seed", 0, 2147483647);
  if (!Number.isFinite(alpha) || alpha <= 0 || alpha > 1) throw new Error("alpha must be in (0, 1].");
  if (!Number.isFinite(gamma) || gamma < 0 || gamma >= 1) throw new Error("gamma must be in [0, 1).");
  const rng = mulberry32(seed);
  const table = Object.fromEntries(states.map((state) => [key(state), [0, 0, 0, 0]]));
  const history = [];
  let windowReturn = 0, windowWins = 0, windowEpisodes = 0;
  for (let episode = 1; episode <= episodes; episode += 1) {
    const epsilon = 0.03 + 0.47 * Math.exp(-episode / Math.max(episodes / 5, 1));
    let state = choose(states, rng);
    for (let step = 1; step <= config.max_steps; step += 1) {
      const action = rng() < epsilon ? Math.floor(rng() * 4) : greedyAction(table, state);
      const result = transition(state, action);
      // Only terminal goal/pit transitions zero the bootstrap. The 60-step
      // rollout cutoff is not a terminal state of the underlying MDP.
      const future = result.terminal ? 0 : Math.max(...table[key(result.state)]);
      const values = table[key(state)];
      values[action] += alpha * (result.reward + gamma * future - values[action]);
      windowReturn += result.reward;
      state = result.state;
      if (result.terminal) {
        windowWins += Number(result.event === "goal");
        break;
      }
    }
    windowEpisodes += 1;
    if (episode % 50 === 0 || episode === episodes) {
      history.push({ episode, epsilon, mean_return: windowReturn / windowEpisodes,
        success_rate: windowWins / windowEpisodes, window_episodes: windowEpisodes });
      windowReturn = 0; windowWins = 0; windowEpisodes = 0;
      onProgress(episode, episodes);
      // A macrotask boundary every <=3000 transitions lets input, painting,
      // and assistive technology run even during the longest training request.
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
  return { table, history };
}

// Each browser tab owns its model. The optional Python API is independent.
export class BrowserEngine {
  constructor() { this.table = null; this.version = 0; this.training = false; }
  async train(options) {
    if (this.training) throw new Error("Training is already running.");
    this.training = true;
    try {
      const { table, history } = await train(options);
      this.table = table;
      this.version += 1;
      return { episodes: options?.episodes ?? 3000, seed: options?.seed ?? 42,
        alpha: options?.alpha ?? 0.2, gamma: options?.gamma ?? 0.95,
        engine: "javascript-mulberry32-v1", model_version: this.version, history,
        policy: states.map((state) => ({ state: [...state], action: greedyAction(table, state), values: [...table[key(state)]] })),
        ...this.evaluate() };
    } finally { this.training = false; }
  }
  evaluate(options) {
    if (!this.table) throw new Error("Train a model first.");
    return { evaluation: evaluate(this.table, options), baseline: evaluate(null, options), model_version: this.version };
  }
  step({ state, action, steps = 0, mode = "human" }) {
    validateState(state);
    integer(steps, "steps", 0, config.max_steps - 1);
    if (mode !== "human" && mode !== "agent") throw new Error("mode must be human or agent.");
    if (mode === "agent" && !this.table) throw new Error("Train a model first.");
    const values = mode === "agent" ? [...this.table[key(state)]] : null;
    if (values) action = greedyAction(this.table, state);
    const result = transition(state, action);
    const truncated = !result.terminal && steps + 1 >= config.max_steps;
    return { ...result, steps: steps + 1, truncated, event: truncated ? "timeout" : result.event,
      action, q_values: values };
  }
}
