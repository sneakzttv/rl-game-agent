"use strict";

// The browser renders state, but Python owns transitions, rewards, and learning.
const $ = (id) => document.getElementById(id);
const arrows = ["↑", "→", "↓", "←"];
let config = null;
let game = { state: [0, 0], steps: 0, total: 0, done: false };
let run = null;
let busy = false;
let playing = false;
let timer = null;
let lesson = 0;

function status(message, error = false) {
  // Use textContent for all variable values: no server strings become HTML.
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function updateControls() {
  const unavailable = busy || !config;
  document.querySelectorAll("[data-action]").forEach((button) => {
    button.disabled = unavailable || game.done || playing;
  });
  $("reset").disabled = unavailable;
  $("agent-step").disabled = unavailable || !run || game.done || playing;
  $("autoplay").disabled = unavailable || !run || game.done;
  $("train").disabled = unavailable || playing;
  $("evaluate").disabled = unavailable || !run || playing;
  $("export").disabled = !run || busy;
  $("csv").disabled = !run || busy;
  $("lesson-example-button").disabled = unavailable;
}

function same(a, b) { return a[0] === b[0] && a[1] === b[1]; }

function renderBoard() {
  if (!config) return;
  $("board").replaceChildren();
  for (let row = 0; row < config.size; row += 1) {
    for (let col = 0; col < config.size; col += 1) {
      const state = [row, col];
      const cell = document.createElement("div");
      cell.className = "cell";
      let label = "safe";
      if (config.walls.some((item) => same(item, state))) {
        cell.classList.add("wall"); cell.textContent = "■"; label = "wall";
      } else if (config.pits.some((item) => same(item, state))) {
        cell.classList.add("pit"); cell.textContent = "×"; label = "pit";
      } else if (same(state, config.goal)) {
        cell.classList.add("goal"); cell.textContent = "G"; label = "goal";
      } else if (run) {
        const entry = run.policy.find((item) => same(item.state, state));
        if (entry) cell.textContent = arrows[entry.action];
      }
      if (same(state, game.state)) {
        cell.classList.add("agent"); cell.textContent = "●"; label += ", player";
      }
      cell.title = `[${row}, ${col}] ${label}`;
      cell.setAttribute("aria-hidden", "true");
      const coordinate = document.createElement("small");
      coordinate.textContent = `${row},${col}`;
      cell.append(coordinate);
      $("board").append(cell);
    }
  }
  // A compact alternative to reading all 36 decorative cells on every move.
  $("board").setAttribute("aria-label",
    `6 by 6 GridWorld. Player row ${game.state[0]}, column ${game.state[1]}. ` +
    "Goal row 5, column 5. Pits at 1,4; 3,3; 4,1. " +
    "Walls at 1,1; 1,2; 2,4; 3,1; 4,3. Arrow keys move. " +
    (run ? "Small arrows display the learned greedy policy." : ""));
  $("state").textContent = JSON.stringify(game.state);
  $("steps").textContent = game.steps;
  $("return").textContent = game.total.toFixed(2);
  updateControls();
}

function stopPlayback() {
  playing = false;
  clearTimeout(timer);
  $("autoplay").textContent = "Watch agent";
}

function resetGame(state = config.start) {
  stopPlayback();
  game = { state: [...state], steps: 0, total: 0, done: false };
  $("event").textContent = "New game. Goal +1 · pit −1 · move −0.02 · wall −0.06.";
  $("mode-label").textContent = "HUMAN CONTROL";
  $("q-values").textContent = "Train and take an agent step to see Q(s, a).";
  renderBoard();
}

async function move(action, mode = "human") {
  if (busy || !config || game.done) return;
  busy = true;
  updateControls();
  try {
    const result = await api("/api/step", {
      state: game.state, steps: game.steps, mode, action,
      model_version: mode === "agent" && run ? run.model_version : undefined,
    });
    game.state = result.state;
    game.steps = result.steps;
    game.total += result.reward;
    game.done = result.terminal || result.truncated;
    $("mode-label").textContent = mode === "agent" ? "LEARNED POLICY" : "HUMAN CONTROL";
    $("event").textContent =
      `${config.actions[result.action]} → ${result.event}; reward ${result.reward.toFixed(2)}.` +
      (game.done ? " Episode finished. Reset to play again." : "");
    if (result.q_values) {
      $("q-values").textContent = result.q_values.map((value, index) =>
        `${config.actions[index].padEnd(5)} Q = ${value.toFixed(4)}`).join("\n") +
        `\nChosen: ${config.actions[result.action]} (largest Q; ties use up/right/down/left order).`;
    }
    if (game.done) stopPlayback();
  } catch (error) {
    stopPlayback();
    status(`Move failed: ${error.message}`, true);
  } finally {
    busy = false;
    renderBoard();
  }
}

async function playback() {
  if (!playing) return;
  await move(undefined, "agent");
  // Schedule after the response, never pile up overlapping network requests.
  if (playing) timer = setTimeout(playback, 300);
}

function showEvaluation(data) {
  const evaluation = data.evaluation;
  $("success").textContent = `${(100 * evaluation.success_rate).toFixed(1)}%`;
  $("baseline").textContent = `${(100 * data.baseline.success_rate).toFixed(1)}%`;
  $("mean-return").textContent = evaluation.mean_return.toFixed(3);
  $("mean-steps").textContent = evaluation.mean_steps.toFixed(2);
  $("eval-detail").textContent =
    `Frozen greedy evaluation: ${evaluation.episodes} episodes, seed ${evaluation.seed}, ` +
    `random safe starts. ${evaluation.wins} goals / ${evaluation.pits} pits / ` +
    `${evaluation.timeouts} timeouts. Same start schedule for the random baseline.`;
}

function showHistory() {
  $("history").replaceChildren();
  for (const point of run.history) {
    const row = document.createElement("tr");
    [point.episode, `${(100 * point.success_rate).toFixed(1)}%`,
      point.mean_return.toFixed(3), point.epsilon.toFixed(3)].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    $("history").append(row);
  }
  $("curve").setAttribute("points", run.history.map((point) =>
    `${40 + 450 * point.episode / run.episodes},${130 - 120 * point.success_rate}`).join(" "));
  $("chart").setAttribute("aria-label",
    `Observed training success over ${run.episodes} episodes. Final window: ` +
    `${(100 * run.history.at(-1).success_rate).toFixed(1)} percent. Full data in the table below.`);
}

$("train-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy || playing) return;
  busy = true;
  updateControls();
  status("Training from scratch on the local Python backend…");
  try {
    const result = await api("/api/train", {
      episodes: Number($("episodes").value), seed: Number($("seed").value),
    });
    run = result;
    showHistory();
    showEvaluation(run);
    resetGame();
    status(`Completed ${run.episodes} real training episodes (seed ${run.seed}). ` +
      "The board arrows now show the learned greedy policy.");
  } catch (error) {
    status(`Training failed: ${error.message}. Is the Python server running?`, true);
  } finally {
    busy = false;
    updateControls();
  }
});

$("evaluate").addEventListener("click", async () => {
  if (busy || playing) return;
  busy = true;
  updateControls();
  try {
    const result = await api("/api/evaluate", { episodes: 200, seed: 12345 });
    // Multiple browser tabs share one model; do not mislabel another tab's run.
    if (result.model_version !== run.model_version) {
      throw new Error("Another tab replaced the shared model. Train again in this tab.");
    }
    run.evaluation = result.evaluation;
    run.baseline = result.baseline;
    showEvaluation(run);
    status("Re-evaluated the frozen policy with seed 12345. No learning during evaluation.");
  } catch (error) {
    status(`Evaluation failed: ${error.message}`, true);
  } finally {
    busy = false;
    updateControls();
  }
});

function download(filename, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url; link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("export").addEventListener("click", () => download(
  `gridworld-seed-${run.seed}.json`, JSON.stringify(run, null, 2), "application/json"));
$("csv").addEventListener("click", () => {
  const keys = ["episode", "epsilon", "mean_return", "success_rate", "window_episodes"];
  const text = [keys.join(","), ...run.history.map((point) =>
    keys.map((key) => point[key]).join(","))].join("\r\n");
  download(`gridworld-training-${run.seed}.csv`, text, "text/csv");
});

document.querySelectorAll("[data-action]").forEach((button) => {
  button.addEventListener("click", () => move(Number(button.dataset.action)));
});
$("board").addEventListener("keydown", (event) => {
  const action = ["ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft"].indexOf(event.key);
  if (action !== -1) {
    event.preventDefault();
    if (!playing) move(action);
  }
});
$("reset").addEventListener("click", () => resetGame());
$("agent-step").addEventListener("click", () => move(undefined, "agent"));
$("autoplay").addEventListener("click", () => {
  if (playing) {
    stopPlayback(); updateControls();
  } else {
    playing = true;
    $("autoplay").textContent = "Pause agent";
    updateControls();
    playback();
  }
});

// Chapters are local data so the walkthrough is usable without training first.
const lessons = [
  {
    title: "The loop: observe → act → learn",
    body: "The environment is a 6×6 board. Python returns the next state and reward; " +
      "the browser only displays them. Training runs thousands of episodes without animation. " +
      "Playback asks the backend for one learned action at a time.",
    example: "Browser POST /api/step → Python transition(state, action) → JSON → board",
    state: [0, 0],
  },
  {
    title: "State and action",
    body: "A state is [row, column], counted from zero in the top-left. " +
      "The four actions are up (0), right (1), down (2), left (3). " +
      "The map never changes, so position fully describes the environment for this task. " +
      "Try this example, then move right: a wall leaves you in place.",
    example: "state = [1, 0]; action = right (1)\nnext state = [1, 0]; event = wall; reward = −0.06",
    state: [1, 0],
  },
  {
    title: "Rewards and episode endings",
    body: "The goal gives +1; a pit gives −1. Both end the episode. " +
      "A safe move costs −0.02, encouraging shorter paths. A wall costs −0.06. " +
      "After 60 actions, the rollout times out. Try this state and move right into a pit.",
    example: "[1, 3] + right → [1, 4], reward −1, terminal = true\n" +
      "[5, 4] + right → goal [5, 5], reward +1",
    state: [1, 3],
  },
  {
    title: "The Q-table: expected future return",
    body: "Each safe cell has four Q values, one per action. Values begin at zero. " +
      "Q-learning estimates discounted future reward, not just immediate reward. " +
      "Small arrows show the largest Q after training. Inspect a real decision below the game.",
    example: "Q([5,4], right) starts at 0.\nA goal transition makes the first update 0 + 0.2 × (1 − 0) = 0.2.",
    state: [5, 4],
  },
  {
    title: "The learning update",
    body: "Alpha (0.20) controls the update size. Gamma (0.95) discounts distant rewards. " +
      "After every training action, the observed reward plus the best future value forms " +
      "a target. Terminal states have no future reward. Timeouts truncate rollouts, so " +
      "nonterminal updates still bootstrap.",
    example: "target = reward + 0.95 × max Q(next, a)  [nonterminal]\n" +
      "Q(s,a) ← Q(s,a) + 0.20 × (target − Q(s,a))\n" +
      "Example: Q=0.1, reward=−0.02, next max=0.8 → target=0.74 → new Q=0.228",
    state: [5, 3],
  },
  {
    title: "Exploration versus exploitation",
    body: "During training, epsilon is the probability of choosing a random action. " +
      "It decays from about 0.50 toward 0.033. Otherwise, the agent chooses the highest Q. " +
      "Episodes begin at random safe cells so the whole map is explored. A seed makes " +
      "the same run reproducible. Each Train click starts over, not fine-tunes.",
    example: "if random() < epsilon: choose random action\nelse: choose argmax Q(state, action)\n" +
      "Ties are resolved consistently: up, right, down, left.",
    state: [2, 2],
  },
  {
    title: "Training charts versus evaluation",
    body: "The chart measures success in windows of up to 50 exploratory training episodes. " +
      "Evaluation freezes learning and chooses greedily for 200 episodes from random safe starts. " +
      "A uniform-random agent uses the same start schedule. Mean return sums all rewards; " +
      "mean steps includes failures and timeouts, so fewer steps alone does not mean better.",
    example: "Success rate = goal endings / evaluation episodes\n" +
      "Return = sum of episode rewards (no discount in the displayed metric)\n" +
      "A ten-action goal episode returns 1 − 9 × 0.02 = 0.82.",
    state: [0, 0],
  },
  {
    title: "Controls, components, and honest scope",
    body: "Use arrows or move buttons to play; Reset returns to [0,0]. Train enables " +
      "Agent: one step and Watch agent. Pause stops automatic playback. The JSON export " +
      "contains the policy and metrics; CSV contains training windows. gridworld.py implements " +
      "learning, server.py exposes a bounded local API, app.js renders this interface. " +
      "This is tabular learning on a known map—not Unity, PPO, DQN, PyTorch, or TensorBoard.",
    example: "1. Train 3000 episodes, seed 42\n2. Compare greedy success with random baseline\n" +
      "3. Watch agent, then play yourself\n4. Re-evaluate with another seed and export real data",
    state: [0, 0],
  },
];

function renderLesson() {
  const current = lessons[lesson];
  $("lesson-count").textContent = `${lesson + 1} / ${lessons.length}`;
  $("lesson-title").textContent = current.title;
  $("lesson-body").textContent = current.body;
  $("lesson-example").textContent = current.example;
  $("lesson-back").disabled = lesson === 0;
  $("lesson-next").disabled = lesson === lessons.length - 1;
  [...$("lesson-tabs").children].forEach((button, index) => {
    if (index === lesson) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
}
lessons.forEach((entry, index) => {
  const button = document.createElement("button");
  button.textContent = `${index + 1}`;
  button.setAttribute("aria-label", `Chapter ${index + 1}: ${entry.title}`);
  button.addEventListener("click", () => { lesson = index; renderLesson(); });
  $("lesson-tabs").append(button);
});
$("lesson-back").addEventListener("click", () => { lesson -= 1; renderLesson(); });
$("lesson-next").addEventListener("click", () => { lesson += 1; renderLesson(); });
$("lesson-example-button").addEventListener("click", () => {
  if (busy || !config) return;
  resetGame(lessons[lesson].state);
  status(`Tutorial example loaded: state ${JSON.stringify(game.state)}. ` +
    "Use the movement buttons or focus the board and press an arrow key.");
});
renderLesson();
updateControls();
api("/api/config").then((data) => {
  config = data;
  resetGame();
  status("Environment ready. Play manually, explore the tutorial, or train an agent.");
}).catch((error) => status(
  `Cannot connect: ${error.message}. Run python server.py and open its printed loopback URL.`, true));
