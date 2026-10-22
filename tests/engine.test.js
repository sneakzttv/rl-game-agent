import test from "node:test";
import assert from "node:assert/strict";
import { BrowserEngine, config, states, transition, greedyAction, train, evaluate, mulberry32 } from "../static/engine.js";

test("environment preserves map, rewards, wall collisions, and terminal transitions", () => {
  assert.equal(states.length, 27);
  assert.deepEqual(config.start, [0, 0]);
  assert.deepEqual(transition([0, 0], 0), { state: [0, 0], reward: -0.06, terminal: false, event: "wall" });
  assert.deepEqual(transition([1, 0], 1), { state: [1, 0], reward: -0.06, terminal: false, event: "wall" });
  assert.deepEqual(transition([0, 0], 1), { state: [0, 1], reward: -0.02, terminal: false, event: "move" });
  assert.deepEqual(transition([1, 3], 1), { state: [1, 4], reward: -1, terminal: true, event: "pit" });
  assert.deepEqual(transition([5, 4], 1), { state: [5, 5], reward: 1, terminal: true, event: "goal" });
  for (const state of states) for (let action = 0; action < 4; action += 1) {
    const result = transition(state, action);
    assert.ok(result.state.every((coordinate) => coordinate >= 0 && coordinate < 6));
    assert.equal(result.terminal, ["pit", "goal"].includes(result.event));
  }
});

test("reject invalid environment and learning inputs", async () => {
  for (const state of [[1, 1], [1, 4], [5, 5], [true, 0], [-1, 0], null]) {
    assert.throws(() => transition(state, 0));
  }
  for (const action of [true, -1, 4, 0.5]) assert.throws(() => transition([0, 0], action));
  for (const options of [{ episodes: 0 }, { episodes: true }, { episodes: 10001 }, { seed: -1 },
    { seed: 2147483648 }, { alpha: NaN }, { alpha: 0 }, { gamma: 1 }]) {
    await assert.rejects(train(options));
  }
  assert.throws(() => evaluate(null, { episodes: 0 }));
});

test("Mulberry32 is reproducible with known output and distinct seeds", () => {
  const first = mulberry32(42), second = mulberry32(42), other = mulberry32(7);
  assert.equal(first(), 0.6011037519201636);
  second();
  for (let draw = 0; draw < 100; draw += 1) assert.equal(first(), second());
  assert.notEqual(mulberry32(42)(), other());
});

test("same seed reproduces weights, observed windows, and frozen evaluation", async () => {
  const first = await train({ episodes: 125, seed: 42 });
  const second = await train({ episodes: 125, seed: 42 });
  assert.deepEqual(first, second);
  assert.notDeepEqual(first.table, (await train({ episodes: 125, seed: 7 })).table);
  assert.deepEqual(first.history.map((point) => point.window_episodes), [50, 50, 25]);
  assert.equal(first.history.at(-1).epsilon, 0.03 + 0.47 * Math.exp(-5));
  const snapshot = JSON.stringify(first.table);
  assert.deepEqual(evaluate(first.table), evaluate(first.table));
  assert.equal(JSON.stringify(first.table), snapshot);
  assert.equal(greedyAction({ "0,0": [0, 0, 0, 0] }, [0, 0]), 0);
});

test("training yields to the event loop and reports all chunks", async () => {
  let yielded = false;
  setTimeout(() => { yielded = true; }, 0);
  const progress = [];
  await train({ episodes: 151, onProgress: (done, total) => progress.push([done, total]) });
  assert.ok(yielded);
  assert.deepEqual(progress, [[50, 151], [100, 151], [150, 151], [151, 151]]);
});

for (const seed of [7, 42, 123]) {
  test(`3000 episodes seed ${seed}: greedy success >=95% and >=50 points above random`, async () => {
    const { table, history } = await train({ episodes: 3000, seed });
    const learned = evaluate(table, { episodes: 400, seed: 2026 });
    const baseline = evaluate(null, { episodes: 400, seed: 2026 });
    assert.ok(learned.success_rate >= 0.95, JSON.stringify(learned));
    assert.ok(learned.success_rate - baseline.success_rate >= 0.5);
    assert.ok(learned.mean_return > baseline.mean_return);
    assert.equal(learned.wins + learned.pits + learned.timeouts, 400);
    assert.equal(history.length, 60);
  });
}

test("browser run supplies policy/exports, timeout semantics and isolated tab models", async () => {
  const engine = new BrowserEngine();
  assert.throws(() => engine.evaluate());
  assert.throws(() => engine.step({ state: [0, 0], mode: "agent" }));
  assert.equal(engine.step({ state: [0, 0], action: 0, steps: 59 }).event, "timeout");
  const goal = engine.step({ state: [5, 4], action: 1, steps: 59 });
  assert.ok(goal.terminal);
  assert.equal(goal.truncated, false);
  const pending = engine.train({ episodes: 3000, seed: 42 });
  await assert.rejects(engine.train({ episodes: 1 }));
  const run = await pending;
  assert.equal(run.policy.length, states.length);
  assert.equal(run.engine, "javascript-mulberry32-v1");
  assert.deepEqual(JSON.parse(JSON.stringify(run)), run);
  const decision = engine.step({ state: [0, 0], mode: "agent" });
  assert.equal(decision.q_values.length, 4);
  assert.equal(decision.action, run.policy[0].action);
  const snapshot = JSON.stringify(run);
  engine.evaluate({ episodes: 200, seed: 12345 });
  assert.equal(JSON.stringify(run), snapshot);
  assert.throws(() => new BrowserEngine().evaluate());
  const replacement = await engine.train({ episodes: 1, seed: 42 });
  assert.equal(replacement.model_version, 2);
  assert.equal(replacement.history.length, 1);
});
