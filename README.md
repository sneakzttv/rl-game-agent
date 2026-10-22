# GridWorld — a real, local reinforcement-learning lab

A playable browser game backed by **actual tabular Q-learning in Python**.
Train from scratch, inspect action values, watch the learned policy, compare it
with a random baseline, and export observed training data. No API keys, packages,
assets, network services, or pretrained models are required.

> **Honest portfolio scope:** this is an educational standalone implementation,
> not a Unity or deep-RL project. It does not implement PyTorch, ML-Agents, PPO,
> DQN, or TensorBoard. Those original resume technologies are not demonstrated
> by this repository.

## Run locally

Requirements: Python **3.10+** and a modern browser (Chrome, Edge, Firefox, Safari).
All Python imports are from the standard library. No `pip install` is needed.

```powershell
cd C:\Users\IEatCodeForBreakfast\Documents\GitHub\resume-ai-projects\rl-game-agent
python server.py
```

Open **http://127.0.0.1:8000**. Keep the terminal running; press **Ctrl+C** to stop.
If `python` is not available on Windows, use `py -3 server.py`.
If the port is occupied, run `python server.py --port 8001` and open the printed URL.
Run the server rather than opening `static\index.html` directly: the game needs
the Python API. The listener is deliberately fixed to `127.0.0.1`, not public
interfaces. Do not reverse-proxy or publicly deploy this development server.

### Five-minute, fully offline demonstration

1. Play manually: focus the board and use arrow keys, or press the movement
   buttons. `●` is you, `G` the goal, `×` a pit, `■` a wall.
2. Open **Learn how it works**. All eight chapters have Back/Next controls,
   direct chapter buttons, worked examples, and a **Try this state example**
   button that places the game at an illustrative state.
3. Train **3000 episodes**, **seed 42**. This computes real updates on the
   backend and replaces the previous in-memory model.
4. Compare frozen greedy success against random success. Training success is
   charted separately, with an accessible data table.
5. Select **Agent: one step**, inspect the Q values, or **Watch agent**.
   Small arrows show the learned policy. Reset to replay manually.
6. Re-evaluate with seed **12345**, then download run JSON and training CSV.
   Downloads are actual run data, not canned samples.

Nothing is sent off-device. The initial page is intentionally untrained; no
fabricated success metric is shown. The seed-42 run above is the reproducible
offline sample, not a checked-in trained model.

## Environment and algorithm

State is the zero-based `[row, column]` on a fixed 6×6 map. Actions are
`0=up`, `1=right`, `2=down`, `3=left`. Transitions are deterministic:

| Event | Reward | Ending |
| --- | ---: | --- |
| Enter goal `[5,5]` | +1.00 | Terminal |
| Enter a pit | −1.00 | Terminal |
| Safe move | −0.02 | Continue |
| Wall or boundary | −0.06 | Stay in place |
| 60 actions | Last action's reward | Truncated rollout |

Training and evaluation start uniformly from every safe nonterminal cell,
not only `[0,0]`. Manual play normally starts at `[0,0]`. Position describes
the underlying stationary MDP; the 60-action limit is a rollout cutoff, not
a modeled time-dependent state. For that reason, Q updates bootstrap on
truncation but never on terminal goal/pit transitions.

```text
target = reward + gamma * max_a Q(next_state, a)  # nonterminal
target = reward                                  # terminal
Q(state, action) += alpha * (target - Q(state, action))
```

`alpha=0.20`, `gamma=0.95`, and epsilon-greedy exploration
`epsilon = 0.03 + 0.47 * exp(-episode / max(episodes/5, 1))`.
Epsilon is the probability of a random training action. Greedy ties always use
the first action in up/right/down/left order. Tables begin at zero; each Train
click starts over. Python's seeded PRNG makes same-seed runs reproducible.

### Metrics mean what they say

- **Training curve:** observed success in nonoverlapping windows of up to 50
  exploratory training episodes, plus the last partial window.
- **Evaluation:** frozen greedy policy, 200 episodes, seed 9001 by default;
  no Q-table updates. The alternate evaluation uses seed 12345.
- **Random baseline:** uniform random actions and the identical start schedule.
  Separate action/start RNGs preserve matched starts despite different episode
  lengths. This is not an optimal-policy baseline.
- **Success:** goal endings divided by episodes. Pits and timeouts are failures.
- **Mean return:** undiscounted sum of actual rewards, averaged over episodes.
  Ten actions ending at the goal yield `1 - 9*0.02 = 0.82`.
- **Mean steps:** includes failures and timeouts. A quick pit failure is short,
  so do not interpret step count in isolation.

The displayed evaluation uses new random draws on the **same map and same
state distribution**. It is not a held-out-map generalization experiment,
statistical significance claim, or production benchmark.

## Test

From this directory:

```powershell
python -m unittest discover -s tests -v
```

Tests start an actual ephemeral loopback HTTP server and clean it up. They cover
rewards, boundaries, reachability, request validation, same-origin/Host guards,
training locks, static routes, training/evaluation/agent API flow, reproducibility,
partial metric windows, and evaluation without learning.

The learning regression trains **3000 episodes on seeds 7, 42, 123** and evaluates
400 episodes with seed 2026. Each must achieve **at least 95% success** and beat
the uniform-random baseline by **at least 50 percentage points**, with better
mean return. These are executable acceptance criteria, not invented run results.
The BFS helper is used only as a reachability test oracle, never by the agent.
GitHub Actions runs the suite on Python 3.10, 3.12, and 3.13 when this directory
is the repository root.

## Layout and API

```text
gridworld.py           deterministic environment, learning, evaluation
server.py              loopback HTTP server and bounded JSON endpoints
static/index.html      semantic, keyboard-usable interface
static/app.js          API client, playback, charts, tutorials, downloads
static/style.css       responsive presentation and visible focus states
tests/                 stdlib unittest environment and HTTP tests
.github/workflows/     CI configuration
```

| Endpoint | Purpose / JSON request |
| --- | --- |
| `GET /api/config` | Map, actions, start, limits |
| `POST /api/train` | `{"episodes":3000,"seed":42}`; policy, windows, evaluation |
| `POST /api/evaluate` | `{"episodes":200,"seed":12345}`; learned and random metrics |
| `POST /api/step` | `{"state":[0,0],"steps":0,"mode":"human","action":1}` |
| `POST /api/step` | `{"state":[0,0],"steps":0,"mode":"agent","model_version":1}` |

POST requires `Content-Type: application/json`. Invalid input returns descriptive
4xx JSON errors. Bodies are capped at 4096 bytes, training at 10000 episodes,
evaluation at 1000 episodes, and concurrent training is rejected. Static routes
are allowlisted; source files cannot be downloaded. There is no arbitrary
execution, CORS allowance, or external URL loading.

## Resume claim mapping and limitations

| Original claim / expectation | Demonstrated here |
| --- | --- |
| Python reinforcement-learning game agent | **Yes:** working tabular Q-learning |
| Playable demo | **Yes:** manual/learned browser control |
| Training visualization and evaluation | **Yes:** SVG chart, table, seeded random baseline |
| TensorBoard | **No:** native chart/JSON/CSV instead; no TensorBoard event files |
| Unity / ML-Agents | **No:** no Unity scene, executable, bridge, or integration |
| PyTorch / PPO / DQN | **No:** no neural network or those algorithms |

A future Unity port would need an actual tested Unity project, observations,
actions, rewards, training configuration, and reproducible evaluation. It is not
present, so no untested “integration” instructions are supplied.

The model lives in server memory and is lost at restart. JSON exports document
results but cannot be imported to restore training. One shared model serves all
tabs; a model-version check prevents one tab silently playing another tab's
policy. Game state is client-supplied for transparent educational replay, not
an authoritative competitive scoreboard. The local development HTTP server
is not production-hardened or multi-user infrastructure. The chart summarizes
training after the request finishes; it is not live streaming.

## License

MIT; see [LICENSE](LICENSE).
