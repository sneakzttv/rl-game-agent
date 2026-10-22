# GridWorld - a real reinforcement-learning lab

**[Live demo: sneakzttv.github.io/rl-game-agent](https://sneakzttv.github.io/rl-game-agent/)**

A playable browser game with **actual seeded tabular Q-learning in JavaScript**,
plus an independent Python standard-library implementation and local API.
Train from scratch, inspect action values, watch the learned policy, compare it
with a random baseline, and export observed training data. No API keys, packages,
assets, network services, or pretrained models are required.

> **Honest portfolio scope:** this is an educational standalone implementation,
> not a Unity or deep-RL project. It does not implement PyTorch, ML-Agents, PPO,
> DQN, or TensorBoard. Those original resume technologies are not demonstrated
> by this repository.

## Browser mode and GitHub Pages

The live demo is entirely static: `static/engine.js` computes transitions,
Q updates, frozen-policy evaluation, and the matched random baseline on-device.
`app.js` renders policy arrows, observed training windows, exports, and all eight
interactive tutorial chapters. No backend, CDN, third-party service, secrets,
or API keys are needed. Assets use relative paths for the `/rl-game-agent/` subpath.
Each tab starts untrained and owns an independent in-memory model.

To preview exactly the Pages files locally (Python 3.10+):

```powershell
python -m http.server 8000 --bind 127.0.0.1 --directory static
```

Open **http://127.0.0.1:8000**. Use an HTTP server rather than opening the HTML
as `file://`, because browsers restrict local ES module loading. Training yields
with `setTimeout` every 50 episodes, keeping the UI responsive; progress appears
in the status message. The chart is populated when training completes.

### Deployment

`.github/workflows/pages.yml` runs on pushes to `main` and manual dispatch.
It runs dependency-free Node tests and syntax checks, uploads only `static/`
with `actions/upload-pages-artifact@v3`, then deploys using
`actions/deploy-pages@v4`. The workflow uses `contents: read`, `pages: write`,
`id-token: write`, a `pages` concurrency group, and the `github-pages` environment.
A repository maintainer must select **GitHub Actions** as the Pages publishing
source in repository Settings > Pages if not already configured. Deployment
requires a successful workflow and repository Pages availability; the demo link
is the intended URL, not proof that a deployment has already occurred.
No GitHub settings are changed by this implementation.

## Run the optional local Python server

Requirements: Python **3.10+** and a modern browser (Chrome, Edge, Firefox, Safari).
All Python imports are from the standard library. No `pip install` is needed.

```powershell
cd C:\Users\IEatCodeForBreakfast\Documents\GitHub\resume-ai-projects\rl-game-agent
python server.py
```

Open **http://127.0.0.1:8000**. Keep the terminal running; press **Ctrl+C** to stop.
If `python` is not available on Windows, use `py -3 server.py`.
If the port is occupied, run `python server.py --port 8001` and open the printed URL.
The frontend still uses the JavaScript engine in this mode; the Python API
is separately available for experiments and tests, not called by the UI. The
listener is deliberately fixed to `127.0.0.1`, not public interfaces. Do not reverse-proxy or publicly deploy this development server.

### Five-minute, fully offline demonstration

1. Play manually: focus the board and use arrow keys, or press the movement
   buttons. `●` is you, `G` the goal, `×` a pit, `■` a wall.
2. Open **Learn how it works**. All eight chapters have Back/Next controls,
   direct chapter buttons, worked examples, and a **Try this state example**
   button that places the game at an illustrative state.
3. Train **3000 episodes**, **seed 42**. This computes real updates in the
   browser and replaces this tab's previous in-memory model.
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
click starts over. The browser uses seeded **Mulberry32**; Python uses `random.Random`.
The map, rewards, hyperparameters, epsilon schedule, update rules, and stable
argmax ties match, but random streams and therefore numerical results **differ
between JavaScript and Python**. Seeds reproduce runs within the same engine,
not across engines. JSON exports label the browser engine and hyperparameters.

### Measured browser sample

With 3000 training episodes, seed 42, and 200 evaluation episodes at seed 9001,
the browser engine achieved **200/200 goals (100%)** versus **29/200 (14.5%)**
for random actions. Mean return was **0.9137** versus **-1.1224**; mean steps
were **5.315** versus **13.435**. Re-evaluation at seed 12345 gave **100%** versus
**21%** random success. These are measured JavaScript engine results on this
fixed map, not Python results or a claim of generalization.

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
node --test
node --check static\engine.js
node --check static\app.js
git diff --check
```

JavaScript tests require Node **22+** with no npm dependencies. They cover
transitions, validation, known seeded RNG output, repeatability, partial windows,
event-loop yielding, policy/export data, independent models, and learning on
seeds 7, 42, and 123 using the same acceptance thresholds described below.

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

## Layout and optional Python API

```text
gridworld.py           deterministic environment, learning, evaluation
server.py              loopback HTTP server and bounded JSON endpoints
static/index.html      semantic, keyboard-usable interface
static/engine.js       seeded browser environment, learning, evaluation
static/app.js          browser engine UI, playback, charts, tutorials, downloads
static/style.css       responsive presentation and visible focus states
tests/                 Python unittest and Node node:test suites
package.json           ES module mode and dependency-free npm test command
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

Browser models live only in tab memory and are lost on reload. The separate
Python API model lives in server memory and is lost at restart; its clients
share one model with version checks. Browser and Python API models are never
synchronized. JSON exports document results but cannot be imported to restore
training. Game state is client-supplied for transparent educational replay, not
an authoritative competitive scoreboard. The local development HTTP server
is not production-hardened or multi-user infrastructure. The chart summarizes
training after computation finishes; it is not live streaming.

## License

MIT; see [LICENSE](LICENSE).
