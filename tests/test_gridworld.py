"""Deterministic learning regression tests; no downloaded models or fixtures."""

import math
import unittest

from gridworld import (
    GOAL, MAX_STEPS, STATES, evaluate, greedy_action,
    shortest_path, train, transition,
)


class EnvironmentTests(unittest.TestCase):
    def test_rewards_and_terminal_states(self):
        self.assertEqual(transition((0, 0), 0), ((0, 0), -0.06, False, "wall"))
        self.assertEqual(transition((1, 0), 1), ((1, 0), -0.06, False, "wall"))
        self.assertEqual(transition((0, 0), 1), ((0, 1), -0.02, False, "move"))
        self.assertEqual(transition((1, 3), 1), ((1, 4), -1.0, True, "pit"))
        self.assertEqual(transition((5, 4), 1), (GOAL, 1.0, True, "goal"))

    def test_every_training_start_can_reach_goal(self):
        # An unreachable start would make the advertised success threshold unfair.
        for state in STATES:
            distance = shortest_path(state)
            self.assertIsNotNone(distance, state)
            self.assertLess(distance, MAX_STEPS, state)

    def test_invalid_transitions_are_rejected(self):
        for state, action in [(GOAL, 1), ((1, 1), 0), ((0, 0), -1), ((0, 0), True)]:
            with self.assertRaises(ValueError):
                transition(state, action)


class LearningTests(unittest.TestCase):
    def test_measurable_learning_across_deterministic_seeds(self):
        # Evaluate on a different RNG seed from training, with exploration off.
        for seed in (7, 42, 123):
            with self.subTest(seed=seed):
                table, history = train(episodes=3000, seed=seed)
                learned = evaluate(table, episodes=400, seed=2026)
                random_baseline = evaluate(episodes=400, seed=2026)
                self.assertGreaterEqual(learned["success_rate"], 0.95)
                self.assertGreaterEqual(
                    learned["success_rate"] - random_baseline["success_rate"], 0.5
                )
                self.assertGreater(learned["mean_return"], random_baseline["mean_return"])
                self.assertEqual(history[-1]["episode"], 3000)
                self.assertTrue(all(math.isfinite(v) for row in table.values() for v in row))
                self.assertTrue(any(v != 0 for row in table.values() for v in row))

    def test_runs_are_reproducible(self):
        self.assertEqual(train(episodes=137, seed=12), train(episodes=137, seed=12))
        self.assertEqual(evaluate(seed=17), evaluate(seed=17))

    def test_history_accounts_for_partial_windows(self):
        _, history = train(episodes=137, seed=42)
        self.assertEqual([point["episode"] for point in history], [50, 100, 137])
        self.assertEqual(sum(point["window_episodes"] for point in history), 137)
        self.assertTrue(all(0 <= point["success_rate"] <= 1 for point in history))

    def test_evaluation_does_not_change_policy(self):
        table, _ = train(episodes=100, seed=42)
        before = {state: values[:] for state, values in table.items()}
        result = evaluate(table, episodes=20, seed=3)
        self.assertEqual(table, before)
        self.assertEqual(result["wins"] + result["pits"] + result["timeouts"], 20)
        self.assertLessEqual(result["mean_steps"], MAX_STEPS)

    def test_ties_and_training_parameter_validation(self):
        self.assertEqual(greedy_action({(0, 0): [0, 0, 0, 0]}, (0, 0)), 0)
        for kwargs in [
            {"episodes": 0}, {"episodes": True}, {"episodes": 10001},
            {"seed": -1}, {"alpha": 0}, {"alpha": float("nan")},
            {"gamma": 1}, {"gamma": float("inf")},
        ]:
            with self.assertRaises(ValueError):
                train(**kwargs)


if __name__ == "__main__":
    unittest.main()
