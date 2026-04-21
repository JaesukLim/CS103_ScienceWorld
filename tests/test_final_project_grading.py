import json
import unittest
from unittest.mock import patch

from cs103_scienceworld import CS103ScienceWorldFinalProjectEnv
from cs103_scienceworld.final_project_eval import select_variation_subset


class DummyCompiledGraph:
    def __init__(self):
        self.states = []

    def invoke(self, state):
        self.states.append(state)
        assert state["valid_actions"] == ["finish task", "wait"]
        assert state["corpus"] == ["doc-1", "doc-2"]
        return {
            "action": "finish task",
            "last_task": state["task_name"],
            "last_variation": state["variation_idx"],
        }


class DummyStateGraph:
    def __init__(self):
        self.compiled = DummyCompiledGraph()

    def compile(self):
        return self.compiled


class FakeFinalProjectEnv(CS103ScienceWorldFinalProjectEnv):
    def __init__(self):
        self.envStepLimit = 3
        self.current_task = ""
        self.current_variation = 0
        self.current_moves = 0
        self.load_calls = []
        self._variation_map = {
            "task-a-unseen": [1, 3, 5, 7],
            "task-b-unseen": [2, 4, 6, 8],
        }
        self._score_map = {
            ("task-a-unseen", 1): 10,
            ("task-a-unseen", 7): 20,
            ("task-b-unseen", 2): 30,
            ("task-b-unseen", 8): 40,
        }

    def get_corpus(self):
        return ["doc-1", "doc-2"]

    def load(self, taskName, variationIdx=0, simplificationStr="", generateGoldPath=False):
        del generateGoldPath
        self.current_task = taskName
        self.current_variation = variationIdx
        self.current_moves = 0
        self.load_calls.append((taskName, variationIdx, simplificationStr))

    def get_variations_test(self):
        return list(self._variation_map[self.current_task])

    def get_max_variations(self, task_name):
        return len(self._variation_map[task_name])

    def reset(self):
        self.current_moves = 0
        return "initial observation", {
            "moves": 0,
            "score": 0,
            "reward": 0,
            "look": "room",
            "inv": "",
            "taskDesc": f"solve {self.current_task}",
            "valid": self.get_valid_action_object_combinations(),
            "variationIdx": self.current_variation,
            "taskName": self.current_task,
            "simplificationStr": "easy,openContainers",
        }

    def get_valid_action_object_combinations(self):
        return ["finish task", "wait"]

    def step(self, action):
        self.current_moves += 1
        completed = action == "finish task"
        score = self._score_map.get((self.current_task, self.current_variation), 0) if completed else 0
        reward = score if self.current_moves == 1 else 0
        return (
            f"after {action}",
            reward,
            completed,
            {
                "moves": self.current_moves,
                "score": score,
                "reward": reward,
                "look": "room",
                "inv": "",
                "taskDesc": f"solve {self.current_task}",
                "valid": self.get_valid_action_object_combinations(),
                "variationIdx": self.current_variation,
                "taskName": self.current_task,
                "simplificationStr": "easy,openContainers",
            },
        )


class FinalProjectGradingTests(unittest.TestCase):
    def test_select_variation_subset_is_deterministic(self):
        self.assertEqual(select_variation_subset([0, 1, 2, 3, 4], 3), [0, 2, 4])
        self.assertEqual(select_variation_subset([10, 20, 30, 40], 2), [10, 40])

    def test_final_project_env_exposes_unseen_task_names_without_loading_jvm(self):
        env = object.__new__(CS103ScienceWorldFinalProjectEnv)
        env._gateway = None
        task_names = env.get_unseen_task_names()
        self.assertIn("cook-unseen", task_names)
        self.assertIn("corrosion-unseen", task_names)
        self.assertIn("recipe-pipeline-unseen", task_names)
        self.assertIn("corrode-circuit-unseen", task_names)

    def test_grade_state_graph_samples_unseen_variations_and_posts_telemetry(self):
        telemetry_calls = []

        def fake_post_episode_telemetry(endpoint_url, student_id, episode, timeout_seconds=3.0):
            telemetry_calls.append(
                {
                    "endpoint_url": endpoint_url,
                    "student_id": student_id,
                    "episode": json.loads(json.dumps(episode.to_dict())),
                    "timeout_seconds": timeout_seconds,
                }
            )
            return True, ""

        with patch("cs103_scienceworld.final_project_eval.post_episode_telemetry", fake_post_episode_telemetry):
            env = FakeFinalProjectEnv()
            graph = DummyStateGraph()
            report = env.grade_state_graph(
                state_graph=graph,
                student_id="20261234",
                variation_sample_count=2,
                unseen_task_names=["task-a-unseen", "task-b-unseen"],
                telemetry_url="http://example.test/final-project/telemetry",
                print_summary=False,
            )

            self.assertEqual(report.total_episodes, 4)
            self.assertEqual(report.total_score, 100)
            self.assertEqual(report.max_score, 400)
            self.assertEqual(report.completed_episodes, 4)
            self.assertEqual(
                [summary.selected_variations for summary in report.task_summaries],
                [[1, 7], [2, 8]],
            )
            self.assertTrue(all(episode.turn_count == 1 for episode in report.episodes))
            self.assertTrue(all(episode.telemetry_posted for episode in report.episodes))
            self.assertEqual(len(telemetry_calls), 4)
            self.assertEqual(
                {payload["episode"]["task_name"] for payload in telemetry_calls},
                {"task-a-unseen", "task-b-unseen"},
            )
            self.assertTrue(all(payload["student_id"] == "20261234" for payload in telemetry_calls))
            self.assertTrue(all(len(payload["episode"]["trajectory"]) == 1 for payload in telemetry_calls))
            self.assertEqual(
                {payload["episode"]["final_score"] for payload in telemetry_calls},
                {10, 20, 30, 40},
            )
            self.assertTrue(
                all(payload["episode"]["total_reward"] == payload["episode"]["final_score"] for payload in telemetry_calls)
            )

            task_loads = [call for call in env.load_calls if call[0] in {"task-a-unseen", "task-b-unseen"}]
            self.assertEqual(
                task_loads,
                [
                    ("task-a-unseen", 0, "easy,openContainers"),
                    ("task-a-unseen", 1, "easy,openContainers"),
                    ("task-a-unseen", 7, "easy,openContainers"),
                    ("task-b-unseen", 0, "easy,openContainers"),
                    ("task-b-unseen", 2, "easy,openContainers"),
                    ("task-b-unseen", 8, "easy,openContainers"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
