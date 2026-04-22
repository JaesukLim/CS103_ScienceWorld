import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cs103_scienceworld import (
    CS103ScienceWorldFinalProjectEnv,
    evaluate_final_project_tasks,
    grade_final_project_unseen_tasks,
)
from cs103_scienceworld.final_project_eval import select_variation_subset


class DummyLLM:
    def __init__(self, name="dummy-llm"):
        self.name = name


class EpisodeCompiledGraph:
    def __init__(self):
        self.states = []

    def invoke(self, state):
        self.states.append(state)
        assert isinstance(state["llm"], DummyLLM)
        assert state["valid_actions"] == ["finish task", "wait"]
        assert state["corpus"] == ["doc-1", "doc-2"]

        env = state["env"]
        trajectory = list(state.get("trajectory", []))
        observation = state["observation"]
        info = dict(state["info"])
        valid_actions = list(state["valid_actions"])
        total_reward = int(state.get("total_reward", 0))
        final_score = int(state.get("final_score", info.get("score", 0)))
        completed = bool(state.get("completed", False))

        while len(trajectory) < env.envStepLimit and not completed:
            action = valid_actions[0]
            observation, reward, completed, info = env.step(action)
            final_score = int(info["score"])
            total_reward += int(reward)
            trajectory.append(
                {
                    "index": len(trajectory),
                    "action": action,
                    "observation": observation,
                    "reward": int(reward),
                    "score": final_score,
                    "completed": bool(completed),
                    "moves": int(info["moves"]),
                    "auto_resolved": False,
                }
            )
            valid_actions = env.get_valid_action_object_combinations()

        return {
            "observation": observation,
            "info": info,
            "trajectory": trajectory,
            "turn_count": len(trajectory),
            "total_reward": total_reward,
            "final_score": final_score,
            "completed": completed,
        }


class BadCompiledGraph:
    def invoke(self, state):
        del state
        return {"completed": True}


class DummyStateGraph:
    def __init__(self, compiled=None):
        self.compiled = compiled or EpisodeCompiledGraph()

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
            "task-a-tiny": [0, 1],
            "task-b-seen": [2, 4],
            "task-a-unseen": [1, 3, 5, 7],
            "task-b-unseen": [2, 4, 6, 8],
        }
        self._score_map = {
            ("task-a-tiny", 0): 10,
            ("task-a-tiny", 1): 20,
            ("task-b-seen", 2): 30,
            ("task-b-seen", 4): 40,
            ("task-a-unseen", 1): 10,
            ("task-a-unseen", 7): 20,
            ("task-b-unseen", 2): 30,
            ("task-b-unseen", 8): 40,
        }

    def get_corpus(self):
        return ["doc-1", "doc-2"]

    def get_unseen_task_names(self):
        return ["task-a-unseen", "task-b-unseen"]

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

    def test_evaluate_final_project_tasks_uses_single_graph_invoke_per_episode(self):
        env = FakeFinalProjectEnv()
        llm = DummyLLM()
        graph = DummyStateGraph()

        report = evaluate_final_project_tasks(
            llm=llm,
            state_graph=graph,
            env=env,
            student_id="20261234",
            task_names=["task-a-tiny", "task-b-seen"],
            variation_sample_count=2,
            print_progress=False,
        )

        self.assertEqual(report.total_episodes, 4)
        self.assertEqual(report.total_score, 100)
        self.assertEqual(report.max_score, 400)
        self.assertEqual(report.completed_episodes, 4)
        self.assertEqual([summary.selected_variations for summary in report.task_summaries], [[0, 1], [2, 4]])
        self.assertEqual(len(graph.compiled.states), 4)
        self.assertTrue(all(state["llm"] is llm for state in graph.compiled.states))
        self.assertTrue(all(state["env"] is env for state in graph.compiled.states))
        self.assertTrue(all(len(episode.steps) == 1 for episode in report.episodes))
        self.assertTrue(all(not episode.telemetry_posted for episode in report.episodes))

    def test_grade_final_project_unseen_tasks_posts_submission_telemetry(self):
        telemetry_calls = []

        def fake_post_submission_telemetry(endpoint_url, report, timeout_seconds=3.0):
            telemetry_calls.append(
                {
                    "endpoint_url": endpoint_url,
                    "report": json.loads(json.dumps(report.to_dict(include_task_name_restore_map=True))),
                    "report_str": str(report),
                    "timeout_seconds": timeout_seconds,
                }
            )
            return True, ""

        with patch("cs103_scienceworld.final_project_eval.post_submission_telemetry", fake_post_submission_telemetry):
            env = FakeFinalProjectEnv()
            llm = DummyLLM()
            graph = DummyStateGraph()
            report = grade_final_project_unseen_tasks(
                llm=llm,
                state_graph=graph,
                env=env,
                student_id="20261234",
                variation_sample_count=2,
                telemetry_url="http://example.test/final-project/telemetry",
                print_progress=False,
            )

        self.assertEqual(report.total_episodes, 4)
        self.assertEqual(len(telemetry_calls), 1)
        self.assertTrue(all(episode.telemetry_posted for episode in report.episodes))
        report_payload = telemetry_calls[0]["report"]
        self.assertEqual(report_payload["student_id"], "20261234")
        self.assertEqual(report_payload["total_score"], 100)
        self.assertEqual(report_payload["max_score"], 400)
        masked_names = {episode["task_name"] for episode in report_payload["episodes"]}
        self.assertEqual(
            {report_payload["task_name_restore_map"][task_name] for task_name in masked_names},
            {"task-a-unseen", "task-b-unseen"},
        )
        self.assertTrue(masked_names.isdisjoint({"task-a-unseen", "task-b-unseen"}))
        self.assertTrue(all(state["env"].envStepLimit == 50 for state in graph.compiled.states))
        self.assertEqual(env.envStepLimit, 3)

    def test_grade_final_project_unseen_tasks_ignores_output_dir_to_avoid_task_leaks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env = FakeFinalProjectEnv()
            llm = DummyLLM()
            graph = DummyStateGraph()

            report = grade_final_project_unseen_tasks(
                llm=llm,
                state_graph=graph,
                env=env,
                student_id="20261234",
                variation_sample_count=1,
                output_dir=Path(temp_dir) / "should_not_exist",
                print_progress=False,
            )

            self.assertEqual(report.total_episodes, 2)
            self.assertFalse((Path(temp_dir) / "should_not_exist").exists())

    def test_invalid_graph_output_is_recorded_as_episode_error(self):
        env = FakeFinalProjectEnv()
        llm = DummyLLM()
        graph = DummyStateGraph(compiled=BadCompiledGraph())

        report = evaluate_final_project_tasks(
            llm=llm,
            state_graph=graph,
            env=env,
            student_id="20261234",
            task_names="task-a-tiny",
            variation_sample_count=1,
        )

        self.assertEqual(report.total_score, 0)
        self.assertEqual(report.total_episodes, 1)
        self.assertIn("trajectory", report.episodes[0].error)

    def test_env_grade_state_graph_wrapper_forwards_to_standalone_unseen_grader(self):
        env = object.__new__(CS103ScienceWorldFinalProjectEnv)
        env._gateway = None
        fake_report = object()

        with patch("cs103_scienceworld.scienceworld.grade_final_project_unseen_tasks", return_value=fake_report) as grader:
            result = env.grade_state_graph(
                state_graph="graph",
                student_id="20261234",
                llm="bound-llm",
                print_summary=False,
            )

        self.assertIs(result, fake_report)
        grader.assert_called_once()
        self.assertEqual(grader.call_args.kwargs["llm"], "bound-llm")
        self.assertEqual(grader.call_args.kwargs["state_graph"], "graph")
        self.assertIs(grader.call_args.kwargs["env"], env)


if __name__ == "__main__":
    unittest.main()
