from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib import error, request

from cs103_scienceworld.constants import TASKS


DEFAULT_FINAL_PROJECT_SIMPLIFICATIONS = ""
DEFAULT_FINAL_PROJECT_TELEMETRY_URL = "http://127.0.0.1:8765/final-project/telemetry"
EASY_SIMPLIFICATIONS = (
    "teleportAction",
    "openDoors",
    "selfWateringFlowerPots",
    "noElectricalAction",
)


@dataclass
class FinalProjectEpisodeStep:
    index: int
    action: str
    observation: str
    reward: int
    score: int
    completed: bool
    moves: int
    auto_resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FinalProjectEpisodeResult:
    task_name: str
    variation_idx: int
    final_score: int
    total_reward: int
    turn_count: int
    completed: bool
    steps: List[FinalProjectEpisodeStep] = field(default_factory=list)
    telemetry_posted: bool = False
    telemetry_error: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        payload["trajectory"] = payload["steps"]
        return payload


@dataclass
class FinalProjectTaskSummary:
    task_name: str
    selected_variations: List[int]
    total_score: int
    max_score: int
    average_score: float
    completed_episodes: int
    total_episodes: int
    total_turns: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FinalProjectEvaluationReport:
    student_id: str
    variation_sample_count: int
    simplifications: str
    telemetry_url: str
    total_score: int
    max_score: int
    average_score: float
    total_reward: int
    completed_episodes: int
    total_episodes: int
    total_turns: int
    task_summaries: List[FinalProjectTaskSummary] = field(default_factory=list)
    episodes: List[FinalProjectEpisodeResult] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_id": self.student_id,
            "variation_sample_count": self.variation_sample_count,
            "simplifications": self.simplifications,
            "telemetry_url": self.telemetry_url,
            "total_score": self.total_score,
            "max_score": self.max_score,
            "average_score": self.average_score,
            "total_reward": self.total_reward,
            "completed_episodes": self.completed_episodes,
            "total_episodes": self.total_episodes,
            "total_turns": self.total_turns,
            "task_summaries": [summary.to_dict() for summary in self.task_summaries],
            "episodes": [episode.to_dict() for episode in self.episodes],
            "created_at": self.created_at,
        }

    def format_summary(self) -> str:
        task_bits = []
        for summary in self.task_summaries:
            task_bits.append(
                f"{summary.task_name}: {summary.total_score}/{summary.max_score}"
                f" across variations {summary.selected_variations}"
            )
        task_line = "\n".join(task_bits) if task_bits else "(no tasks evaluated)"
        return (
            f"Student {self.student_id} final project score: "
            f"{self.total_score}/{self.max_score} "
            f"(avg {self.average_score:.2f}, completed {self.completed_episodes}/{self.total_episodes}, "
            f"turns {self.total_turns})\n"
            f"{task_line}"
        )


def get_final_project_unseen_task_names() -> List[str]:
    return [
        task["task_name"]
        for task in TASKS
        if task.get("topic") == "CS103_Final_Project"
        and not task.get("visible_in_task_list", True)
    ]


def select_variation_subset(variation_indices: Sequence[int], sample_count: int) -> List[int]:
    normalized = sorted({int(variation_idx) for variation_idx in variation_indices})
    if sample_count <= 0:
        raise ValueError("sample_count must be at least 1.")
    if len(normalized) <= sample_count:
        return normalized
    if sample_count == 1:
        return [normalized[len(normalized) // 2]]

    selected_positions = {
        round(step_idx * (len(normalized) - 1) / (sample_count - 1))
        for step_idx in range(sample_count)
    }
    selected = [normalized[position] for position in sorted(selected_positions)]

    if len(selected) < sample_count:
        selected_set = set(selected)
        for variation_idx in normalized:
            if variation_idx in selected_set:
                continue
            selected.append(variation_idx)
            selected_set.add(variation_idx)
            if len(selected) == sample_count:
                break

    return sorted(selected[:sample_count])


def build_episode_telemetry_payload(
    student_id: str,
    episode: FinalProjectEpisodeResult,
) -> Dict[str, Any]:
    return {
        "student_id": student_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_name": episode.task_name,
        "variation_idx": episode.variation_idx,
        "final_score": episode.final_score,
        "total_reward": episode.total_reward,
        "turn_count": episode.turn_count,
        "completed": episode.completed,
        "error": episode.error,
        "trajectory": [step.to_dict() for step in episode.steps],
    }


def normalize_simplifications(simplifications: str) -> str:
    parts = [part.strip() for part in simplifications.split(",") if part.strip()]
    expanded: List[str] = []

    for part in parts:
        if part == "easy":
            for easy_part in EASY_SIMPLIFICATIONS:
                if easy_part not in expanded:
                    expanded.append(easy_part)
            continue

        if part not in expanded:
            expanded.append(part)

    return ",".join(expanded)


def post_episode_telemetry(
    endpoint_url: str,
    student_id: str,
    episode: FinalProjectEpisodeResult,
    timeout_seconds: float = 3.0,
) -> Tuple[bool, str]:
    payload = build_episode_telemetry_payload(student_id, episode)
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        endpoint_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            if 200 <= response.status < 300:
                return True, ""
            return False, f"HTTP {response.status}"
    except error.URLError as exc:
        return False, str(exc)


def prepare_langgraph_controller(state_graph: Any) -> Any:
    if hasattr(state_graph, "invoke"):
        return state_graph
    if hasattr(state_graph, "compile"):
        compiled_graph = state_graph.compile()
        if hasattr(compiled_graph, "invoke"):
            return compiled_graph
    raise TypeError(
        "state_graph must be a LangGraph StateGraph or compiled graph with an invoke() method."
    )


def choose_graph_action(
    controller: Any,
    env: Any,
    student_id: str,
    task_name: str,
    variation_idx: int,
    observation: str,
    info: Dict[str, Any],
    valid_actions: Sequence[str],
    corpus: Sequence[str],
    step_index: int,
    graph_state: Optional[Dict[str, Any]],
    trajectory: Sequence[FinalProjectEpisodeStep],
) -> Tuple[str, Dict[str, Any]]:
    input_state: Dict[str, Any] = {}
    if graph_state:
        input_state.update(graph_state)

    input_state.update(
        {
            "student_id": student_id,
            "task_name": task_name,
            "variation_idx": variation_idx,
            "step_index": step_index,
            "observation": observation,
            "info": info,
            "valid_actions": list(valid_actions),
            "task_description": info.get("taskDesc", ""),
            "score": int(info.get("score", 0)),
            "reward": int(info.get("reward", 0)),
            "turn_count": int(info.get("moves", 0)),
            "corpus": list(corpus),
            "trajectory": [step.to_dict() for step in trajectory],
            "env": env,
        }
    )
    input_state.pop("action", None)

    output_state = controller.invoke(input_state)
    if not isinstance(output_state, Mapping):
        raise TypeError("LangGraph controller must return a state mapping that includes 'action'.")

    merged_state = dict(input_state)
    merged_state.update(output_state)
    action = merged_state.get("action")
    if not isinstance(action, str) or not action.strip():
        raise ValueError("LangGraph controller must return a non-empty string in state['action'].")
    if action not in valid_actions:
        raise ValueError(f"Graph returned invalid action: {action!r}")

    return action, merged_state


def _safe_copy_state(initial_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not initial_state:
        return {}
    try:
        return copy.deepcopy(initial_state)
    except Exception:
        return dict(initial_state)


def evaluate_final_project_state_graph(
    env: Any,
    state_graph: Any,
    student_id: str,
    variation_sample_count: int = 3,
    simplifications: str = DEFAULT_FINAL_PROJECT_SIMPLIFICATIONS,
    telemetry_url: str = DEFAULT_FINAL_PROJECT_TELEMETRY_URL,
    initial_graph_state: Optional[Dict[str, Any]] = None,
    unseen_task_names: Optional[Sequence[str]] = None,
    auto_resolve_ambiguity: bool = True,
    telemetry_timeout_seconds: float = 3.0,
) -> FinalProjectEvaluationReport:
    controller = prepare_langgraph_controller(state_graph)
    selected_task_names = list(unseen_task_names or get_final_project_unseen_task_names())
    if not selected_task_names:
        raise ValueError("No Final Project unseen tasks were found.")
    simplifications = normalize_simplifications(simplifications)

    corpus = list(env.get_corpus())
    episodes: List[FinalProjectEpisodeResult] = []
    task_summaries: List[FinalProjectTaskSummary] = []

    for task_name in selected_task_names:
        env.load(task_name, 0, simplifications)
        candidate_variations = list(env.get_variations_test())
        if not candidate_variations:
            candidate_variations = list(range(env.get_max_variations(task_name)))
        selected_variations = select_variation_subset(candidate_variations, variation_sample_count)

        task_episodes: List[FinalProjectEpisodeResult] = []
        for variation_idx in selected_variations:
            graph_state = _safe_copy_state(initial_graph_state)
            env.load(task_name, variation_idx, simplifications)

            try:
                observation, info = env.reset()
                trajectory: List[FinalProjectEpisodeStep] = []
                total_reward = 0
                completed = False
                final_score = int(info.get("score", 0))

                for _ in range(env.envStepLimit):
                    valid_actions = env.get_valid_action_object_combinations()
                    action, graph_state = choose_graph_action(
                        controller=controller,
                        env=env,
                        student_id=student_id,
                        task_name=task_name,
                        variation_idx=variation_idx,
                        observation=observation,
                        info=info,
                        valid_actions=valid_actions,
                        corpus=corpus,
                        step_index=len(trajectory),
                        graph_state=graph_state,
                        trajectory=trajectory,
                    )

                    observation, reward, completed, info = env.step(action)
                    final_score = int(info["score"])
                    total_reward += int(reward)
                    trajectory.append(
                        FinalProjectEpisodeStep(
                            index=len(trajectory),
                            action=action,
                            observation=observation,
                            reward=int(reward),
                            score=final_score,
                            completed=bool(completed),
                            moves=int(info["moves"]),
                            auto_resolved=False,
                        )
                    )

                    if auto_resolve_ambiguity:
                        while observation.startswith("Ambiguous request:"):
                            observation, reward, completed, info = env.step("0")
                            final_score = int(info["score"])
                            total_reward += int(reward)
                            trajectory.append(
                                FinalProjectEpisodeStep(
                                    index=len(trajectory),
                                    action="0",
                                    observation=observation,
                                    reward=int(reward),
                                    score=final_score,
                                    completed=bool(completed),
                                    moves=int(info["moves"]),
                                    auto_resolved=True,
                                )
                            )

                    if completed:
                        break

                episode_result = FinalProjectEpisodeResult(
                    task_name=task_name,
                    variation_idx=variation_idx,
                    final_score=final_score,
                    total_reward=total_reward,
                    turn_count=len(trajectory),
                    completed=bool(completed),
                    steps=trajectory,
                )
            except Exception as exc:
                episode_result = FinalProjectEpisodeResult(
                    task_name=task_name,
                    variation_idx=variation_idx,
                    final_score=0,
                    total_reward=0,
                    turn_count=0,
                    completed=False,
                    steps=[],
                    error=f"{exc.__class__.__name__}: {exc}",
                )

            telemetry_ok, telemetry_error = post_episode_telemetry(
                endpoint_url=telemetry_url,
                student_id=student_id,
                episode=episode_result,
                timeout_seconds=telemetry_timeout_seconds,
            )
            episode_result.telemetry_posted = telemetry_ok
            episode_result.telemetry_error = telemetry_error

            task_episodes.append(episode_result)
            episodes.append(episode_result)

        task_total_score = sum(episode.final_score for episode in task_episodes)
        task_completed = sum(1 for episode in task_episodes if episode.completed)
        task_turns = sum(episode.turn_count for episode in task_episodes)
        task_summaries.append(
            FinalProjectTaskSummary(
                task_name=task_name,
                selected_variations=selected_variations,
                total_score=task_total_score,
                max_score=100 * len(task_episodes),
                average_score=(task_total_score / len(task_episodes)) if task_episodes else 0.0,
                completed_episodes=task_completed,
                total_episodes=len(task_episodes),
                total_turns=task_turns,
            )
        )

    total_score = sum(episode.final_score for episode in episodes)
    total_reward = sum(episode.total_reward for episode in episodes)
    completed_episodes = sum(1 for episode in episodes if episode.completed)
    total_turns = sum(episode.turn_count for episode in episodes)

    return FinalProjectEvaluationReport(
        student_id=student_id,
        variation_sample_count=variation_sample_count,
        simplifications=simplifications,
        telemetry_url=telemetry_url,
        total_score=total_score,
        max_score=100 * len(episodes),
        average_score=(total_score / len(episodes)) if episodes else 0.0,
        total_reward=total_reward,
        completed_episodes=completed_episodes,
        total_episodes=len(episodes),
        total_turns=total_turns,
        task_summaries=task_summaries,
        episodes=episodes,
    )
