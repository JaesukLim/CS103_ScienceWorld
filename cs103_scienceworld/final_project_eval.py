from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
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

ADJECTIVES = (
    "Amber",
    "Brisk",
    "Calm",
    "Clever",
    "Dusky",
    "Gentle",
    "Golden",
    "Hidden",
    "Icy",
    "Lucky",
    "Quiet",
    "Swift",
)

NOUNS = (
    "Badger",
    "Comet",
    "Falcon",
    "Forest",
    "Lantern",
    "Meadow",
    "Otter",
    "Pine",
    "River",
    "Sparrow",
    "Tiger",
    "Valley",
)


def make_task_codename(index: int) -> str:
    adjective = ADJECTIVES[index % len(ADJECTIVES)]
    noun = NOUNS[(index // len(ADJECTIVES)) % len(NOUNS)]
    if index < len(ADJECTIVES) * len(NOUNS):
        return f"{adjective} {noun}"
    return f"{adjective} {noun} {index}"


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
    telemetry_url: str = field(repr=False)
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
    _task_name_codename_map: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def anonymize_task_names(self) -> None:
        if self._task_name_codename_map:
            return

        task_names = sorted(
            {episode.task_name for episode in self.episodes}
            | {summary.task_name for summary in self.task_summaries}
        )
        self._task_name_codename_map = {
            task_name: make_task_codename(index)
            for index, task_name in enumerate(task_names)
        }

        for episode in self.episodes:
            episode.task_name = self._task_name_codename_map[episode.task_name]

        for summary in self.task_summaries:
            summary.task_name = self._task_name_codename_map[summary.task_name]

    def get_task_name_restore_map(self) -> Dict[str, str]:
        return {
            codename: task_name
            for task_name, codename in self._task_name_codename_map.items()
        }

    def to_dict(
        self,
        *,
        include_telemetry_url: bool = False,
        include_task_name_restore_map: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "student_id": self.student_id,
            "variation_sample_count": self.variation_sample_count,
            "simplifications": self.simplifications,
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

        if include_telemetry_url:
            payload["telemetry_url"] = self.telemetry_url
        if include_task_name_restore_map:
            payload["task_name_restore_map"] = self.get_task_name_restore_map()

        return payload

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

    def __str__(self) -> str:
        return self.format_summary()

    __repr__ = __str__


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


def build_submission_telemetry_payload(
    report: FinalProjectEvaluationReport,
) -> Dict[str, Any]:
    return report.to_dict(include_task_name_restore_map=True)


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


def post_submission_telemetry(
    endpoint_url: str,
    report: FinalProjectEvaluationReport,
    timeout_seconds: float = 3.0,
) -> Tuple[bool, str]:
    payload = build_submission_telemetry_payload(report)
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


def _safe_copy_state(initial_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not initial_state:
        return {}
    try:
        return copy.deepcopy(initial_state)
    except Exception:
        return dict(initial_state)


def _coerce_task_names(task_names: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(task_names, str):
        normalized = [task_names]
    else:
        normalized = [str(task_name) for task_name in task_names]

    cleaned = [task_name.strip() for task_name in normalized if task_name and task_name.strip()]
    if not cleaned:
        raise ValueError("At least one Final Project task name is required.")
    return cleaned


def _build_initial_episode_graph_state(
    *,
    llm: Any,
    env: Any,
    student_id: str,
    task_name: str,
    variation_idx: int,
    observation: str,
    info: Dict[str, Any],
    valid_actions: Sequence[str],
    corpus: Sequence[str],
    initial_graph_state: Optional[Dict[str, Any]],
    auto_resolve_ambiguity: bool,
) -> Dict[str, Any]:
    input_state = _safe_copy_state(initial_graph_state)
    input_state.update(
        {
            "llm": llm,
            "env": env,
            "student_id": student_id,
            "task_name": task_name,
            "variation_idx": variation_idx,
            "step_index": 0,
            "observation": observation,
            "info": dict(info),
            "valid_actions": list(valid_actions),
            "task_description": info.get("taskDesc", ""),
            "score": int(info.get("score", 0)),
            "reward": int(info.get("reward", 0)),
            "turn_count": int(info.get("moves", 0)),
            "corpus": list(corpus),
            "trajectory": [],
            "total_reward": 0,
            "final_score": int(info.get("score", 0)),
            "completed": False,
            "auto_resolve_ambiguity": auto_resolve_ambiguity,
        }
    )
    input_state.pop("action", None)
    return input_state


def _normalize_episode_step(step: Any, index: int) -> FinalProjectEpisodeStep:
    if isinstance(step, FinalProjectEpisodeStep):
        return FinalProjectEpisodeStep(
            index=index,
            action=step.action,
            observation=step.observation,
            reward=step.reward,
            score=step.score,
            completed=step.completed,
            moves=step.moves,
            auto_resolved=step.auto_resolved,
        )

    if not isinstance(step, Mapping):
        raise TypeError("Graph trajectory entries must be mappings or FinalProjectEpisodeStep objects.")

    action = step.get("action")
    observation = step.get("observation")
    if not isinstance(action, str) or not action.strip():
        raise ValueError("Each graph trajectory step must include a non-empty string 'action'.")
    if not isinstance(observation, str):
        raise ValueError("Each graph trajectory step must include a string 'observation'.")

    return FinalProjectEpisodeStep(
        index=int(step.get("index", index)),
        action=action,
        observation=observation,
        reward=int(step.get("reward", 0)),
        score=int(step.get("score", 0)),
        completed=bool(step.get("completed", False)),
        moves=int(step.get("moves", index + 1)),
        auto_resolved=bool(step.get("auto_resolved", False)),
    )


def _normalize_episode_trajectory(trajectory: Any) -> List[FinalProjectEpisodeStep]:
    if trajectory is None:
        raise ValueError(
            "Graph output must include 'trajectory'. The student graph is responsible for "
            "running env.step() and returning the full episode trajectory."
        )
    if isinstance(trajectory, (str, bytes)) or not isinstance(trajectory, Sequence):
        raise TypeError("Graph output 'trajectory' must be a sequence of step mappings.")

    return [_normalize_episode_step(step, index) for index, step in enumerate(list(trajectory))]


def _episode_result_from_graph_output(
    task_name: str,
    variation_idx: int,
    graph_output: Any,
) -> FinalProjectEpisodeResult:
    if not isinstance(graph_output, Mapping):
        raise TypeError("LangGraph controller must return a state mapping for the completed episode.")

    steps = _normalize_episode_trajectory(graph_output.get("trajectory"))
    error_value = graph_output.get("error", "")
    error_message = "" if error_value is None else str(error_value)
    if not steps and not error_message:
        raise ValueError(
            "Graph output trajectory is empty. The student graph must run env.step() "
            "cyclically and return the completed episode trajectory."
        )

    final_score = int(graph_output.get("final_score", steps[-1].score if steps else 0))
    total_reward = int(graph_output.get("total_reward", sum(step.reward for step in steps)))
    turn_count = int(graph_output.get("turn_count", len(steps)))
    completed = bool(graph_output.get("completed", steps[-1].completed if steps else False))

    return FinalProjectEpisodeResult(
        task_name=task_name,
        variation_idx=variation_idx,
        final_score=final_score,
        total_reward=total_reward,
        turn_count=turn_count,
        completed=completed,
        steps=steps,
        error=error_message,
    )


def _build_grading_plan(
    env: Any,
    task_names: Sequence[str],
    variation_sample_count: int,
    simplifications: str,
) -> List[Tuple[str, List[int]]]:
    grading_plan: List[Tuple[str, List[int]]] = []
    for task_name in task_names:
        env.load(task_name, 0, simplifications)
        candidate_variations = list(env.get_variations_test())
        if not candidate_variations:
            candidate_variations = list(range(env.get_max_variations(task_name)))
        grading_plan.append((task_name, select_variation_subset(candidate_variations, variation_sample_count)))
    return grading_plan


def _sanitize_output_name(value: str) -> str:
    cleaned = [character if character.isalnum() or character in {"-", "_"} else "_" for character in value]
    return "".join(cleaned).strip("_") or "item"


def _write_evaluation_artifacts(
    output_dir: Union[str, Path],
    report: FinalProjectEvaluationReport,
) -> None:
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = base_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    report_path = base_dir / "final_project_report.json"
    report_payload = report.to_dict(include_task_name_restore_map=True)
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for episode in report.episodes:
        episode_name = _sanitize_output_name(episode.task_name)
        episode_path = episodes_dir / f"{episode_name}_variation_{episode.variation_idx}.json"
        episode_path.write_text(
            json.dumps(episode.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _evaluate_final_project_tasks(
    *,
    llm: Any,
    state_graph: Any,
    env: Any,
    student_id: str,
    task_names: Sequence[str],
    variation_sample_count: int,
    simplifications: str,
    telemetry_url: str,
    initial_graph_state: Optional[Dict[str, Any]],
    auto_resolve_ambiguity: bool,
    telemetry_timeout_seconds: float,
    print_progress: bool,
    submit_report: bool,
    output_dir: Optional[Union[str, Path]],
) -> FinalProjectEvaluationReport:
    controller = prepare_langgraph_controller(state_graph)
    if not task_names:
        raise ValueError("No Final Project task names were provided.")

    simplifications = normalize_simplifications(simplifications)
    grading_plan = _build_grading_plan(env, task_names, variation_sample_count, simplifications)
    total_expected_episodes = sum(len(selected_variations) for _, selected_variations in grading_plan)
    completed_episode_count = 0
    corpus = list(env.get_corpus())
    episodes: List[FinalProjectEpisodeResult] = []
    task_summaries: List[FinalProjectTaskSummary] = []

    for task_name, selected_variations in grading_plan:
        task_episodes: List[FinalProjectEpisodeResult] = []
        for variation_idx in selected_variations:
            env.load(task_name, variation_idx, simplifications)

            try:
                observation, info = env.reset()
                graph_input_state = _build_initial_episode_graph_state(
                    llm=llm,
                    env=env,
                    student_id=student_id,
                    task_name=task_name,
                    variation_idx=variation_idx,
                    observation=observation,
                    info=info,
                    valid_actions=env.get_valid_action_object_combinations(),
                    corpus=corpus,
                    initial_graph_state=initial_graph_state,
                    auto_resolve_ambiguity=auto_resolve_ambiguity,
                )
                graph_output = controller.invoke(graph_input_state)
                episode_result = _episode_result_from_graph_output(task_name, variation_idx, graph_output)
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

            task_episodes.append(episode_result)
            episodes.append(episode_result)
            completed_episode_count += 1
            if print_progress:
                print(
                    f"Grading progress: {completed_episode_count}/{total_expected_episodes} episodes completed",
                    flush=True,
                )

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
    report = FinalProjectEvaluationReport(
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

    if submit_report:
        report.anonymize_task_names()
        telemetry_ok, telemetry_error = post_submission_telemetry(
            endpoint_url=telemetry_url,
            report=report,
            timeout_seconds=telemetry_timeout_seconds,
        )
        for episode in report.episodes:
            episode.telemetry_posted = telemetry_ok
            episode.telemetry_error = telemetry_error

    if output_dir is not None:
        _write_evaluation_artifacts(output_dir, report)

    return report


def evaluate_final_project_tasks(
    llm: Any,
    state_graph: Any,
    env: Any,
    *,
    student_id: str,
    task_names: Union[str, Sequence[str]],
    variation_sample_count: int = 1,
    simplifications: str = DEFAULT_FINAL_PROJECT_SIMPLIFICATIONS,
    initial_graph_state: Optional[Dict[str, Any]] = None,
    auto_resolve_ambiguity: bool = True,
    print_progress: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
) -> FinalProjectEvaluationReport:
    """Evaluate a cyclic student graph on explicit practice task(s).

    The provided graph must own the episode loop. The grader loads and resets the env,
    then invokes the graph once per episode with an initial state that includes at least:
    `llm`, `env`, `student_id`, `task_name`, `variation_idx`, `observation`, `info`,
    `task_description`, `valid_actions`, `corpus`, `trajectory`, `step_index`,
    `turn_count`, `total_reward`, `final_score`, `completed`, and
    `auto_resolve_ambiguity`.

    The graph is expected to call `env.step()` itself until the episode ends, then return
    a final state mapping that includes `trajectory`. Aggregate fields such as
    `completed`, `turn_count`, `total_reward`, and `final_score` are accepted if present
    and otherwise derived from the returned trajectory.
    """

    return _evaluate_final_project_tasks(
        llm=llm,
        state_graph=state_graph,
        env=env,
        student_id=student_id,
        task_names=_coerce_task_names(task_names),
        variation_sample_count=variation_sample_count,
        simplifications=simplifications,
        telemetry_url=DEFAULT_FINAL_PROJECT_TELEMETRY_URL,
        initial_graph_state=initial_graph_state,
        auto_resolve_ambiguity=auto_resolve_ambiguity,
        telemetry_timeout_seconds=3.0,
        print_progress=print_progress,
        submit_report=False,
        output_dir=output_dir,
    )


def grade_final_project_unseen_tasks(
    llm: Any,
    state_graph: Any,
    env: Any,
    *,
    student_id: str,
    variation_sample_count: int = 3,
    simplifications: str = DEFAULT_FINAL_PROJECT_SIMPLIFICATIONS,
    telemetry_url: str = DEFAULT_FINAL_PROJECT_TELEMETRY_URL,
    initial_graph_state: Optional[Dict[str, Any]] = None,
    auto_resolve_ambiguity: bool = True,
    telemetry_timeout_seconds: float = 3.0,
    print_progress: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
) -> FinalProjectEvaluationReport:
    """Grade a cyclic student graph on the hidden Final Project tasks and submit telemetry.

    Unseen grading never writes local artifact files, even if `output_dir` is provided,
    so that hidden task identities are not leaked to students.
    """

    if hasattr(env, "get_unseen_task_names"):
        task_names = list(env.get_unseen_task_names())
    else:
        task_names = get_final_project_unseen_task_names()

    original_env_step_limit = getattr(env, "envStepLimit", None)
    if original_env_step_limit is not None:
        env.envStepLimit = 50

    try:
        return _evaluate_final_project_tasks(
            llm=llm,
            state_graph=state_graph,
            env=env,
            student_id=student_id,
            task_names=task_names,
            variation_sample_count=variation_sample_count,
            simplifications=simplifications,
            telemetry_url=telemetry_url,
            initial_graph_state=initial_graph_state,
            auto_resolve_ambiguity=auto_resolve_ambiguity,
            telemetry_timeout_seconds=telemetry_timeout_seconds,
            print_progress=print_progress,
            submit_report=True,
            output_dir=None,
        )
    finally:
        if original_env_step_limit is not None:
            env.envStepLimit = original_env_step_limit


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
    print_progress: bool = False,
    llm: Any = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> FinalProjectEvaluationReport:
    """Deprecated compatibility wrapper for the older env-owned grading entry point."""

    task_names = list(unseen_task_names or get_final_project_unseen_task_names())
    return _evaluate_final_project_tasks(
        llm=llm,
        state_graph=state_graph,
        env=env,
        student_id=student_id,
        task_names=task_names,
        variation_sample_count=variation_sample_count,
        simplifications=simplifications,
        telemetry_url=telemetry_url,
        initial_graph_state=initial_graph_state,
        auto_resolve_ambiguity=auto_resolve_ambiguity,
        telemetry_timeout_seconds=telemetry_timeout_seconds,
        print_progress=print_progress,
        submit_report=True,
        output_dir=None,
    )
