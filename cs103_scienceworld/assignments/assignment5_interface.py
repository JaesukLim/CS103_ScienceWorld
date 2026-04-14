import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .assignment1_prompting_template import (
    ASSIGNMENT_1_TASK_NAME,
    Assignment1PromptingTemplateAgent,
    create_assignment_1_env,
    run_assignment_1_episode,
)


ASSIGNMENT_5_NAME = "assignment5"
ASSIGNMENT_5_TASK_NAME = ASSIGNMENT_1_TASK_NAME


class Assignment5PromptingTemplateAgent(Assignment1PromptingTemplateAgent):
    """Student-facing name for CS103 Assignment5."""


def create_assignment_5_env(*args, **kwargs):
    return create_assignment_1_env(*args, **kwargs)


def run_assignment_5_episode(*args, **kwargs):
    return run_assignment_1_episode(*args, **kwargs)


def build_assignment5_submission(
    episode_result,
    agent_name: str,
    student_id: str = "",
    variation_idx: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    submission_variation = (
        episode_result.variation_idx if variation_idx is None else variation_idx
    )
    return {
        "assignment": ASSIGNMENT_5_NAME,
        "task_name": episode_result.task_name,
        "variation_idx": submission_variation,
        "student_id": student_id,
        "agent_name": agent_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "final_score": int(episode_result.final_score),
        "completed": bool(episode_result.completed),
        "step_count": len(episode_result.steps),
        "steps": [
            {
                "index": step.index,
                "action": step.action,
                "observation": step.observation,
                "reward": step.reward,
                "score": step.score,
                "completed": step.completed,
            }
            for step in episode_result.steps
        ],
        "metadata": metadata or {},
    }


def save_assignment5_submission(
    episode_result,
    output_path: Union[str, Path],
    agent_name: str,
    student_id: str = "",
    variation_idx: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    submission = build_assignment5_submission(
        episode_result=episode_result,
        agent_name=agent_name,
        student_id=student_id,
        variation_idx=variation_idx,
        metadata=metadata,
    )
    output = Path(output_path)
    output.write_text(json.dumps(submission, ensure_ascii=False, indent=2))
    return submission


def run_and_save_assignment5_submission(
    agent,
    output_path: Union[str, Path],
    variation_idx: int = 0,
    simplifications: str = "easy",
    env_step_limit: int = 30,
    verbose: bool = False,
    jar_path: Optional[str] = None,
    student_id: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = run_assignment_5_episode(
        agent,
        variation_idx=variation_idx,
        simplifications=simplifications,
        env_step_limit=env_step_limit,
        verbose=verbose,
        jar_path=jar_path,
    )
    return save_assignment5_submission(
        episode_result=result,
        output_path=output_path,
        agent_name=agent.__class__.__name__,
        student_id=student_id,
        variation_idx=variation_idx,
        metadata=metadata,
    )


def load_assignment5_submission(input_path: Union[str, Path]) -> Dict[str, Any]:
    return json.loads(Path(input_path).read_text())


def grade_assignment5_submission(submission: Dict[str, Any]) -> Dict[str, Any]:
    feedback = []

    if submission.get("assignment") != ASSIGNMENT_5_NAME:
        feedback.append("assignment field must be 'assignment5'.")
    if submission.get("task_name") != ASSIGNMENT_5_TASK_NAME:
        feedback.append(f"task_name must be '{ASSIGNMENT_5_TASK_NAME}'.")

    steps = submission.get("steps", [])
    if not isinstance(steps, list) or not steps:
        feedback.append("submission must include a non-empty steps list.")

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            feedback.append(f"step {idx} must be a JSON object.")
            continue
        if not step.get("action"):
            feedback.append(f"step {idx} is missing a non-empty action.")

    raw_score = submission.get("final_score", 0)
    try:
        final_score = int(raw_score)
    except (TypeError, ValueError):
        final_score = 0
        feedback.append("final_score must be an integer.")

    if feedback:
        return {
            "assignment": ASSIGNMENT_5_NAME,
            "passed_schema_checks": False,
            "score": 0,
            "max_score": 100,
            "completed": bool(submission.get("completed", False)),
            "feedback": feedback,
        }

    normalized_score = max(0, min(100, final_score))
    feedback.append(f"Recorded final_score: {normalized_score}")
    feedback.append(f"Completed: {bool(submission.get('completed', False))}")
    feedback.append(f"Step count: {len(steps)}")
    return {
        "assignment": ASSIGNMENT_5_NAME,
        "passed_schema_checks": True,
        "score": normalized_score,
        "max_score": 100,
        "completed": bool(submission.get("completed", False)),
        "feedback": feedback,
    }


def grade_assignment5_submission_file(input_path: Union[str, Path]) -> Dict[str, Any]:
    return grade_assignment5_submission(load_assignment5_submission(input_path))
