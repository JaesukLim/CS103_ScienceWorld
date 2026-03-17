from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from cs103_scienceworld import CS103ScienceWorldEnv

EASY_SIMPLIFICATIONS = (
    "teleportAction",
    "openDoors",
    "selfWateringFlowerPots",
    "noElectricalAction",
)


@dataclass
class EpisodeStep:
    index: int
    action: str
    observation: str
    reward: int
    score: int
    completed: bool


@dataclass
class EpisodeResult:
    task_name: str
    variation_idx: int
    final_score: int
    completed: bool
    steps: List[EpisodeStep] = field(default_factory=list)


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


def create_env(
    task_name: str,
    variation_idx: int = 0,
    simplifications: str = "easy",
    env_step_limit: int = 50,
    jar_path: Optional[str] = None,
    generate_gold_path: bool = False,
) -> CS103ScienceWorldEnv:
    env = CS103ScienceWorldEnv("", jar_path, envStepLimit=env_step_limit)
    simplifications = normalize_simplifications(simplifications)
    env.load(
        task_name,
        variation_idx,
        simplifications,
        generateGoldPath=generate_gold_path,
    )
    return env


def select_action(
    valid_actions: Sequence[str],
    startswith: Optional[str] = None,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> Optional[str]:
    normalized_startswith = startswith.lower() if startswith else None
    include_terms = [term.lower() for term in include if term]
    exclude_terms = [term.lower() for term in exclude if term]

    matches: List[str] = []
    for action in valid_actions:
        normalized = action.lower()
        if normalized_startswith and not normalized.startswith(normalized_startswith):
            continue
        if any(term not in normalized for term in include_terms):
            continue
        if any(term in normalized for term in exclude_terms):
            continue
        matches.append(action)

    if not matches:
        return None

    return sorted(matches, key=len)[0]


def run_episode(
    env: CS103ScienceWorldEnv,
    agent: Any,
    max_steps: Optional[int] = None,
    verbose: bool = False,
    auto_resolve_ambiguity: bool = True,
) -> EpisodeResult:
    observation, info = env.reset()
    if hasattr(agent, "reset"):
        agent.reset(env, observation, info)

    limit = max_steps or env.envStepLimit
    steps: List[EpisodeStep] = []
    final_score = int(info.get("score", 0))
    completed = False

    for step_idx in range(limit):
        action = agent.act(env, observation, info)
        if not isinstance(action, str) or not action.strip():
            raise ValueError("Agent.act() must return a non-empty action string.")

        observation, reward, completed, info = env.step(action)
        if auto_resolve_ambiguity:
            while observation.startswith("Ambiguous request:"):
                observation, reward, completed, info = env.step("0")
        final_score = int(info["score"])
        step = EpisodeStep(
            index=step_idx,
            action=action,
            observation=observation,
            reward=reward,
            score=final_score,
            completed=completed,
        )
        steps.append(step)

        if verbose:
            print(f"[{step_idx:02d}] {action}")
            print(observation)
            print(f"score={final_score} reward={reward} completed={completed}")
            print("")

        if completed:
            break

    return EpisodeResult(
        task_name=info["taskName"],
        variation_idx=int(info["variationIdx"]),
        final_score=final_score,
        completed=completed,
        steps=steps,
    )
