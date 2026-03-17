import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from .common import create_env, run_episode, select_action


ASSIGNMENT_1_TASK_NAME = "assignment-1-prompting"
DEFAULT_ASSIGNMENT_1_VARIATIONS = tuple(range(5))


@dataclass
class Assignment1Plan:
    source_location: str
    target_object: str
    destination_location: str = ""
    requires_pickup: bool = False


def parse_assignment_1_task(task_description: str) -> Assignment1Plan:
    source_match = re.search(r"Go to the (.+?)\.", task_description)
    object_match = re.search(r"focus on the (.+?)\.", task_description, flags=re.IGNORECASE)
    destination_match = re.search(r"Then go to the (.+?)\.", task_description)

    if not source_match or not object_match:
        raise ValueError(f"Unable to parse assignment 1 task description: {task_description!r}")

    return Assignment1Plan(
        source_location=source_match.group(1).strip(),
        target_object=object_match.group(1).strip(),
        destination_location=destination_match.group(1).strip() if destination_match else "",
        requires_pickup="Pick up the" in task_description,
    )


def create_assignment_1_env(
    variation_idx: int = 0,
    simplifications: str = "easy",
    env_step_limit: int = 30,
    jar_path: Optional[str] = None,
):
    return create_env(
        ASSIGNMENT_1_TASK_NAME,
        variation_idx=variation_idx,
        simplifications=simplifications,
        env_step_limit=env_step_limit,
        jar_path=jar_path,
    )


def run_assignment_1_episode(
    agent,
    variation_idx: int = 0,
    simplifications: str = "easy",
    env_step_limit: int = 30,
    verbose: bool = False,
    jar_path: Optional[str] = None,
):
    env = create_assignment_1_env(
        variation_idx=variation_idx,
        simplifications=simplifications,
        env_step_limit=env_step_limit,
        jar_path=jar_path,
    )
    try:
        return run_episode(env, agent, verbose=verbose)
    finally:
        env.close()


class Assignment1PromptingTemplateAgent:
    """Student template for Assignment 1.

    Fill in the TODO sections. The overall agent structure is intentionally the
    same as the reference solution so students can focus on the missing parts.
    """

    def __init__(self):
        self.plan: Optional[Assignment1Plan] = None
        self.stage = "travel_to_source"

    def reset(self, env, observation: str, info: Dict[str, object]) -> None:
        """Called once at the start of each episode."""

        del env, observation
        self.plan = parse_assignment_1_task(str(info["taskDesc"]))
        self.stage = "travel_to_source"

        # TODO(Assignment 1):
        # Add any extra per-episode state reset you need.

    def build_prompt(
        self,
        observation: str,
        info: Dict[str, object],
        valid_actions: Sequence[str],
    ) -> str:
        """Build the prompt sent to your model.

        TODO(Assignment 1):
        - Rewrite this prompt.
        - Include enough context for the model to choose the next action.
        - Force the model to return exactly one action from `valid_actions`.
        """

        if self.plan is None:
            raise RuntimeError("reset() must be called before build_prompt().")

        preview = "\n".join(f"- {action}" for action in list(valid_actions)[:12])
        return (
            "TODO: replace this starter prompt with your own prompt.\n"
            f"Task: {info['taskDesc']}\n"
            f"Observation: {observation}\n"
            f"Current stage: {self.stage}\n"
            f"Target object: {self.plan.target_object}\n"
            f"Candidate actions:\n{preview}\n"
            "Return exactly one action from the candidate list."
        )

    def _advance_after_source(self) -> None:
        assert self.plan is not None
        if self.plan.requires_pickup:
            self.stage = "pick_up_object"
        elif self.plan.destination_location:
            self.stage = "travel_to_destination"
        else:
            self.stage = "focus_on_object"

    def _focus_action(self, valid_actions: Sequence[str]) -> Optional[str]:
        assert self.plan is not None
        inventory_hint = ["inventory"] if self.plan.requires_pickup else []
        action = self.find_action(
            valid_actions,
            startswith="focus on",
            include=[self.plan.target_object, *inventory_hint],
        )
        if action:
            return action
        return self.find_action(valid_actions, startswith="focus on", include=[self.plan.target_object])

    def choose_action(
        self,
        valid_actions: Sequence[str],
        prompt: str,
        observation: str,
        info: Dict[str, object],
    ) -> str:
        """Choose the next valid action.

        TODO(Assignment 1):
        - Replace this method with your prompting logic.
        - You may call an LLM, or implement another policy.
        - The returned string must be one of `valid_actions`.
        """

        del prompt, observation, info

        # TODO(Assignment 1):
        # A minimal safe fallback is:
        # return valid_actions[0]
        raise NotImplementedError("TODO: implement choose_action() for Assignment 1.")

    def find_action(
        self,
        valid_actions: Sequence[str],
        startswith: Optional[str] = None,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
    ) -> Optional[str]:
        """Helper method for deterministic valid-action matching."""

        return select_action(valid_actions, startswith=startswith, include=include, exclude=exclude)

    def find_travel_action(self, valid_actions: Sequence[str], location: str) -> Optional[str]:
        """Helper method for moving to a location."""

        return self.find_action(valid_actions, startswith="teleport to", include=[location]) or self.find_action(
            valid_actions,
            startswith="go to",
            include=[location],
        )

    def act(self, env, observation: str, info: Dict[str, object]) -> str:
        """Main step function used by the episode runner."""

        valid_actions = env.get_valid_action_object_combinations()
        prompt = self.build_prompt(observation, info, valid_actions)
        return self.choose_action(valid_actions, prompt, observation, info)
