from typing import Dict, Optional, Sequence

from .assignment1_prompting_template import Assignment1PromptingTemplateAgent


class Assignment1PromptingSolutionAgent(Assignment1PromptingTemplateAgent):
    """Reference solution for Assignment 1.

    This agent is intentionally simple:
    - it parses the structured task description,
    - it tracks a tiny stage machine,
    - and it always picks from the current valid action list.
    """

    def __init__(self):
        super().__init__()
        self.stage = "travel_to_source"

    def reset(self, env, observation: str, info: Dict[str, object]) -> None:
        super().reset(env, observation, info)
        self.stage = "travel_to_source"

    def build_prompt(
        self,
        observation: str,
        info: Dict[str, object],
        valid_actions: Sequence[str],
    ) -> str:
        """Filled example for the Assignment 1 prompt TODO.

        The reference solution still uses a deterministic controller below,
        but this method shows one concrete prompt shape students can start from.
        """

        assert self.plan is not None
        preview = "\n".join(f"- {action}" for action in list(valid_actions)[:12])
        return (
            "You are controlling an agent in ScienceWorld.\n"
            f"Task: {info['taskDesc']}\n"
            f"Current observation: {observation}\n"
            f"Current subgoal: {self.stage}\n"
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
        """Filled example for the Assignment 1 action-selection TODO.

        Instead of calling an LLM, the reference solution uses a small stage
        machine and only returns actions already present in `valid_actions`.
        """

        del prompt, observation, info

        assert self.plan is not None

        if self.stage == "travel_to_source":
            action = self.find_travel_action(valid_actions, self.plan.source_location)
            if action:
                self._advance_after_source()
                return action
            self._advance_after_source()

        if self.stage == "pick_up_object":
            action = self.find_action(valid_actions, startswith="pick up", include=[self.plan.target_object])
            if action:
                self.stage = "travel_to_destination" if self.plan.destination_location else "focus_on_object"
                return action
            return "look around"

        if self.stage == "travel_to_destination":
            action = self.find_travel_action(valid_actions, self.plan.destination_location)
            if action:
                self.stage = "focus_on_object"
                return action
            self.stage = "focus_on_object"

        if self.stage == "focus_on_object":
            action = self._focus_action(valid_actions)
            if action:
                return action
            return "look around"

        return "look around"
