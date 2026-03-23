import re
from typing import Dict, List, Optional, Sequence

from .assignment2_rag_tool_use_template import (
    Assignment2RAGToolUseTemplateAgent,
    parse_recipe_text,
)


class Assignment2RAGToolUseSolutionAgent(Assignment2RAGToolUseTemplateAgent):
    """Reference solution for Assignment 2.

    The agent uses two lightweight tools:
    - a lexical retriever for the recipe text,
    - a valid-action matcher that never emits out-of-space commands.
    """

    CONTAINER_KEYWORDS = ("pot", "cup")

    def __init__(self):
        super().__init__()
        self.stage = "travel_to_kitchen"
        self.container_name: Optional[str] = None
        self.pending_ingredients: List[str] = []
        self.active_ingredient: Optional[str] = None

    def reset(self, env, observation: str, info: Dict[str, object]) -> None:
        super().reset(env, observation, info)
        self.stage = "travel_to_kitchen"
        self.container_name = None
        self.pending_ingredients = []
        self.active_ingredient = None

    def build_prompt(
        self,
        observation: str,
        info: Dict[str, object],
        valid_actions: Sequence[str],
        retrieved_notes: Sequence[str],
    ) -> str:
        """Filled example for the Assignment 2 prompt TODO."""

        preview = "\n".join(f"- {action}" for action in list(valid_actions)[:15])
        notes = "\n".join(f"- {note}" for note in retrieved_notes) or "- <no retrieved notes yet>"
        return (
            "You are controlling an agent in ScienceWorld.\n"
            f"Task: {info['taskDesc']}\n"
            f"Observation: {observation}\n"
            f"Controller stage: {self.stage}\n"
            f"Retrieved notes:\n{notes}\n"
            f"Candidate actions:\n{preview}\n"
            "Return exactly one valid action."
        )

    def retrieve_relevant_notes(self, query: str, top_k: int = 1) -> List[str]:
        """Filled example for the Assignment 2 retrieval TODO.

        The reference solution keeps the default lexical retriever and returns
        the retrieved text passages directly.
        """

        document_ids = self.retriever.query(query, top_k=top_k)
        return [self.retriever.get(document_id) for document_id in document_ids]

    def _choose_container_action(self, valid_actions: Sequence[str]) -> Optional[str]:
        candidates = []
        for action in valid_actions:
            if not action.startswith("pick up "):
                continue

            object_name = action.replace("pick up ", "", 1).strip().lower()
            tokens = re.findall(r"[a-z]+", object_name)
            if any(keyword in tokens for keyword in self.CONTAINER_KEYWORDS):
                candidates.append(action)

        if not candidates:
            return None

        return sorted(candidates, key=len)[0]

    def _travel_to(self, valid_actions: Sequence[str], location: str) -> Optional[str]:
        return self.find_travel_action(valid_actions, location)

    def _parse_container_name(self, action: str) -> str:
        return action.replace("pick up ", "", 1).strip()

    def _container_terms(self) -> List[str]:
        terms = list(self.CONTAINER_KEYWORDS)
        if self.container_name:
            normalized = self.container_name.lower()
            terms.append(normalized)
            terms.extend(re.findall(r"[a-z]+", normalized))
        return sorted(set(term for term in terms if term))

    def _find_move_to_container_action(self, valid_actions: Sequence[str], ingredient: str) -> Optional[str]:
        container_terms = self._container_terms()
        candidates: List[str] = []

        for action in valid_actions:
            if not action.startswith("move"):
                continue
            normalized = action.lower()
            if ingredient.lower() not in normalized:
                continue
            if any(term in normalized for term in container_terms):
                candidates.append(action)

        if not candidates:
            return None

        return sorted(candidates, key=len)[0]

    def _find_mix_action(self, valid_actions: Sequence[str]) -> Optional[str]:
        container_terms = self._container_terms()
        candidates = [
            action
            for action in valid_actions
            if action.startswith("mix") and any(term in action.lower() for term in container_terms)
        ]
        if not candidates:
            return None
        return sorted(candidates, key=len)[0]

    def _find_examine_action(self, valid_actions: Sequence[str]) -> Optional[str]:
        container_terms = self._container_terms()
        candidates = [
            action
            for action in valid_actions
            if action.startswith("examine") and any(term in action.lower() for term in container_terms)
        ]
        if not candidates:
            return None
        return sorted(candidates, key=len)[0]

    def _parse_recipe_if_needed(self, observation: str = "") -> None:
        if self.pending_ingredients:
            return

        del observation
        query = self.plan.result_name if self.plan is not None else "recipe"
        retrieved = self.retrieve_relevant_notes(query, top_k=1)
        if retrieved:
            _, ingredients = parse_recipe_text(retrieved[0])
            self.pending_ingredients = ingredients

    def _focus_result(self, valid_actions: Sequence[str]) -> Optional[str]:
        assert self.plan is not None
        return self.find_action(
            valid_actions,
            startswith="focus on",
            include=[self.plan.result_name],
            exclude=["recipe", "instructions"],
        )

    def choose_action(
        self,
        valid_actions: Sequence[str],
        prompt: str,
        observation: str,
        info: Dict[str, object],
    ) -> str:
        """Filled example for the Assignment 2 controller TODO.

        The reference controller combines:
        - deterministic tool calls for retrieval and action matching,
        - a small state machine for sequencing the task.
        """

        del prompt, info
        assert self.plan is not None

        if self.stage == "travel_to_kitchen":
            action = self._travel_to(valid_actions, "kitchen")
            if action:
                self.stage = "open_cupboard"
                return action
            self.stage = "open_cupboard"

        if self.stage == "open_cupboard":
            action = self.find_action(valid_actions, startswith="open", include=["cupboard"])
            if action:
                self.stage = "take_container"
                return action
            self.stage = "take_container"

        if self.stage == "take_container":
            action = self._choose_container_action(valid_actions)
            if action:
                self.container_name = self._parse_container_name(action)
                self.stage = "travel_to_recipe"
                return action
            return "look around"

        if self.stage == "travel_to_recipe":
            action = self._travel_to(valid_actions, self.plan.source_location)
            if action:
                self.stage = "parse_recipe"
                return action
            self.stage = "parse_recipe"

        if self.stage == "parse_recipe":
            self._parse_recipe_if_needed(observation)
            self.stage = "collect_ingredients"

        if self.stage == "collect_ingredients":
            if not self.pending_ingredients:
                self.stage = "mix_container"
            else:
                self.active_ingredient = self.pending_ingredients[0]
                action = self.find_action(valid_actions, startswith="pick up", include=[self.active_ingredient])
                if action:
                    self.stage = "store_ingredient"
                    return action

                move_action = None
                if self.container_name:
                    move_action = self._find_move_to_container_action(valid_actions, self.active_ingredient)
                if move_action:
                    self.pending_ingredients.pop(0)
                    self.active_ingredient = None
                    return move_action

                return "look around"

        if self.stage == "store_ingredient":
            move_action = None
            if self.container_name and self.active_ingredient:
                move_action = self._find_move_to_container_action(valid_actions, self.active_ingredient)
            if move_action:
                self.pending_ingredients.pop(0)
                self.active_ingredient = None
                self.stage = "collect_ingredients"
                return move_action
            return "look around"

        if self.stage == "mix_container":
            if self.container_name:
                examine_action = self._find_examine_action(valid_actions)
                if examine_action:
                    self.stage = "mix_after_examine"
                    return examine_action
                mix_action = self._find_mix_action(valid_actions)
                if mix_action:
                    self.stage = "focus_result"
                    return mix_action
            return "look around"

        if self.stage == "mix_after_examine":
            if self.container_name:
                mix_action = self._find_mix_action(valid_actions)
                if mix_action:
                    self.stage = "focus_result"
                    return mix_action
            return "look around"

        if self.stage == "focus_result":
            action = self._focus_result(valid_actions)
            if action:
                return action
            return "look around"

        return "look around"
