import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from cs103_scienceworld import CS103ScienceWorldHW6Env

from .common import create_env, run_episode, select_action
from .assignment2_recipe_db import get_assignment_2_recipe_documents


ASSIGNMENT_2_TASK_NAME = "assignment-2-rag-tool-use"
DEFAULT_ASSIGNMENT_2_VARIATIONS = tuple(range(5))


@dataclass
class Assignment2Plan:
    source_location: str
    result_name: str


def parse_assignment_2_task(task_description: str) -> Assignment2Plan:
    location_match = re.search(r"ingredients are in the (.+?)\.", task_description)
    result_match = re.search(r"focus on the (.+?)\.", task_description, flags=re.IGNORECASE)

    if not location_match or not result_match:
        raise ValueError(f"Unable to parse assignment 2 task description: {task_description!r}")

    return Assignment2Plan(
        source_location=location_match.group(1).strip(),
        result_name=result_match.group(1).strip(),
    )


def parse_recipe_text(recipe_text: str) -> Tuple[Optional[str], List[str]]:
    match = re.search(r"To make (.+?), you need to mix (.+?)\.", recipe_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None, []

    result_name = match.group(1).strip()
    ingredients = [item.strip() for item in match.group(2).split(",") if item.strip()]
    return result_name, ingredients


class SimpleKeywordRetriever:
    """Tiny lexical retriever for notebook experiments."""

    def __init__(self):
        self._documents: Dict[str, str] = {}

    def add(self, document_id: str, text: str) -> None:
        self._documents[document_id] = text

    def query(self, query_text: str, top_k: int = 1) -> List[str]:
        query_tokens = set(re.findall(r"[a-z0-9]+", query_text.lower()))
        scored = []
        for document_id, text in self._documents.items():
            doc_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
            overlap = len(query_tokens & doc_tokens)
            scored.append((overlap, document_id))

        scored.sort(reverse=True)
        return [document_id for overlap, document_id in scored[:top_k] if overlap > 0]

    def get(self, document_id: str) -> str:
        return self._documents[document_id]


def build_assignment_2_retriever() -> SimpleKeywordRetriever:
    retriever = SimpleKeywordRetriever()
    for result_name, recipe_text in get_assignment_2_recipe_documents().items():
        retriever.add(result_name, recipe_text)
    return retriever


def create_assignment_2_env(
    variation_idx: int = 0,
    simplifications: str = "easy,openContainers",
    env_step_limit: int = 40,
    jar_path: Optional[str] = None,
):
    return create_env(
        ASSIGNMENT_2_TASK_NAME,
        variation_idx=variation_idx,
        simplifications=simplifications,
        env_step_limit=env_step_limit,
        jar_path=jar_path,
        env_cls=CS103ScienceWorldHW6Env,
    )


def run_assignment_2_episode(
    agent,
    variation_idx: int = 0,
    simplifications: str = "easy,openContainers",
    env_step_limit: int = 40,
    verbose: bool = False,
    jar_path: Optional[str] = None,
):
    env = create_assignment_2_env(
        variation_idx=variation_idx,
        simplifications=simplifications,
        env_step_limit=env_step_limit,
        jar_path=jar_path,
    )
    try:
        return run_episode(env, agent, verbose=verbose)
    finally:
        env.close()


class Assignment2RAGToolUseTemplateAgent:
    """Student template for Assignment 2.

    Fill in the TODO sections. The overall agent structure is intentionally the
    same as the reference solution so students can focus on the missing parts.
    """

    CONTAINER_KEYWORDS = ("pot", "cup")

    def __init__(self):
        self.plan: Optional[Assignment2Plan] = None
        self.retriever = build_assignment_2_retriever()
        self.stage = "travel_to_kitchen"
        self.container_name: Optional[str] = None
        self.pending_ingredients: List[str] = []
        self.active_ingredient: Optional[str] = None

    def reset(self, env, observation: str, info: Dict[str, object]) -> None:
        """Called once at the start of each episode."""

        del env, observation
        self.plan = parse_assignment_2_task(str(info["taskDesc"]))
        self.retriever = build_assignment_2_retriever()
        self.stage = "travel_to_kitchen"
        self.container_name = None
        self.pending_ingredients = []
        self.active_ingredient = None

        # TODO(Assignment 2):
        # Add any extra per-episode state reset you need.

    def build_prompt(
        self,
        observation: str,
        info: Dict[str, object],
        valid_actions: Sequence[str],
        retrieved_notes: Sequence[str],
    ) -> str:
        """Build the prompt sent to your model.

        TODO(Assignment 2):
        - Rewrite this prompt.
        - Include the retrieved recipe notes in a useful way.
        - Force the model to return exactly one action from `valid_actions`.
        """

        preview = "\n".join(f"- {action}" for action in list(valid_actions)[:15])
        notes = "\n".join(f"- {note}" for note in retrieved_notes) or "- <no retrieved notes yet>"
        return (
            "TODO: replace this starter prompt with your own prompt.\n"
            f"Task: {info['taskDesc']}\n"
            f"Observation: {observation}\n"
            f"Controller stage: {self.stage}\n"
            f"Retrieved notes:\n{notes}\n"
            f"Candidate actions:\n{preview}\n"
            "Return exactly one valid action."
        )

    def retrieve_relevant_notes(self, query: str, top_k: int = 1) -> List[str]:
        """Retrieve recipe notes or other helpful passages.

        TODO(Assignment 2):
        - Replace this with your own retriever or vector DB if desired.
        - Return text passages, not ids.
        """

        # TODO(Assignment 2):
        # You can keep this simple lexical retriever, or replace it.
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
        """Choose the next valid action.

        TODO(Assignment 2):
        - Replace this method with your own controller.
        - You may call an LLM, or implement another policy.
        - The intended workflow is to query the external recipe corpus, not to
          rely on reading an in-environment recipe document.
        - The returned string must be one of `valid_actions`.
        """

        del prompt, observation, info

        # TODO(Assignment 2):
        # A minimal safe fallback is:
        # return valid_actions[0]
        raise NotImplementedError("TODO: implement choose_action() for Assignment 2.")

    def find_action(
        self,
        valid_actions: Sequence[str],
        startswith: Optional[str] = None,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
    ) -> Optional[str]:
        return select_action(valid_actions, startswith=startswith, include=include, exclude=exclude)

    def find_travel_action(self, valid_actions: Sequence[str], location: str) -> Optional[str]:
        return self.find_action(valid_actions, startswith="teleport to", include=[location]) or self.find_action(
            valid_actions,
            startswith="go to",
            include=[location],
        )

    def act(self, env, observation: str, info: Dict[str, object]) -> str:
        valid_actions = env.get_valid_action_object_combinations()
        query = self.plan.result_name if self.plan is not None else str(info["taskDesc"])
        retrieved_notes = self.retrieve_relevant_notes(query)
        prompt = self.build_prompt(observation, info, valid_actions, retrieved_notes)
        return self.choose_action(valid_actions, prompt, observation, info)
