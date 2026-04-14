import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

from cs103_scienceworld import CS103ScienceWorldEnv, CS103ScienceWorldHW6Env

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
except ImportError:  # pragma: no cover - optional dependency
    StrOutputParser = None
    PromptTemplate = None

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # pragma: no cover - optional dependency
    END = None
    START = None
    StateGraph = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None


ASSIGNMENT_2_TASK_NAME = "assignment-2-rag-tool-use"
DEFAULT_ASSIGNMENT_2_VARIATIONS = tuple(range(5))
OPENAI_API_KEY = ""
MODEL_NAME = "gpt-4o-mini"
EASY_SIMPLIFICATIONS = (
    "teleportAction",
    "openDoors",
    "selfWateringFlowerPots",
    "noElectricalAction",
)


@dataclass
class Assignment2Plan:
    source_location: str
    result_name: str


class Assignment2GraphState(TypedDict, total=False):
    observation: str
    info: Dict[str, object]
    valid_actions: List[str]
    query: str
    top_k: int
    retrieved_notes: List[str]
    stage: str
    recent_actions: List[str]
    prompt: str
    raw_output: str
    action: str


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


def require_langgraph() -> None:
    if PromptTemplate is None or StrOutputParser is None:
        raise ImportError("Install `langchain` to use this assignment agent.")
    if StateGraph is None or START is None or END is None:
        raise ImportError("Install `langgraph` to use this assignment agent.")


def build_default_llm(api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.0):
    require_langgraph()
    if ChatOpenAI is None:
        raise ImportError("Install `langchain-openai` to use the default assignment LLM.")
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "API key is empty. Fill in the assignment file's OPENAI_API_KEY placeholder "
            "or pass a custom LangChain-compatible `llm=...` object."
        )
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


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
    env_cls: type[CS103ScienceWorldEnv] = CS103ScienceWorldEnv,
) -> CS103ScienceWorldEnv:
    env = env_cls("", jar_path, envStepLimit=env_step_limit)
    env.load(task_name, variation_idx, normalize_simplifications(simplifications), generateGoldPath=False)
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


def coerce_action_from_text(valid_actions: Sequence[str], raw_text: str) -> Optional[str]:
    if not raw_text:
        return None
    cleaned = raw_text.strip().strip("`").strip()
    cleaned = re.sub(r"^action\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    normalized = cleaned.lower()
    exact = {action.lower(): action for action in valid_actions}
    if normalized in exact:
        return exact[normalized]
    first_line = normalized.splitlines()[0].strip() if normalized.splitlines() else normalized
    if first_line in exact:
        return exact[first_line]
    for action in valid_actions:
        action_lower = action.lower()
        if action_lower in normalized or action_lower in first_line:
            return action
    return None


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
        steps.append(
            EpisodeStep(
                index=step_idx,
                action=action,
                observation=observation,
                reward=reward,
                score=final_score,
                completed=completed,
            )
        )
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


def build_assignment_2_retriever(
    recipe_documents: Optional[Dict[str, str]] = None,
) -> SimpleKeywordRetriever:
    retriever = SimpleKeywordRetriever()
    for result_name, recipe_text in (recipe_documents or {}).items():
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

    PROMPT_TEMPLATE = """You are controlling an agent in ScienceWorld.
Pick exactly one next action from the candidate list.
Task: {task_desc}
Observation: {observation}
Controller stage: {stage}
Retrieved notes:
{retrieved_notes}
Recent actions:
{recent_actions}
Candidate actions:
{candidate_actions}
Rules:
- Return exactly one candidate action.
- Do not explain your answer.
- Use the retrieved recipe notes to decide what ingredient or tool to handle next."""

    def __init__(self, llm=None, retriever: Optional[SimpleKeywordRetriever] = None):
        require_langgraph()
        self.plan: Optional[Assignment2Plan] = None
        self.retriever = retriever or build_assignment_2_retriever()
        self._initial_retriever = self.retriever
        self.stage = "travel_to_kitchen"
        self.container_name: Optional[str] = None
        self.pending_ingredients: List[str] = []
        self.active_ingredient: Optional[str] = None
        self.llm = llm or build_default_llm(api_key=OPENAI_API_KEY, model=MODEL_NAME)
        self.action_history = []
        self.prompt_template = PromptTemplate.from_template(self.PROMPT_TEMPLATE)
        self.output_parser = StrOutputParser()
        self.graph = self.compile_graph()

    def reset(self, env, observation: str, info: Dict[str, object]) -> None:
        """Called once at the start of each episode."""

        del env, observation
        self.plan = parse_assignment_2_task(str(info["taskDesc"]))
        self.retriever = self._initial_retriever
        self.stage = "travel_to_kitchen"
        self.container_name = None
        self.pending_ingredients = []
        self.active_ingredient = None
        self.action_history = []

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
        recent_actions = "\n".join(f"- {action}" for action in self.action_history[-6:]) or "- <none yet>"
        return self.prompt_template.format(
            task_desc=info["taskDesc"],
            observation=observation,
            stage=self.stage,
            retrieved_notes=notes,
            recent_actions=recent_actions,
            candidate_actions=preview,
        )

    def build_initial_state(
        self,
        observation: str,
        info: Dict[str, object],
        valid_actions: Sequence[str],
    ) -> Assignment2GraphState:
        """Construct the per-step LangGraph state.

        TODO(Assignment 2):
        - Decide what fields your retrieval and action nodes should share.
        - Add any extra state fields your graph will need.
        """

        query = self.plan.result_name if self.plan is not None else str(info["taskDesc"])
        return {
            "observation": observation,
            "info": info,
            "valid_actions": list(valid_actions),
            "query": query,
            "top_k": 1,
            "stage": self.stage,
            "recent_actions": list(self.action_history[-6:]),
        }

    def add_graph_nodes(self, graph) -> None:
        """Register the nodes used by your LangGraph agent.

        TODO(Assignment 2):
        - Add a retrieval node and action-selection nodes.
        - If you want extra planning or validation nodes, add them here.
        """
        raise NotImplementedError("TODO: register your HW6 LangGraph nodes in add_graph_nodes().")

    def add_graph_edges(self, graph) -> None:
        """Connect the nodes in your LangGraph agent.

        TODO(Assignment 2):
        - Wire retrieval into prompting and action selection.
        - Add branches or retry loops here if your graph needs them.
        """
        raise NotImplementedError("TODO: connect your HW6 LangGraph edges in add_graph_edges().")

    def compile_graph(self):
        """Create and compile the LangGraph workflow for one action step."""

        graph = StateGraph(Assignment2GraphState)
        self.add_graph_nodes(graph)
        self.add_graph_edges(graph)
        return graph.compile()

    def retrieve_notes_node(self, state: Assignment2GraphState) -> Assignment2GraphState:
        """Node 1: retrieve external recipe notes for the current task.

        TODO(Assignment 2):
        - Read the retrieval query from the graph state.
        - Call your retriever or tool.
        - Return a state update like `{"retrieved_notes": ...}`.
        """

        raise NotImplementedError("TODO: implement retrieve_notes_node() for HW6.")

    def prepare_prompt_node(self, state: Assignment2GraphState) -> Assignment2GraphState:
        """Node 2: build the prompt using observation plus retrieved notes.

        TODO(Assignment 2):
        - Read the current observation, task description, valid actions, and retrieved notes.
        - Build a prompt string for your LLM.
        - Return a state update like `{"prompt": ...}`.
        """

        raise NotImplementedError("TODO: implement prepare_prompt_node() for HW6.")

    def call_model_node(self, state: Assignment2GraphState) -> Assignment2GraphState:
        """Node 3: call the LLM with the prompt from the previous node.

        TODO(Assignment 2):
        - Read `state["prompt"]`.
        - Call your model.
        - Return a state update like `{"raw_output": ...}`.
        """

        raise NotImplementedError("TODO: implement call_model_node() for HW6.")

    def coerce_action_node(self, state: Assignment2GraphState) -> Assignment2GraphState:
        """Node 4: map model output back to a valid environment action.

        TODO(Assignment 2):
        - Read the model output from `state["raw_output"]`.
        - Convert it to one action from `state["valid_actions"]`.
        - Return a state update like `{"action": ...}`.
        - Add repair logic if the model output is not directly usable.
        """

        raise NotImplementedError("TODO: implement coerce_action_node() for HW6.")

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
        - You should route retrieval and action selection through your LangGraph state graph.
        - The intended workflow is to query the external recipe corpus, not to
          rely on reading an in-environment recipe document.
        - The returned string must be one of `valid_actions`.
        """

        del prompt
        final_state = self.graph.invoke(self.build_initial_state(observation, info, valid_actions))
        return final_state["action"]

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
        action = self.choose_action(valid_actions, "", observation, info)
        self.action_history.append(action)
        return action
