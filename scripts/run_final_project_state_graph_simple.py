import argparse
import json
import re
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from cs103_scienceworld import CS103ScienceWorldFinalProjectEnv, evaluate_final_project_tasks


CHATOPENAI_KWARGS = {
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "temperature": 0.7,
    "base_url": "http://192.249.19.233:7000/v1",
    "timeout": 10,
    "max_retries": 0,
    "api_key": "LOCAL_DUMMY",
}


class AgentState(TypedDict, total=False):
    """Minimal student-facing graph contract for cyclic Final Project episodes.

    The grader injects `llm`, `env`, `student_id`, `task_name`, `variation_idx`,
    `observation`, `info`, `task_description`, `valid_actions`, `corpus`, `score`,
    `reward`, `auto_resolve_ambiguity`, and bookkeeping fields such as `trajectory`,
    `step_index`, `turn_count`, `total_reward`, `final_score`, and `completed`.
    The graph must call `env.step()` itself until the episode ends and return the final state.
    """

    llm: Any
    env: Any
    student_id: str
    task_name: str
    variation_idx: int
    observation: str
    info: Dict[str, Any]
    task_description: str
    valid_actions: List[str]
    corpus: List[str]
    trajectory: List[Dict[str, Any]]
    action: str
    step_index: int
    score: int
    reward: int
    turn_count: int
    total_reward: int
    final_score: int
    completed: bool
    auto_resolve_ambiguity: bool


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def parse_action(valid_actions: Sequence[str], raw_text: str) -> Optional[str]:
    if not raw_text:
        return None

    lookup = {action.lower(): action for action in valid_actions}
    text = raw_text.strip()

    match = re.search(r"Action\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().strip("`").strip()
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    stripped = text.strip("`").strip()
    if stripped.lower() in lookup:
        return lookup[stripped.lower()]

    for line in text.splitlines():
        candidate = line.strip().strip("`").strip()
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    for action in valid_actions:
        if action.lower() in text.lower():
            return action

    return None


def fallback_action(valid_actions: Sequence[str]) -> str:
    for action in valid_actions:
        if action.lower().startswith("wait"):
            return action
    return valid_actions[0]


def decide_action(state: AgentState) -> AgentState:
    observation = state.get("observation", "")
    info = state.get("info", {})
    valid_actions = state.get("valid_actions", [])
    corpus = state.get("corpus", [])

    query = " ".join([str(info.get("taskDesc", "")), observation]).strip().lower()
    query_words = set(re.findall(r"[a-z0-9]+", query))
    ranked_docs = []
    for doc in corpus:
        doc_words = set(re.findall(r"[a-z0-9]+", doc.lower()))
        score = len(query_words & doc_words)
        if score > 0:
            ranked_docs.append((score, doc))
    ranked_docs.sort(key=lambda item: (-item[0], len(item[1])))
    docs = "\n\n".join(doc for _, doc in ranked_docs[:3]) or "<none>"
    candidates = "\n".join(f"- {action}" for action in valid_actions)

    prompt = f"""You are controlling a ScienceWorld agent.
Pick exactly one action from the valid action list.

Task: {info.get('taskDesc', '')}
Observation:
{observation}

Info:
{info}

Corpus snippets:
{docs}

Valid actions:
{candidates}

Reply like this:
Thought: short reasoning
Action: exact action"""

    response = state["llm"].invoke(prompt)
    raw_output = extract_text_content(response.content)
    action = parse_action(valid_actions, raw_output)
    if action is None:
        action = fallback_action(valid_actions)
    return {"action": action}


def step_env(state: AgentState) -> AgentState:
    env = state["env"]
    action = state["action"]
    observation, reward, completed, info = env.step(action)
    trajectory = list(state.get("trajectory", []))
    trajectory.append(
        {
            "index": len(trajectory),
            "action": action,
            "observation": observation,
            "reward": int(reward),
            "score": int(info.get("score", 0)),
            "completed": bool(completed),
            "moves": int(info.get("moves", len(trajectory) + 1)),
            "auto_resolved": False,
        }
    )
    return {
        "observation": observation,
        "info": info,
        "valid_actions": env.get_valid_action_object_combinations(),
        "trajectory": trajectory,
        "step_index": len(trajectory),
        "turn_count": len(trajectory),
        "total_reward": int(state.get("total_reward", 0)) + int(reward),
        "final_score": int(info.get("score", 0)),
        "completed": bool(completed),
    }


def route_after_step(state: AgentState) -> str:
    if state.get("completed", False):
        return "done"
    if state.get("step_index", 0) >= state["env"].envStepLimit:
        return "done"
    return "continue"


def build_graph():
    state_graph = StateGraph(AgentState)
    state_graph.add_node("decide_action", decide_action)
    state_graph.add_node("step_env", step_env)
    state_graph.add_edge(START, "decide_action")
    state_graph.add_edge("decide_action", "step_env")
    state_graph.add_conditional_edges(
        "step_env",
        route_after_step,
        {"continue": "decide_action", "done": END},
    )
    return state_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal cyclic LangGraph agent on a named CS103 Final Project task.")
    parser.add_argument("--task-name", default="recipe-pipeline-tiny")
    parser.add_argument("--student-id", default="20260000")
    parser.add_argument("--env-step-limit", type=int, default=40)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    llm = ChatOpenAI(**CHATOPENAI_KWARGS)
    env = CS103ScienceWorldFinalProjectEnv(envStepLimit=args.env_step_limit)
    state_graph = build_graph()
    try:
        report = evaluate_final_project_tasks(
            llm=llm,
            state_graph=state_graph,
            env=env,
            student_id=args.student_id,
            task_names=args.task_name,
            variation_sample_count=1,
            print_progress=True,
        )
        print(report)
        print(f"score: {report.total_score}/{report.max_score}")
        if args.print_json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
