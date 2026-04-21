import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from cs103_scienceworld import (
    CS103ScienceWorldFinalProjectEnv,
    DEFAULT_FINAL_PROJECT_SIMPLIFICATIONS,
)


CHATOPENAI_KWARGS = {
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "temperature": 0.7,
    "base_url": "http://192.249.19.233:7000/v1",
    "timeout": 10,
    "max_retries": 0,
}


class AgentState(TypedDict, total=False):
    task_name: str
    variation_idx: int
    observation: str
    info: Dict[str, Any]
    valid_actions: List[str]
    corpus: List[str]
    retrieved_docs: List[str]
    prompt: str
    raw_output: str
    action: str


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


def lexical_retrieve(corpus: Sequence[str], query_text: str, top_k: int = 3) -> List[str]:
    query_tokens = set(re.findall(r"[a-z0-9]+", query_text.lower()))
    scored = []
    for doc in corpus:
        doc_tokens = set(re.findall(r"[a-z0-9]+", doc.lower()))
        overlap = len(query_tokens & doc_tokens)
        if overlap > 0:
            scored.append((overlap, doc))
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return [doc for _, doc in scored[:top_k]]


def parse_action_from_text(valid_actions: Sequence[str], raw_text: str) -> Optional[str]:
    if not raw_text:
        return None

    lut = {action.lower(): action for action in valid_actions}
    text = raw_text.strip()

    action_match = re.search(r"Action\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if action_match:
        candidate = action_match.group(1).strip().strip("`").strip()
        if candidate.lower() in lut:
            return lut[candidate.lower()]

    stripped = text.strip("`").strip()
    if stripped.lower() in lut:
        return lut[stripped.lower()]

    for line in text.splitlines():
        candidate = line.strip().strip("`").strip()
        if candidate.lower() in lut:
            return lut[candidate.lower()]

    normalized_text = text.lower()
    for action in valid_actions:
        if action.lower() in normalized_text:
            return action

    return None


def fallback_action(valid_actions: Sequence[str]) -> str:
    for action in valid_actions:
        if action.lower().startswith("wait"):
            return action
    return valid_actions[0]


def build_prompt(state: AgentState) -> str:
    docs = "\n\n".join(state.get("retrieved_docs", [])) or "<none>"
    candidates = "\n".join(f"- {action}" for action in state["valid_actions"][:40])
    inventory = str(state["info"].get("inv", "")).strip() or "<empty>"

    return f"""You are controlling an agent in ScienceWorld.
Pick exactly one next action from the candidate list.

Task: {state['info'].get('taskDesc', '')}
Observation:
{state['observation']}

Inventory:
{inventory}

Retrieved corpus snippets:
{docs}

Candidate actions:
{candidates}

Rules:
1. Choose exactly one action from the candidate list.
2. Do not invent actions.
3. Reply in this exact format:
Thought: <short reasoning>
Action: <exact action>"""


def retrieve_node(state: AgentState) -> AgentState:
        query_text = " ".join(
            [
                str(state["info"].get("taskDesc", "")),
                state["observation"],
            ]
        )
        return {"retrieved_docs": lexical_retrieve(state.get("corpus", []), query_text, top_k=3)}

def prompt_node(state: AgentState) -> AgentState:
    return {"prompt": build_prompt(state)}

def build_call_model_node(llm: ChatOpenAI):
    def call_model_node(state: AgentState) -> AgentState:
        try:
            response = llm.invoke(state["prompt"])
            raw_output = extract_text_content(response.content)
        except Exception as exc:
            raw_output = f"Thought: LLM call failed ({exc.__class__.__name__}).\nAction: {fallback_action(state['valid_actions'])}"
        return {"raw_output": raw_output}

    return call_model_node


def parse_action_node(state: AgentState) -> AgentState:
    action = parse_action_from_text(state["valid_actions"], state.get("raw_output", ""))
    if action is None:
        action = fallback_action(state["valid_actions"])
    return {"action": action}


def build_state_graph(llm: ChatOpenAI):
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("prompt", prompt_node)
    graph.add_node("call_model", build_call_model_node(llm))
    graph.add_node("parse_action", parse_action_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "prompt")
    graph.add_edge("prompt", "call_model")
    graph.add_edge("call_model", "parse_action")
    graph.add_edge("parse_action", END)
    return graph.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Final Project StateGraph grader.")
    parser.add_argument("--student-id", default="20260000")
    parser.add_argument("--variation-sample-count", type=int, default=1)
    parser.add_argument("--env-step-limit", type=int, default=40)
    parser.add_argument("--simplifications", default=DEFAULT_FINAL_PROJECT_SIMPLIFICATIONS)
    parser.add_argument("--task-name", default="")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def build_public_score_report(report: Any) -> Dict[str, Any]:
    return {
        "student_id": report.student_id,
        "total_score": report.total_score,
        "max_score": report.max_score,
        "average_score": report.average_score,
        "completed_episodes": report.completed_episodes,
        "total_episodes": report.total_episodes,
    }


def main() -> int:
    args = parse_args()
    env = CS103ScienceWorldFinalProjectEnv(envStepLimit=args.env_step_limit)
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "LOCAL_DUMMY"),
        **CHATOPENAI_KWARGS,
    )
    state_graph = build_state_graph(llm)

    try:
        report = env.grade_state_graph(
            state_graph=state_graph,
            student_id=args.student_id,
            variation_sample_count=args.variation_sample_count,
            simplifications=args.simplifications,
            unseen_task_names=[args.task_name] if args.task_name else None,
            print_summary=False,
        )
    finally:
        env.close()

    print(f"score: {report.total_score}/{report.max_score}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(build_public_score_report(report), ensure_ascii=False, indent=2))
        print(f"saved report: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
