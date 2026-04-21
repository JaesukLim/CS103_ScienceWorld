import os
import re
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from cs103_scienceworld import CS103ScienceWorldFinalProjectEnv


CHATOPENAI_KWARGS = {
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "temperature": 0.7,
    "base_url": "http://192.249.19.233:7000/v1",
    "timeout": 10,
    "max_retries": 0,
}


class AgentState(TypedDict, total=False):
    observation: str
    info: Dict[str, Any]
    valid_actions: List[str]
    corpus: List[str]
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


llm = ChatOpenAI(
    **CHATOPENAI_KWARGS,
)


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

    response = llm.invoke(prompt)
    raw_output = extract_text_content(response.content)
    action = parse_action(valid_actions, raw_output)
    if action is None:
        action = fallback_action(valid_actions)
    return {"action": action}


state_graph = StateGraph(AgentState)
state_graph.add_node("decide_action", decide_action)
state_graph.add_edge(START, "decide_action")
state_graph.add_edge("decide_action", END)


env = CS103ScienceWorldFinalProjectEnv(envStepLimit=40)
report = env.grade_state_graph(
    state_graph=state_graph,
    student_id="20260000",
    print_progress=True,
)

print(report)
print(f"score: {report.total_score}/{report.max_score}")
env.close()
