import argparse
import json
import os
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from cs103_scienceworld import (
    CS103ScienceWorldFinalProjectEnv,
    DEFAULT_FINAL_PROJECT_SIMPLIFICATIONS,
    grade_final_project_unseen_tasks,
)


CHATOPENAI_KWARGS = {
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "temperature": 0.7,
    "base_url": "http://192.249.19.233:7000/v1",
    "timeout": 10,
    "max_retries": 0,
}

DEFAULT_STUDENT_ID = "20260000"
DEFAULT_VARIATION_SAMPLE_COUNT = 1
DEFAULT_ENV_STEP_LIMIT = 60
DEFAULT_MAX_LLM_CALLS_PER_EPISODE = 10
DEFAULT_REPORT_DIR = Path("artifacts/final_project")

ROOM_SEARCH_ORDERS = {
    "cook-unseen": ["living room", "bedroom", "kitchen", "art studio", "workshop"],
    "corrosion-unseen": ["living room", "bedroom", "kitchen", "art studio", "workshop"],
    "recipe-pipeline-unseen": ["living room", "bedroom", "art studio", "workshop", "kitchen"],
    "corrode-circuit-unseen": ["living room", "bedroom", "workshop", "art studio", "kitchen"],
}

KNOWN_ITEMS = [
    "potato",
    "baked potato",
    "burnt potato",
    "dough",
    "bread",
    "burnt bread",
    "marshmallow",
    "toasted marshmallow",
    "burnt marshmallow",
    "flour",
    "water",
    "jam",
    "banana",
    "peanut",
    "peanuts",
    "peanut butter sandwich",
    "jam sandwich",
    "banana sandwich",
    "peanut butter with jam sandwich",
    "peanut butter with banana sandwich",
    "metal pot",
    "glass cup",
    "salt water",
    "sodium chloride",
    "iron block",
    "rust",
    "wire",
    "corroded wire",
    "battery",
    "corroded battery",
    "signal light bulb",
    "sink",
    "stove",
    "oven",
]


class ReactState(TypedDict, total=False):
    llm: Any
    env: Any
    student_id: str
    task_name: str
    variation_idx: int
    step_index: int
    observation: str
    info: Dict[str, Any]
    valid_actions: List[str]
    corpus: List[str]
    trajectory: List[Dict[str, Any]]
    task_type: str
    current_room: str
    visited_rooms: List[str]
    discovered_items: List[str]
    source_name: str
    target_name: str
    target_product: str
    required_finishers: List[str]
    suggested_action: str
    retrieved_docs: List[str]
    prompt: str
    llm_raw: str
    thought: str
    action: str
    llm_calls: int
    total_reward: int
    final_score: int
    turn_count: int
    completed: bool
    water_added_to_pot: bool
    salt_water_prepared: bool
    heating_started: bool
    corrosion_started: bool
    target_component_focused: bool


@dataclass
class TelemetryRecord:
    payload: Dict[str, Any]
    received_at: str


class _TelemetryServer(ThreadingHTTPServer):
    def __init__(self, server_address, request_handler_class, output_path: Optional[Path]):
        super().__init__(server_address, request_handler_class)
        self.output_path = output_path
        self.records: List[TelemetryRecord] = []


class _TelemetryHandler(BaseHTTPRequestHandler):
    server: _TelemetryServer

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode("utf-8"))
        record = TelemetryRecord(
            payload=payload,
            received_at=datetime.now(timezone.utc).isoformat(),
        )
        self.server.records.append(record)
        if self.server.output_path is not None:
            self.server.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.server.output_path.open("a", encoding="utf-8") as output_file:
                output_file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        response = json.dumps({"status": "accepted"}).encode("utf-8")
        self.send_response(202)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args) -> None:
        del format, args


def start_local_telemetry_server(output_path: Path) -> Tuple[_TelemetryServer, threading.Thread, str]:
    server = _TelemetryServer(("127.0.0.1", 0), _TelemetryHandler, output_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    endpoint = f"http://{host}:{port}/final-project/telemetry"
    return server, thread, endpoint


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


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


def parse_action_from_llm(raw_text: str, valid_actions: Sequence[str]) -> Optional[str]:
    if not raw_text:
        return None

    text = raw_text.strip()
    exact_lut = {action.lower(): action for action in valid_actions}

    action_match = re.search(r"Action\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if action_match:
        candidate = action_match.group(1).strip().strip("`").strip()
        lowered = candidate.lower()
        if lowered in exact_lut:
            return exact_lut[lowered]

    stripped = text.strip("`").strip()
    if stripped.lower() in exact_lut:
        return exact_lut[stripped.lower()]

    for line in text.splitlines():
        candidate = line.strip().strip("`").strip()
        if candidate.lower() in exact_lut:
            return exact_lut[candidate.lower()]

    normalized_text = text.lower()
    for action in valid_actions:
        lowered = action.lower()
        if lowered in normalized_text:
            return action

    return None


def find_action(
    valid_actions: Sequence[str],
    startswith: Optional[str] = None,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> Optional[str]:
    normalized_startswith = startswith.lower() if startswith else None
    include_terms = [term.lower() for term in include if term]
    exclude_terms = [term.lower() for term in exclude if term]
    candidates: List[str] = []

    for action in valid_actions:
        lowered = action.lower()
        if normalized_startswith and not lowered.startswith(normalized_startswith):
            continue
        if any(term not in lowered for term in include_terms):
            continue
        if any(term in lowered for term in exclude_terms):
            continue
        candidates.append(action)

    if not candidates:
        return None
    return sorted(candidates, key=len)[0]


def find_travel_action(valid_actions: Sequence[str], room_name: str) -> Optional[str]:
    return (
        find_action(valid_actions, startswith="teleport to", include=[room_name])
        or find_action(valid_actions, startswith="go to", include=[room_name])
        or find_action(valid_actions, startswith="move to", include=[room_name])
        or find_action(valid_actions, startswith="open", include=[room_name, "door"])
    )


def parse_current_room(observation: str) -> str:
    lowered = observation.lower()
    for room_name in ["kitchen", "living room", "bedroom", "art studio", "workshop", "test kitchen", "test lab"]:
        if room_name in lowered:
            return room_name

    patterns = [
        r"you are in the ([a-z ]+)\.",
        r"you are in a ([a-z ]+)\.",
        r"you find yourself in the ([a-z ]+)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return match.group(1).strip()
    return ""


def parse_cook_description(task_desc: str) -> Tuple[str, str]:
    match = re.search(r"heat the (.+?) until it becomes (.+?)\.", task_desc, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    match = re.search(r"cook the (.+?) until it becomes (.+?)\.", task_desc, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", ""


def parse_corrosion_description(task_desc: str) -> Tuple[str, str]:
    patterns = [
        r"corrode the (.+?) until it becomes (.+?)\.",
        r"damage the (.+?) using corrosion until it becomes (.+?)\.",
        r"damage the target component until it becomes (.+?),",
    ]
    for pattern in patterns:
        match = re.search(pattern, task_desc, flags=re.IGNORECASE)
        if not match:
            continue
        if len(match.groups()) == 2:
            return match.group(1).strip(), match.group(2).strip()
        return "", match.group(1).strip()
    return "", ""


def lexical_retrieve(corpus: Sequence[str], query_terms: Sequence[str], top_k: int = 3) -> List[str]:
    query_tokens = {token for token in re.findall(r"[a-z0-9]+", " ".join(query_terms).lower()) if token}
    scored: List[Tuple[int, str]] = []
    for doc in corpus:
        doc_tokens = set(re.findall(r"[a-z0-9]+", doc.lower()))
        score = len(query_tokens & doc_tokens)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return [doc for _, doc in scored[:top_k]]


def deduce_recipe_target(discovered_items: Set[str]) -> Tuple[str, List[str]]:
    has_peanut = "peanut" in discovered_items or "peanuts" in discovered_items
    has_jam = "jam" in discovered_items
    has_banana = "banana" in discovered_items

    if has_peanut and has_jam:
        return "peanut butter with jam sandwich", ["peanut", "jam"]
    if has_peanut and has_banana:
        return "peanut butter with banana sandwich", ["peanut", "banana"]
    if has_jam:
        return "jam sandwich", ["jam"]
    if has_banana:
        return "banana sandwich", ["banana"]
    if has_peanut:
        return "peanut butter sandwich", ["peanut"]
    return "", []


def update_discovered_items(existing: Sequence[str], observation: str, info: Dict[str, Any], valid_actions: Sequence[str]) -> List[str]:
    inventory = str(info.get("inv", ""))
    haystack = "\n".join([observation, inventory, "\n".join(valid_actions)]).lower()
    discovered = set(existing)
    for item in KNOWN_ITEMS:
        if item.lower() in haystack:
            discovered.add(item.lower())
    return sorted(discovered)


def in_inventory(info: Dict[str, Any], object_name: str) -> bool:
    return object_name.lower() in str(info.get("inv", "")).lower()


def build_candidate_shortlist(state: ReactState, limit: int = 40) -> List[str]:
    valid_actions = list(state["valid_actions"])
    shortlisted: List[str] = []
    seen: Set[str] = set()

    def add(action: Optional[str]) -> None:
        if not action or action in seen:
            return
        seen.add(action)
        shortlisted.append(action)

    add(state.get("suggested_action"))
    for action in valid_actions:
        lowered = action.lower()
        if lowered.startswith(("focus on", "pick up", "move", "mix", "use", "activate", "pour", "dunk", "wait", "teleport to")):
            add(action)
        if len(shortlisted) >= limit:
            break

    if len(shortlisted) < limit:
        for action in valid_actions:
            add(action)
            if len(shortlisted) >= limit:
                break

    return shortlisted


def next_room_search_action(state: ReactState, room_order: Sequence[str]) -> Optional[str]:
    visited = set(state.get("visited_rooms", []))
    valid_actions = state["valid_actions"]
    for room_name in room_order:
        if room_name in visited:
            continue
        action = find_travel_action(valid_actions, room_name)
        if action:
            return action
    return None


def maybe_focus_target(valid_actions: Sequence[str], target_name: str) -> Optional[str]:
    if not target_name:
        return None
    return find_action(valid_actions, startswith="focus on", include=[target_name])


def heuristic_for_cook(state: ReactState) -> Optional[str]:
    valid_actions = state["valid_actions"]
    info = state["info"]
    source_name = state.get("source_name", "")
    target_name = state.get("target_name", "")
    current_room = state.get("current_room", "")

    focus_action = maybe_focus_target(valid_actions, target_name)
    if focus_action:
        return focus_action

    if state.get("heating_started"):
        activate_heat = find_action(valid_actions, startswith="activate", include=["stove"])
        if activate_heat:
            return activate_heat
        activate_heat = find_action(valid_actions, startswith="activate", include=["oven"])
        if activate_heat:
            return activate_heat
        wait_action = find_action(valid_actions, startswith="wait")
        if wait_action:
            return wait_action

    move_to_heat = find_action(valid_actions, startswith="move", include=[source_name, "stove"])
    if move_to_heat:
        return move_to_heat
    move_to_heat = find_action(valid_actions, startswith="move", include=[source_name, "oven"])
    if move_to_heat:
        return move_to_heat

    activate_heat = find_action(valid_actions, startswith="activate", include=["stove"])
    if activate_heat:
        return activate_heat
    activate_heat = find_action(valid_actions, startswith="activate", include=["oven"])
    if activate_heat:
        return activate_heat

    if in_inventory(info, source_name) and current_room != "kitchen":
        travel_kitchen = find_travel_action(valid_actions, "kitchen")
        if travel_kitchen:
            return travel_kitchen

    pick_source = find_action(valid_actions, startswith="pick up", include=[source_name])
    if pick_source:
        return pick_source

    if state.get("heating_started"):
        wait_action = find_action(valid_actions, startswith="wait")
        if wait_action:
            return wait_action

    search_action = next_room_search_action(state, ROOM_SEARCH_ORDERS["cook-unseen"])
    if search_action:
        return search_action

    wait_action = find_action(valid_actions, startswith="wait")
    if wait_action:
        return wait_action
    return None


def heuristic_for_corrosion(state: ReactState) -> Optional[str]:
    valid_actions = state["valid_actions"]
    info = state["info"]
    task_desc = str(info.get("taskDesc", ""))
    source_name = state.get("source_name", "")
    target_name = state.get("target_name", "")
    current_room = state.get("current_room", "")
    requires_mix = "prepare salt water" in task_desc.lower()

    focus_action = maybe_focus_target(valid_actions, target_name)
    if focus_action:
        return focus_action

    move_to_cup = find_action(valid_actions, startswith="move", include=[source_name, "glass cup"])
    if move_to_cup:
        return move_to_cup
    dunk_in_cup = find_action(valid_actions, startswith="dunk", include=[source_name, "glass cup"])
    if dunk_in_cup:
        return dunk_in_cup

    if state.get("corrosion_started"):
        wait_action = find_action(valid_actions, startswith="wait")
        if wait_action:
            return wait_action

    mix_cup = find_action(valid_actions, startswith="mix", include=["glass cup"])
    if mix_cup and (requires_mix or "sodium chloride" in "\n".join(valid_actions).lower()):
        return mix_cup

    move_salt = find_action(valid_actions, startswith="move", include=["sodium chloride", "glass cup"])
    if move_salt:
        return move_salt

    use_sink = find_action(valid_actions, startswith="use", include=["sink", "glass cup"])
    if use_sink and not state.get("salt_water_prepared"):
        return use_sink

    if in_inventory(info, source_name) and current_room != "kitchen":
        travel_kitchen = find_travel_action(valid_actions, "kitchen")
        if travel_kitchen:
            return travel_kitchen

    pick_source = find_action(valid_actions, startswith="pick up", include=[source_name])
    if pick_source:
        return pick_source

    pick_salt = find_action(valid_actions, startswith="pick up", include=["sodium chloride"])
    if pick_salt:
        return pick_salt

    search_action = next_room_search_action(state, ROOM_SEARCH_ORDERS["corrosion-unseen"])
    if search_action:
        return search_action

    wait_action = find_action(valid_actions, startswith="wait")
    if wait_action:
        return wait_action
    return None


def heuristic_for_corrode_circuit(state: ReactState) -> Optional[str]:
    valid_actions = state["valid_actions"]
    info = state["info"]
    task_desc = str(info.get("taskDesc", ""))
    requires_mix = "prepare salt water" in task_desc.lower()

    focus_corroded_wire = maybe_focus_target(valid_actions, "corroded wire")
    if focus_corroded_wire:
        return focus_corroded_wire

    focus_corroded_battery = maybe_focus_target(valid_actions, "corroded battery")
    if focus_corroded_battery:
        return focus_corroded_battery

    pour_into_sink = find_action(valid_actions, startswith="pour", include=["glass cup", "sink"])
    if pour_into_sink:
        return pour_into_sink

    move_cup_to_sink = find_action(valid_actions, startswith="move", include=["glass cup", "sink"])
    if move_cup_to_sink:
        return move_cup_to_sink

    if state.get("corrosion_started"):
        wait_action = find_action(valid_actions, startswith="wait")
        if wait_action:
            return wait_action

    mix_cup = find_action(valid_actions, startswith="mix", include=["glass cup"])
    if mix_cup and (requires_mix or "sodium chloride" in "\n".join(valid_actions).lower()):
        return mix_cup

    move_salt = find_action(valid_actions, startswith="move", include=["sodium chloride", "glass cup"])
    if move_salt:
        return move_salt

    use_sink = find_action(valid_actions, startswith="use", include=["sink", "glass cup"])
    if use_sink and not state.get("salt_water_prepared"):
        return use_sink

    pick_glass_cup = find_action(valid_actions, startswith="pick up", include=["glass cup"])
    if pick_glass_cup:
        return pick_glass_cup

    pick_salt = find_action(valid_actions, startswith="pick up", include=["sodium chloride"])
    if pick_salt:
        return pick_salt

    search_action = next_room_search_action(state, ROOM_SEARCH_ORDERS["corrode-circuit-unseen"])
    if search_action:
        return search_action

    wait_action = find_action(valid_actions, startswith="wait")
    if wait_action:
        return wait_action
    return None


def heuristic_for_recipe(state: ReactState) -> Optional[str]:
    valid_actions = state["valid_actions"]
    info = state["info"]
    discovered_items = set(state.get("discovered_items", []))
    target_product = state.get("target_product", "")
    required_finishers = state.get("required_finishers", [])
    current_room = state.get("current_room", "")

    focus_action = maybe_focus_target(valid_actions, target_product)
    if focus_action:
        return focus_action

    if state.get("heating_started"):
        activate_stove = find_action(valid_actions, startswith="activate", include=["stove"])
        if activate_stove:
            return activate_stove
        wait_action = find_action(valid_actions, startswith="wait")
        if wait_action:
            return wait_action

    if in_inventory(info, "flour") and current_room != "kitchen":
        travel_kitchen = find_travel_action(valid_actions, "kitchen")
        if travel_kitchen:
            return travel_kitchen

    move_flour = find_action(valid_actions, startswith="move", include=["flour", "metal pot"])
    if move_flour:
        return move_flour

    use_sink_on_pot = find_action(valid_actions, startswith="use", include=["sink", "metal pot"])
    if use_sink_on_pot and not state.get("water_added_to_pot"):
        return use_sink_on_pot

    mix_pot = find_action(valid_actions, startswith="mix", include=["metal pot"])
    if mix_pot and "dough" not in discovered_items and "bread" not in discovered_items:
        return mix_pot

    move_pot_to_stove = find_action(valid_actions, startswith="move", include=["metal pot", "stove"])
    if move_pot_to_stove and ("dough" in discovered_items or state.get("water_added_to_pot")):
        return move_pot_to_stove

    move_dough_to_stove = find_action(valid_actions, startswith="move", include=["dough", "stove"])
    if move_dough_to_stove:
        return move_dough_to_stove

    if "bread" in discovered_items:
        for finisher in required_finishers:
            move_finisher = find_action(valid_actions, startswith="move", include=[finisher, "metal pot"])
            if move_finisher:
                return move_finisher
        mix_final = find_action(valid_actions, startswith="mix", include=["metal pot"])
        if mix_final:
            return mix_final

    activate_stove = find_action(valid_actions, startswith="activate", include=["stove"])
    if activate_stove:
        return activate_stove

    # Prefer collecting whichever required ingredients are visible when no stronger kitchen action exists.
    for ingredient in ["flour"] + list(required_finishers or ["peanut", "jam", "banana"]):
        pick_action = find_action(valid_actions, startswith="pick up", include=[ingredient])
        if pick_action:
            return pick_action

    search_action = next_room_search_action(state, ROOM_SEARCH_ORDERS["recipe-pipeline-unseen"])
    if search_action:
        return search_action

    wait_action = find_action(valid_actions, startswith="wait")
    if wait_action:
        return wait_action
    return None


def heuristic_action(state: ReactState) -> Optional[str]:
    task_name = state["task_name"]
    if task_name == "cook-unseen":
        return heuristic_for_cook(state)
    if task_name == "corrosion-unseen":
        return heuristic_for_corrosion(state)
    if task_name == "corrode-circuit-unseen":
        return heuristic_for_corrode_circuit(state)
    if task_name == "recipe-pipeline-unseen":
        return heuristic_for_recipe(state)
    return None


def update_state_from_context(state: ReactState) -> ReactState:
    info = state["info"]
    observation = state["observation"]
    task_desc = str(info.get("taskDesc", ""))
    valid_actions = state["valid_actions"]
    discovered_items = set(update_discovered_items(state.get("discovered_items", []), observation, info, valid_actions))
    current_room = parse_current_room(observation) or state.get("current_room", "")

    source_name = state.get("source_name", "")
    target_name = state.get("target_name", "")
    if state["task_name"] == "cook-unseen" and (not source_name or not target_name):
        source_name, target_name = parse_cook_description(task_desc)
    elif state["task_name"] in {"corrosion-unseen", "corrode-circuit-unseen"} and not target_name:
        parsed_source, parsed_target = parse_corrosion_description(task_desc)
        if not source_name:
            source_name = parsed_source
        target_name = parsed_target or target_name

    target_product = state.get("target_product", "")
    required_finishers = list(state.get("required_finishers", []))
    if state["task_name"] == "recipe-pipeline-unseen":
        inferred_product, inferred_finishers = deduce_recipe_target(discovered_items)
        if inferred_product:
            target_product = inferred_product
        if inferred_finishers:
            required_finishers = inferred_finishers

    visited_rooms = list(state.get("visited_rooms", []))
    if current_room and current_room not in visited_rooms:
        visited_rooms.append(current_room)

    return {
        "task_type": state["task_name"],
        "current_room": current_room,
        "visited_rooms": visited_rooms,
        "discovered_items": sorted(discovered_items),
        "source_name": source_name,
        "target_name": target_name,
        "target_product": target_product,
        "required_finishers": required_finishers,
        "suggested_action": heuristic_action(
            {
                **state,
                "current_room": current_room,
                "visited_rooms": visited_rooms,
                "discovered_items": sorted(discovered_items),
                "source_name": source_name,
                "target_name": target_name,
                "target_product": target_product,
                "required_finishers": required_finishers,
            }
        ) or "",
    }


def build_prompt(state: ReactState) -> str:
    info = state["info"]
    task_desc = str(info.get("taskDesc", ""))
    inventory = str(info.get("inv", "")).strip() or "<empty>"
    discovered_items = ", ".join(state.get("discovered_items", [])) or "<none>"
    retrieved_docs = "\n\n".join(state.get("retrieved_docs", [])) or "<none>"
    shortlist = build_candidate_shortlist(state)
    trajectory = state.get("trajectory", [])[-6:]
    trajectory_lines = []
    for step in trajectory:
        trajectory_lines.append(
            f"- step={step.get('index')} action={step.get('action')} score={step.get('score')} reward={step.get('reward')}"
        )
    recent_trajectory = "\n".join(trajectory_lines) or "- <none>"
    suggested_action = state.get("suggested_action", "") or "<none>"
    target_hint = state.get("target_product") or state.get("target_name") or "<unknown>"

    return f"""You are a ReAct-style controller for ScienceWorld.
Choose exactly one next action from the candidate list.

Task name: {state['task_name']}
Task description: {task_desc}
Likely target/result: {target_hint}
Current room guess: {state.get('current_room', '') or '<unknown>'}
Inventory:
{inventory}

Observation:
{state['observation']}

Discovered items:
{discovered_items}

Recent trajectory:
{recent_trajectory}

Retrieved notes:
{retrieved_docs}

Heuristic suggestion:
{suggested_action}

Candidate actions:
{chr(10).join(f"- {action}" for action in shortlist)}

Rules:
1. Think briefly about the current subgoal.
2. If the heuristic suggestion is valid and sensible, prefer it.
3. You must output exactly one action from the candidate list.
4. Use this exact format:
Thought: <short reasoning>
Action: <exact action string>"""


class FinalProjectReactAgent:
    def __init__(self, max_llm_calls_per_episode: int = DEFAULT_MAX_LLM_CALLS_PER_EPISODE):
        self.max_llm_calls_per_episode = max_llm_calls_per_episode
        self.graph = self._compile_graph()

    def _compile_graph(self):
        graph = StateGraph(ReactState)
        graph.add_node("analyze", self.analyze_node)
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("reason", self.reason_node)
        graph.add_node("finalize", self.finalize_node)
        graph.add_node("act", self.act_node)
        graph.add_edge(START, "analyze")
        graph.add_edge("analyze", "retrieve")
        graph.add_edge("retrieve", "reason")
        graph.add_edge("reason", "finalize")
        graph.add_edge("finalize", "act")
        graph.add_conditional_edges(
            "act",
            self.route_after_act,
            {"continue": "analyze", "done": END},
        )
        return graph.compile()

    def analyze_node(self, state: ReactState) -> ReactState:
        updates = update_state_from_context(state)
        if "llm_calls" not in state:
            updates["llm_calls"] = 0
        if "water_added_to_pot" not in state:
            updates["water_added_to_pot"] = False
        if "salt_water_prepared" not in state:
            updates["salt_water_prepared"] = False
        if "heating_started" not in state:
            updates["heating_started"] = False
        if "corrosion_started" not in state:
            updates["corrosion_started"] = False
        if "target_component_focused" not in state:
            updates["target_component_focused"] = False
        return updates

    def retrieve_node(self, state: ReactState) -> ReactState:
        query_terms = [state["task_name"], str(state["info"].get("taskDesc", ""))]
        query_terms.extend(state.get("discovered_items", []))
        if state.get("target_name"):
            query_terms.append(state["target_name"])
        if state.get("target_product"):
            query_terms.append(state["target_product"])
        return {"retrieved_docs": lexical_retrieve(state.get("corpus", []), query_terms, top_k=3)}

    def reason_node(self, state: ReactState) -> ReactState:
        prompt = build_prompt(state)
        llm_calls = int(state.get("llm_calls", 0))
        suggested_action = state.get("suggested_action", "")

        should_call_llm = (
            llm_calls < self.max_llm_calls_per_episode
            and (
                state.get("step_index", 0) == 0
                or not suggested_action
                or state["task_name"] == "recipe-pipeline-unseen"
                or state.get("step_index", 0) % 5 == 0
            )
        )

        if not should_call_llm:
            fallback_thought = "Following the strongest heuristic action."
            return {
                "prompt": prompt,
                "thought": fallback_thought,
                "llm_raw": f"Thought: {fallback_thought}\nAction: {suggested_action or '<none>'}",
            }

        try:
            response = state["llm"].invoke(prompt)
            raw_text = extract_text_content(response.content)
            thought_match = re.search(r"Thought\s*:\s*(.+)", raw_text, flags=re.IGNORECASE)
            thought = thought_match.group(1).strip() if thought_match else raw_text.splitlines()[0].strip()
        except Exception as exc:
            fallback_thought = f"LLM call failed ({exc.__class__.__name__}); using heuristic fallback."
            return {
                "prompt": prompt,
                "thought": fallback_thought,
                "llm_raw": f"Thought: {fallback_thought}\nAction: {suggested_action or '<none>'}",
            }
        return {
            "prompt": prompt,
            "thought": thought,
            "llm_raw": raw_text,
            "llm_calls": llm_calls + 1,
        }

    def finalize_node(self, state: ReactState) -> ReactState:
        valid_actions = state["valid_actions"]
        llm_action = parse_action_from_llm(state.get("llm_raw", ""), valid_actions)
        action = llm_action or state.get("suggested_action") or find_action(valid_actions, startswith="wait")
        if action is None:
            action = valid_actions[0]

        updates: ReactState = {"action": action}
        visited_rooms = list(state.get("visited_rooms", []))

        room_match = re.search(r"(?:teleport to|go to|move to) (.+)$", action, flags=re.IGNORECASE)
        if room_match:
            room_name = room_match.group(1).strip().lower()
            if room_name not in visited_rooms:
                visited_rooms.append(room_name)
            updates["visited_rooms"] = visited_rooms

        lowered = action.lower()
        if lowered.startswith("use sink on metal pot"):
            updates["water_added_to_pot"] = True
        if lowered.startswith("mix glass cup"):
            updates["salt_water_prepared"] = True
        if "to stove" in lowered and ("metal pot" in lowered or "dough" in lowered or "potato" in lowered or "marshmallow" in lowered):
            updates["heating_started"] = True
        if "to glass cup" in lowered or lowered.startswith("pour glass cup into sink") or lowered.startswith("move glass cup to sink"):
            updates["corrosion_started"] = True
        if lowered.startswith("focus on") and ("corroded wire" in lowered or "corroded battery" in lowered):
            updates["target_component_focused"] = True

        return updates

    def act_node(self, state: ReactState) -> ReactState:
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

    def route_after_act(self, state: ReactState) -> str:
        if state.get("completed", False):
            return "done"
        if state.get("step_index", 0) >= state["env"].envStepLimit:
            return "done"
        return "continue"


def build_initial_graph_state() -> ReactState:
    return {
        "visited_rooms": [],
        "discovered_items": [],
        "llm_calls": 0,
        "water_added_to_pot": False,
        "salt_water_prepared": False,
        "heating_started": False,
        "corrosion_started": False,
        "target_component_focused": False,
    }


def save_report(report: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(include_task_name_restore_map=True), ensure_ascii=False, indent=2)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight ReAct LangGraph agent on CS103 Final Project grading.")
    parser.add_argument("--student-id", default=DEFAULT_STUDENT_ID)
    parser.add_argument("--variation-sample-count", type=int, default=DEFAULT_VARIATION_SAMPLE_COUNT)
    parser.add_argument("--env-step-limit", type=int, default=DEFAULT_ENV_STEP_LIMIT)
    parser.add_argument("--max-llm-calls-per-episode", type=int, default=DEFAULT_MAX_LLM_CALLS_PER_EPISODE)
    parser.add_argument("--simplifications", default=DEFAULT_FINAL_PROJECT_SIMPLIFICATIONS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--report-name", default="")
    parser.add_argument("--telemetry-name", default="")
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_dir = args.report_dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_name = args.report_name or f"report_{args.student_id}_{timestamp}.json"
    telemetry_name = args.telemetry_name or f"telemetry_{args.student_id}_{timestamp}.jsonl"

    telemetry_path = report_dir / telemetry_name
    report_path = report_dir / report_name

    telemetry_server, telemetry_thread, telemetry_url = start_local_telemetry_server(telemetry_path)
    del telemetry_thread

    agent = FinalProjectReactAgent(max_llm_calls_per_episode=args.max_llm_calls_per_episode)
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        **CHATOPENAI_KWARGS,
    )
    env = CS103ScienceWorldFinalProjectEnv(envStepLimit=args.env_step_limit)

    try:
        report = grade_final_project_unseen_tasks(
            llm=llm,
            state_graph=agent.graph,
            env=env,
            student_id=args.student_id,
            variation_sample_count=args.variation_sample_count,
            simplifications=args.simplifications,
            telemetry_url=telemetry_url,
            initial_graph_state=build_initial_graph_state(),
            print_progress=True,
        )
        print(report)
        save_report(report, report_path)
        print(f"Saved report to {report_path}")
        print(f"Saved telemetry to {telemetry_path}")
        if args.print_json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    finally:
        env.close()
        telemetry_server.shutdown()
        telemetry_server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
