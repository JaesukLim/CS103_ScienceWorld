import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re

import uvicorn
from fastapi import FastAPI, Request


app = FastAPI()
app.state.output_dir = Path("./telemetry")
app.state.records = []


def summarize_payload(payload: dict, received_at: str) -> dict:
    return {
        "student_id": payload.get("student_id", ""),
        "total_score": payload.get("total_score", payload.get("final_score", 0)),
        "max_score": payload.get("max_score", 0),
        "total_episodes": payload.get("total_episodes", payload.get("episode_count", 1)),
        "completed_episodes": payload.get("completed_episodes", int(bool(payload.get("completed", False)))),
        "received_at": received_at,
    }


def sanitize_filename_part(value: str, default: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_-]+", "-", value).strip("-")
    return sanitized or default


def make_output_path(output_dir: Path, student_id: str, submitted_at: str) -> Path:
    output_path = output_dir / f"{student_id}_{submitted_at}.json"
    suffix = 1
    while output_path.exists():
        output_path = output_dir / f"{student_id}_{submitted_at}_{suffix}.json"
        suffix += 1
    return output_path


def restore_task_names(payload: dict) -> dict:
    restore_map = payload.get("task_name_restore_map", {})
    if not restore_map:
        return payload

    restored_payload = dict(payload)
    restored_payload.pop("task_name_restore_map", None)

    restored_payload["task_summaries"] = [
        {
            **summary,
            "task_name": restore_map.get(summary.get("task_name"), summary.get("task_name")),
        }
        for summary in payload.get("task_summaries", [])
    ]
    restored_payload["episodes"] = [
        {
            **episode,
            "task_name": restore_map.get(episode.get("task_name"), episode.get("task_name")),
        }
        for episode in payload.get("episodes", [])
    ]
    return restored_payload


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "received_count": len(app.state.records),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/final-project/telemetry")
def list_telemetry() -> dict:
    return {
        "status": "ok",
        "received_count": len(app.state.records),
        "items": app.state.records,
    }


@app.post("/final-project/telemetry", status_code=202)
async def receive_telemetry(request: Request) -> dict:
    payload = await request.json()
    received_dt = datetime.now(timezone.utc)
    received_at = received_dt.isoformat()
    summary = summarize_payload(payload, received_at)
    app.state.records.append(summary)

    output_dir = app.state.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    student_id = sanitize_filename_part(str(payload.get("student_id", "")), "unknown-student")
    submitted_at = sanitize_filename_part(str(payload.get("created_at") or received_at), "unknown-time")
    output_path = make_output_path(output_dir, student_id, submitted_at)

    stored_payload = restore_task_names(payload)
    stored_payload["received_at"] = received_at

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(stored_payload, output_file, ensure_ascii=False, indent=2)

    print(
        f"received student_id={summary['student_id']} "
        f"score={summary['total_score']}/{summary['max_score']} "
        f"episodes={summary['completed_episodes']}/{summary['total_episodes']}"
    )
    return {
        "status": "accepted",
        "received_count": len(app.state.records),
        "saved_to": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastAPI dummy receiver for Final Project telemetry.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
