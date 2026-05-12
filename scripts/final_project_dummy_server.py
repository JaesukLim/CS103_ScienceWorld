import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse


app = FastAPI()
app.state.output_dir = Path("./telemetry")
app.state.records = []
LEADERBOARD_HTML_PATH = Path(__file__).with_name("final_project_leaderboard.html")
LEADERBOARD_MAX_SCORE = 400


def summarize_payload(payload: dict, received_at: str) -> dict:
    return {
        "student_id": payload.get("student_id", ""),
        "total_score": payload.get("total_score", payload.get("final_score", 0)),
        "max_score": LEADERBOARD_MAX_SCORE,
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


def load_saved_records(output_dir: Path) -> list:
    records = []
    if not output_dir.exists():
        return records

    for output_path in sorted(output_dir.glob("*.json")):
        if not output_path.is_file():
            continue

        try:
            payload = json.loads(output_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping telemetry file {output_path}: {exc}")
            continue

        received_at = str(
            payload.get("received_at")
            or payload.get("created_at")
            or datetime.fromtimestamp(
                output_path.stat().st_mtime,
                timezone.utc,
            ).isoformat()
        )
        records.append(summarize_payload(payload, received_at))
    return records


def numeric_score(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def format_score(value) -> str:
    score = numeric_score(value)
    if score.is_integer():
        return str(int(score))
    return f"{score:.2f}".rstrip("0").rstrip(".")


def build_leaderboard(records: list) -> list:
    grouped = {}
    for record in records:
        student_id = str(record.get("student_id") or "unknown")
        entry = grouped.setdefault(
            student_id,
            {
                "student_id": student_id,
                "submission_count": 0,
                "_score_value": float("-inf"),
                "score": 0,
                "max_score": LEADERBOARD_MAX_SCORE,
                "completed_episodes": 0,
                "total_episodes": 0,
                "received_at": "",
            },
        )
        entry["submission_count"] += 1

        score_value = numeric_score(record.get("total_score", 0))
        received_at = str(record.get("received_at") or "")
        is_better_score = score_value > entry["_score_value"]
        is_earlier_tie = (
            score_value == entry["_score_value"]
            and received_at < entry["received_at"]
        )
        if is_better_score or is_earlier_tie:
            entry.update(
                {
                    "_score_value": score_value,
                    "score": record.get("total_score", 0),
                    "max_score": LEADERBOARD_MAX_SCORE,
                    "completed_episodes": record.get("completed_episodes", 0),
                    "total_episodes": record.get("total_episodes", 0),
                    "received_at": received_at,
                }
            )

    leaderboard = sorted(
        grouped.values(),
        key=lambda item: (
            -item["_score_value"],
            item["received_at"],
            item["student_id"],
        ),
    )
    previous_score = None
    previous_rank = 0
    for index, entry in enumerate(leaderboard, start=1):
        if entry["_score_value"] != previous_score:
            previous_rank = index
            previous_score = entry["_score_value"]
        entry["rank"] = previous_rank
        entry["score_display"] = (
            f"{format_score(entry['score'])} / {format_score(LEADERBOARD_MAX_SCORE)}"
        )
        del entry["_score_value"]
    return leaderboard


@app.on_event("startup")
def load_saved_telemetry() -> None:
    app.state.records = load_saved_records(app.state.output_dir)
    if app.state.records:
        print(
            f"loaded {len(app.state.records)} telemetry records "
            f"from {app.state.output_dir}"
        )


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


@app.get("/", response_class=HTMLResponse)
@app.get("/final-project/leaderboard-page", response_class=HTMLResponse)
def leaderboard_page() -> HTMLResponse:
    return HTMLResponse(LEADERBOARD_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/final-project/leaderboard")
def list_leaderboard() -> dict:
    return {
        "status": "ok",
        "received_count": len(app.state.records),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "items": build_leaderboard(app.state.records),
    }


@app.delete("/final-project/submissions")
def delete_submissions() -> dict:
    deleted_files = 0
    output_dir = app.state.output_dir
    if output_dir.exists():
        for output_path in output_dir.glob("*.json"):
            if output_path.is_file():
                output_path.unlink()
                deleted_files += 1

    deleted_records = len(app.state.records)
    app.state.records.clear()
    return {
        "status": "deleted",
        "deleted_records": deleted_records,
        "deleted_files": deleted_files,
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
