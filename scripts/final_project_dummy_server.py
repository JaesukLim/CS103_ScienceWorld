import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request


app = FastAPI()
app.state.output_path = None
app.state.records = []


def summarize_payload(payload: dict, received_at: str) -> dict:
    return {
        "student_id": payload.get("student_id", ""),
        "final_score": payload.get("final_score", 0),
        "total_reward": payload.get("total_reward", 0),
        "turn_count": payload.get("turn_count", 0),
        "completed": bool(payload.get("completed", False)),
        "received_at": received_at,
    }


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
    received_at = datetime.now(timezone.utc).isoformat()
    summary = summarize_payload(payload, received_at)
    app.state.records.append(summary)

    output_path: Optional[Path] = app.state.output_path
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(
        f"received student_id={summary['student_id']} "
        f"score={summary['final_score']} "
        f"turns={summary['turn_count']} "
        f"completed={summary['completed']}"
    )
    return {"status": "accepted", "received_count": len(app.state.records)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastAPI dummy receiver for Final Project telemetry.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app.state.output_path = args.output
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
