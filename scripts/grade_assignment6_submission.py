import json
import sys
from pathlib import Path

from cs103_scienceworld.assignments import grade_assignment6_submission_file


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/grade_assignment6_submission.py <submission.json>")
        return 1

    input_path = Path(sys.argv[1])
    report = grade_assignment6_submission_file(input_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
