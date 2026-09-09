import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from graph import graph  # noqa: E402  (import after load_dotenv so the LLM client sees env vars)


def run(lease_path: str) -> dict:
    lease_text = Path(lease_path).read_text()
    initial_state = {
        "lease_text": lease_text,
        "extracted": {},
        "tool_trace": [],
        "confidence": 0.0,
        "flags": [],
        "decision": "",
        "audit_id": "",
    }
    return graph.invoke(initial_state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lease-processing agent on a lease file.")
    parser.add_argument("lease_path", help="Path to a lease .txt file")
    args = parser.parse_args()

    final_state = run(args.lease_path)

    print(json.dumps({
        "decision": final_state["decision"],
        "confidence": final_state["confidence"],
        "extracted": final_state["extracted"],
        "flags": final_state["flags"],
        "audit_id": final_state["audit_id"],
    }, indent=2))


if __name__ == "__main__":
    main()
