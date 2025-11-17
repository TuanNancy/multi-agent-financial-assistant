from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_json_to_jsonl(src: Path, dst: Path) -> int:
    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of news objects.")

    dst.parent.mkdir(parents=True, exist_ok=True)

    with dst.open("w", encoding="utf-8") as f:
        for item in data:
            if "embedding_id" not in item:
                item["embedding_id"] = f"emb_{item.get('id', 'unknown')}"
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert news JSON array into newline-delimited JSONL."
    )
    parser.add_argument(
        "--input",
        default="data/news_raw/financial_news_tsla.json",
        help="Path to source JSON array file.",
    )
    parser.add_argument(
        "--output",
        default="data/news_index.jsonl",
        help="Destination JSONL file path.",
    )
    args = parser.parse_args()

    count = convert_json_to_jsonl(Path(args.input), Path(args.output))
    print(f"✅ Đã tạo lại JSONL với {count} bản ghi tại {args.output}")


if __name__ == "__main__":
    main()

