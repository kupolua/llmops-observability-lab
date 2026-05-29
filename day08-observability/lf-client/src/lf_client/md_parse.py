from pathlib import Path


def read_markdown(path: Path) -> str:
    """Прочитать markdown-файл, вернуть его содержимое."""
    return path.read_text(encoding="utf-8")


def main() -> None:
    base = Path("data/anthropic-cookbook")

    # Найдём все .md из нашего корпуса
    target_paths = [
        base / "README.md",
        base / "CONTRIBUTING.md",
        base / "CLAUDE.md",
        base / "patterns/agents/README.md",
    ]

    for path in target_paths:
        if not path.exists():
            print(f"SKIP (не найден): {path}")
            continue

        content = read_markdown(path)
        print(f"\n=== {path.relative_to(base)} ===")
        print(f"  размер: {len(content)} символов")
        print(f"  первые 200 символов: {content[:200]!r}")


if __name__ == "__main__":
    main()
