"""Generate synthetic test files for load testing.

This module provides utilities to generate large datasets of test files
for stress testing the markit batch processing pipeline.

Example usage:
    ```bash
    # Generate 1000 test files
    python -m tests.fixtures.heavy_load.generate_dataset -o ./test_data -n 1000

    # Generate with custom size range
    python -m tests.fixtures.heavy_load.generate_dataset -o ./test_data -n 500 \
        --min-size 10 --max-size 100
    ```
"""

import argparse
import random
from pathlib import Path

# Lorem ipsum paragraphs for content generation
LOREM_PARAGRAPHS = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, "
    "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore "
    "eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, "
    "sunt in culpa qui officia deserunt mollit anim id est laborum.",
    "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium "
    "doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore "
    "veritatis et quasi architecto beatae vitae dicta sunt explicabo.",
    "Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, "
    "sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. "
    "Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet.",
    "At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis "
    "praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias "
    "excepturi sint occaecati cupiditate non provident.",
    "Similique sunt in culpa qui officia deserunt mollitia animi, id est laborum "
    "et dolorum fuga. Et harum quidem rerum facilis est et expedita distinctio. "
    "Nam libero tempore, cum soluta nobis est eligendi optio cumque nihil impedit.",
    "Temporibus autem quibusdam et aut officiis debitis aut rerum necessitatibus "
    "saepe eveniet ut et voluptates repudiandae sint et molestiae non recusandae. "
    "Itaque earum rerum hic tenetur a sapiente delectus.",
    "Ut aut reiciendis voluptatibus maiores alias consequatur aut perferendis "
    "doloribus asperiores repellat. Quo usque tandem abutere, Catilina, patientia "
    "nostra? Quam diu etiam furor iste tuus nos eludet?",
]

# Technical terms for more realistic document content
TECHNICAL_TERMS = [
    "API",
    "microservices",
    "cloud computing",
    "machine learning",
    "neural networks",
    "data pipeline",
    "distributed systems",
    "containerization",
    "DevOps",
    "CI/CD",
    "infrastructure",
    "scalability",
    "reliability",
    "observability",
    "monitoring",
]


def generate_markdown_file(
    output_path: Path,
    size_kb: int = 10,
    include_code: bool = True,
    include_tables: bool = True,
) -> None:
    """Generate a synthetic Markdown file.

    Args:
        output_path: Where to write the file
        size_kb: Approximate size in KB
        include_code: Include code blocks
        include_tables: Include tables
    """
    content_lines = []
    target_chars = size_kb * 1024
    current_chars = 0
    section_num = 1

    # Title
    title = f"Test Document {output_path.stem}"
    content_lines.append(f"# {title}\n\n")
    current_chars += len(title) + 4

    while current_chars < target_chars:
        # Add section header
        term = random.choice(TECHNICAL_TERMS)
        header = f"## Section {section_num}: {term}\n\n"
        content_lines.append(header)
        current_chars += len(header)

        # Add 2-4 paragraphs
        for _ in range(random.randint(2, 4)):
            para = random.choice(LOREM_PARAGRAPHS)
            content_lines.append(f"{para}\n\n")
            current_chars += len(para) + 2

        # Occasionally add a list (30% chance)
        if random.random() < 0.3:
            list_items = [
                f"- {random.choice(TECHNICAL_TERMS)}: {random.choice(LOREM_PARAGRAPHS)[:50]}..."
                for _ in range(random.randint(3, 6))
            ]
            list_text = "\n".join(list_items) + "\n\n"
            content_lines.append(list_text)
            current_chars += len(list_text)

        # Occasionally add a code block (20% chance)
        if include_code and random.random() < 0.2:
            code_block = """```python
def process_data(items):
    results = []
    for item in items:
        result = transform(item)
        results.append(result)
    return results
```\n\n"""
            content_lines.append(code_block)
            current_chars += len(code_block)

        # Occasionally add a table (15% chance)
        if include_tables and random.random() < 0.15:
            table = """| Column A | Column B | Column C |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |
| Value 7  | Value 8  | Value 9  |

"""
            content_lines.append(table)
            current_chars += len(table)

        section_num += 1

    output_path.write_text("".join(content_lines), encoding="utf-8")


def generate_text_file(output_path: Path, size_kb: int = 10) -> None:
    """Generate a synthetic plain text file.

    Args:
        output_path: Where to write the file
        size_kb: Approximate size in KB
    """
    content_lines = []
    target_chars = size_kb * 1024
    current_chars = 0

    while current_chars < target_chars:
        para = random.choice(LOREM_PARAGRAPHS)
        content_lines.append(f"{para}\n\n")
        current_chars += len(para) + 2

    output_path.write_text("".join(content_lines), encoding="utf-8")


def generate_dataset(
    output_dir: Path,
    count: int,
    prefix: str = "doc",
    min_size_kb: int = 5,
    max_size_kb: int = 50,
    format_type: str = "markdown",
) -> list[Path]:
    """Generate a dataset of synthetic files.

    Args:
        output_dir: Directory to write files
        count: Number of files to generate
        prefix: Filename prefix
        min_size_kb: Minimum file size in KB
        max_size_kb: Maximum file size in KB
        format_type: File format ("markdown" or "text")

    Returns:
        List of generated file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []

    extension = ".md" if format_type == "markdown" else ".txt"
    generator = generate_markdown_file if format_type == "markdown" else generate_text_file

    for i in range(count):
        filename = f"{prefix}_{i:05d}{extension}"
        file_path = output_dir / filename
        size_kb = random.randint(min_size_kb, max_size_kb)

        if format_type == "markdown":
            generator(file_path, size_kb, include_code=True, include_tables=True)
        else:
            generator(file_path, size_kb)

        files.append(file_path)

        # Progress indicator every 100 files
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{count} files...")

    return files


# =============================================================================
# SPEC.md Required Preset Datasets
# =============================================================================


def generate_1k_mix(base_dir: Path) -> list[Path]:
    """Generate 1k_mix dataset: 1000 files with mixed content types.

    As specified in docs/SPEC.md section 2.2.A:
    - Generates 1000 files alternating between .md and .txt
    - Random word count (100-1000 words for md, 50-500 for txt)
    - Predictable filenames: doc_0000.md, doc_0001.txt, etc.

    Args:
        base_dir: Base directory for the dataset

    Returns:
        List of generated file paths
    """
    dataset_dir = base_dir / "1k_mix"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    files = []

    print(f"Generating 1k_mix dataset in {dataset_dir}")

    for i in range(1000):
        if i % 2 == 0:
            # Markdown file
            filename = f"doc_{i:04d}.md"
            filepath = dataset_dir / filename
            size_kb = random.randint(1, 10)  # 100-1000 words ≈ 1-10 KB
            generate_markdown_file(filepath, size_kb)
        else:
            # Text file
            filename = f"doc_{i:04d}.txt"
            filepath = dataset_dir / filename
            size_kb = random.randint(1, 5)  # 50-500 words ≈ 0.5-5 KB
            generate_text_file(filepath, size_kb)

        files.append(filepath)

        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/1000 files...")

    print(f"  Completed: {len(files)} files")
    return files


def generate_10k_text(base_dir: Path) -> list[Path]:
    """Generate 10k_text dataset: 10000 plain text files.

    As specified in docs/SPEC.md section 2.2.A:
    - Generates 10000 plain text files
    - Larger content for token consumption (200-2000 words)
    - Predictable filenames: doc_00000.txt to doc_09999.txt

    Args:
        base_dir: Base directory for the dataset

    Returns:
        List of generated file paths
    """
    dataset_dir = base_dir / "10k_text"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    files = []

    print(f"Generating 10k_text dataset in {dataset_dir}")

    for i in range(10000):
        filename = f"doc_{i:05d}.txt"
        filepath = dataset_dir / filename
        size_kb = random.randint(2, 20)  # 200-2000 words ≈ 2-20 KB
        generate_text_file(filepath, size_kb)
        files.append(filepath)

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/10000 files...")

    print(f"  Completed: {len(files)} files")
    return files


def generate_deep_nested(base_dir: Path) -> list[Path]:
    """Generate deep_nested dataset: deeply nested directory structure.

    As specified in docs/SPEC.md section 2.2.A:
    - Creates 10 levels of nested directories
    - 5-20 files at each level
    - Tests path handling and recursive processing

    Args:
        base_dir: Base directory for the dataset

    Returns:
        List of generated file paths
    """
    dataset_dir = base_dir / "deep_nested"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    files = []

    print(f"Generating deep_nested dataset in {dataset_dir}")

    current_dir = dataset_dir
    for depth in range(10):
        current_dir = current_dir / f"level_{depth}"
        current_dir.mkdir(exist_ok=True)

        # Add 5-20 files at each level
        num_files = random.randint(5, 20)
        for i in range(num_files):
            filename = f"doc_d{depth}_{i:03d}.txt"
            filepath = current_dir / filename
            size_kb = random.randint(1, 5)
            generate_text_file(filepath, size_kb)
            files.append(filepath)

        print(f"  Level {depth}: {num_files} files")

    print(f"  Completed: {len(files)} files across 10 levels")
    return files


def generate_all_presets(base_dir: Path) -> dict[str, list[Path]]:
    """Generate all preset datasets required by SPEC.md.

    Args:
        base_dir: Base directory for all datasets

    Returns:
        Dictionary mapping dataset name to list of generated files
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating all preset datasets in {base_dir}\n")

    results = {
        "1k_mix": generate_1k_mix(base_dir),
        "10k_text": generate_10k_text(base_dir),
        "deep_nested": generate_deep_nested(base_dir),
    }

    total_files = sum(len(files) for files in results.values())
    total_size = sum(f.stat().st_size for files in results.values() for f in files)

    print(f"\n{'=' * 50}")
    print(f"Total: {total_files} files, {total_size / 1024 / 1024:.2f} MB")

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic test dataset for load testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=1000,
        help="Number of files to generate",
    )
    parser.add_argument(
        "--prefix",
        default="doc",
        help="Filename prefix",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=5,
        help="Minimum file size in KB",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=50,
        help="Maximum file size in KB",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "text"],
        default="markdown",
        help="File format to generate",
    )
    parser.add_argument(
        "--preset",
        choices=["1k_mix", "10k_text", "deep_nested", "all"],
        help="Generate preset dataset (overrides other options)",
    )

    args = parser.parse_args()

    # Handle preset datasets
    if args.preset:
        if args.preset == "all":
            generate_all_presets(args.output)
        elif args.preset == "1k_mix":
            generate_1k_mix(args.output)
        elif args.preset == "10k_text":
            generate_10k_text(args.output)
        elif args.preset == "deep_nested":
            generate_deep_nested(args.output)
        return

    # Custom dataset generation
    print(f"Generating {args.count} {args.format} files...")
    print(f"Size range: {args.min_size}-{args.max_size} KB")
    print(f"Output directory: {args.output}")

    files = generate_dataset(
        output_dir=args.output,
        count=args.count,
        prefix=args.prefix,
        min_size_kb=args.min_size,
        max_size_kb=args.max_size,
        format_type=args.format,
    )

    total_size = sum(f.stat().st_size for f in files)
    print(f"\nGenerated {len(files)} files")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
