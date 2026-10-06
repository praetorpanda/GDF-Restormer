#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
audit_repo_imports.py
=====================

Read-only repository-wide import dependency audit for the GDF-Restormer
open-source tree.

The repository intentionally preserves historical and experimental code.
This script helps identify:

1. Which Python files import:
   - config
   - utils
   - loss
   - gdf_loss
   - models
   - engines
   - data

2. Which exact symbols are imported from those modules.

3. Which local files depend on which other local modules.

4. Which files belong to likely categories:
   - main release
   - paper / ablation
   - historical experiments
   - reference architectures
   - support utilities

5. Which top-level duplicate names exist:
   - config.py vs config/
   - utils.py vs utils/
   - loss.py vs loss/
   - gdf_loss.py

6. Which local Python files appear to have no incoming local imports.

Important:
- This is a STATIC audit.
- It does NOT import project modules.
- It does NOT modify, delete, rename, or execute training code.
- "Unreferenced" does not mean "safe to delete": runnable entry scripts are
  often intentionally imported by nobody.

Usage
-----

From repository root:

    python scripts/audit_repo_imports.py

Save a text report:

    python scripts/audit_repo_imports.py > docs/import_dependency_audit.txt

Also save CSV files:

    python scripts/audit_repo_imports.py --csv-dir docs/import_audit_csv

Scan a specific repository:

    python scripts/audit_repo_imports.py --root /path/to/open_source

Include __pycache__ / hidden dirs (not recommended):

    python scripts/audit_repo_imports.py --include-hidden
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


FOCUS_ROOTS = {
    "config",
    "utils",
    "loss",
    "gdf_loss",
    "models",
    "engines",
    "data",
}

DEFAULT_EXCLUDED_DIRS = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "checkpoints",
    "checkpoints_smoke",
    "results",
}

MAIN_RELEASE_FILES = {
    "train.py",
    "config.py",
    "gdf_loss.py",
    "utils.py",
    "models/Res_Strict.py",
    "models/Resbase.py",
    "engines/engine_strict_global_degfield.py",
    "data/dataset_GlobalMeta_FullCanvas.py",
    "scripts/check_environment.py",
    "scripts/download_dataset.sh",
    "scripts/smoke_test.sh",
    "scripts/test_release_local.sh",
}

ENTRY_NAME_HINTS = (
    "train",
    "run",
    "eval",
    "test",
    "check",
    "audit",
    "verify",
    "setup",
    "smoke",
)


@dataclass(frozen=True)
class ImportRecord:
    source_file: str
    lineno: int
    kind: str
    module: str
    root: str
    imported_name: str
    alias: str
    statement: str
    is_focus: bool
    local_target: str


@dataclass
class FileInfo:
    path: str
    category: str
    is_entry_like: bool
    import_count: int = 0
    focus_import_count: int = 0


def parse_args():
    p = argparse.ArgumentParser(
        description="Static repository-wide Python import dependency audit."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root. Default: auto-detect.",
    )
    p.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Optional directory for CSV/JSON audit outputs.",
    )
    p.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include normally excluded hidden/cache/output directories.",
    )
    p.add_argument(
        "--all-imports",
        action="store_true",
        help="Print every import, not only focus/local imports.",
    )
    return p.parse_args()


def detect_root(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()

    here = Path(__file__).resolve()

    if here.parent.name == "scripts":
        candidate = here.parent.parent
        if (candidate / "train.py").is_file():
            return candidate

    if (here.parent / "train.py").is_file():
        return here.parent

    cwd = Path.cwd().resolve()
    if (cwd / "train.py").is_file():
        return cwd

    raise RuntimeError(
        "Could not auto-detect repository root. "
        "Run from repository root or pass --root."
    )


def header(title: str):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def should_skip(path: Path, root: Path, include_hidden: bool) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True

    if include_hidden:
        return False

    for part in parts[:-1]:
        if part in DEFAULT_EXCLUDED_DIRS:
            return True
        if part.startswith("."):
            return True

    return False


def discover_python_files(root: Path, include_hidden: bool) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if not path.is_file():
            continue
        if should_skip(path, root, include_hidden):
            continue
        files.append(path)
    return sorted(files, key=lambda p: str(p.relative_to(root)))


def module_name_for_file(path: Path, root: Path) -> str:
    relative = path.relative_to(root)

    if relative.name == "__init__.py":
        parts = relative.parent.parts
    else:
        parts = relative.with_suffix("").parts

    return ".".join(parts)


def build_local_module_index(
    py_files: Iterable[Path],
    root: Path,
) -> dict[str, str]:
    index: dict[str, str] = {}

    for path in py_files:
        module = module_name_for_file(path, root)
        if module:
            index[module] = rel(path, root)

    return index


def resolve_relative_module(
    source_module: str,
    level: int,
    imported_module: str,
) -> str:
    if level <= 0:
        return imported_module

    source_parts = source_module.split(".")
    # If source is a normal file module, relative imports are based on package.
    package_parts = source_parts[:-1]

    trim = level - 1
    if trim > len(package_parts):
        base = []
    else:
        base = package_parts[: len(package_parts) - trim]

    if imported_module:
        base.extend(imported_module.split("."))

    return ".".join(x for x in base if x)


def find_local_target(
    module: str,
    imported_name: str,
    local_index: dict[str, str],
) -> str:
    candidates = []

    if module:
        if imported_name and imported_name != "*":
            candidates.append(f"{module}.{imported_name}")
        candidates.append(module)

        # Parent-module fallback.
        parts = module.split(".")
        for i in range(len(parts) - 1, 0, -1):
            candidates.append(".".join(parts[:i]))

    for candidate in candidates:
        if candidate in local_index:
            return local_index[candidate]

    return ""


def render_import(node: ast.AST) -> str:
    if isinstance(node, ast.Import):
        items = []
        for a in node.names:
            s = a.name
            if a.asname:
                s += f" as {a.asname}"
            items.append(s)
        return "import " + ", ".join(items)

    if isinstance(node, ast.ImportFrom):
        names = []
        for a in node.names:
            s = a.name
            if a.asname:
                s += f" as {a.asname}"
            names.append(s)
        module = "." * node.level + (node.module or "")
        return f"from {module} import " + ", ".join(names)

    return ""


def parse_file_imports(
    path: Path,
    root: Path,
    local_index: dict[str, str],
) -> tuple[list[ImportRecord], Optional[str]]:
    source_rel = rel(path, root)
    source_module = module_name_for_file(path, root)

    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"

    records: list[ImportRecord] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            statement = render_import(node)

            for alias in node.names:
                module = alias.name
                root_name = module.split(".", 1)[0]
                local_target = find_local_target(
                    module, "", local_index
                )

                records.append(
                    ImportRecord(
                        source_file=source_rel,
                        lineno=node.lineno,
                        kind="import",
                        module=module,
                        root=root_name,
                        imported_name="",
                        alias=alias.asname or "",
                        statement=statement,
                        is_focus=root_name in FOCUS_ROOTS,
                        local_target=local_target,
                    )
                )

        elif isinstance(node, ast.ImportFrom):
            statement = render_import(node)
            absolute_module = resolve_relative_module(
                source_module,
                node.level,
                node.module or "",
            )
            root_name = (
                absolute_module.split(".", 1)[0]
                if absolute_module
                else ""
            )

            for alias in node.names:
                local_target = find_local_target(
                    absolute_module,
                    alias.name,
                    local_index,
                )

                records.append(
                    ImportRecord(
                        source_file=source_rel,
                        lineno=node.lineno,
                        kind="from",
                        module=absolute_module,
                        root=root_name,
                        imported_name=alias.name,
                        alias=alias.asname or "",
                        statement=statement,
                        is_focus=root_name in FOCUS_ROOTS,
                        local_target=local_target,
                    )
                )

    records.sort(
        key=lambda r: (
            r.source_file,
            r.lineno,
            r.statement,
            r.imported_name,
        )
    )
    return records, None


def categorize_file(path_rel: str) -> str:
    p = Path(path_rel)
    name = p.name.lower()
    parts = [x.lower() for x in p.parts]

    if path_rel in MAIN_RELEASE_FILES:
        return "main_release"

    if p.parts and p.parts[0] == "models":
        experimental_tokens = (
            "res_",
            "prior",
            "psf",
            "global",
            "wavelet",
            "moe",
            "restormer",
        )
        if any(tok in name for tok in experimental_tokens):
            return "research_model"
        return "reference_model"

    if p.parts and p.parts[0] == "engines":
        return "research_engine"

    if p.parts and p.parts[0] == "data":
        return "dataset"

    if p.parts and p.parts[0] == "loss":
        return "legacy_loss"

    if p.parts and p.parts[0] == "utils":
        return "legacy_utils"

    if p.parts and p.parts[0] == "config":
        return "legacy_config"

    if p.parts and p.parts[0] == "tools":
        return "tool"

    if p.parts and p.parts[0] == "scripts":
        return "script"

    if name.startswith("train_"):
        if any(tok in name for tok in ("strict", "degfield", "global", "psf")):
            return "research_entry"
        return "historical_entry"

    return "other"


def entry_like(path_rel: str) -> bool:
    stem = Path(path_rel).stem.lower()
    return any(stem.startswith(prefix) for prefix in ENTRY_NAME_HINTS)


def duplicate_top_level_report(root: Path):
    header("TOP-LEVEL MODULE / PACKAGE DUPLICATES")

    any_dup = False
    names = sorted({
        p.stem
        for p in root.glob("*.py")
        if p.is_file()
    })

    for name in names:
        module_path = root / f"{name}.py"
        package_init = root / name / "__init__.py"

        if module_path.is_file() and package_init.is_file():
            any_dup = True
            print(
                f"[DUPLICATE] {name}: "
                f"{module_path.name}  <->  {name}/__init__.py"
            )

    if not any_dup:
        print("No top-level module/package duplicates detected.")


def print_focus_dependencies(
    records: list[ImportRecord],
    all_imports: bool,
):
    header("PER-FILE IMPORT DEPENDENCIES")

    by_file: dict[str, list[ImportRecord]] = defaultdict(list)
    for r in records:
        if all_imports or r.is_focus or r.local_target:
            by_file[r.source_file].append(r)

    if not by_file:
        print("No matching imports found.")
        return

    for source in sorted(by_file):
        print(f"\n[{source}]")
        seen = set()

        for r in sorted(
            by_file[source],
            key=lambda x: (x.lineno, x.statement, x.imported_name),
        ):
            key = (
                r.lineno,
                r.statement,
                r.imported_name,
                r.local_target,
            )
            if key in seen:
                continue
            seen.add(key)

            suffix = ""
            if r.local_target:
                suffix += f"  -> local:{r.local_target}"
            if r.is_focus:
                suffix += "  [FOCUS]"

            if r.kind == "from" and r.imported_name:
                detail = (
                    f"L{r.lineno:<4} {r.module} :: "
                    f"{r.imported_name}"
                )
                if r.alias:
                    detail += f" as {r.alias}"
            else:
                detail = f"L{r.lineno:<4} {r.module}"
                if r.alias:
                    detail += f" as {r.alias}"

            print(f"  {detail}{suffix}")


def print_focus_root_summary(records: list[ImportRecord]):
    header("FOCUS IMPORT SUMMARY")

    root_to_sources: dict[str, set[str]] = defaultdict(set)
    root_to_symbols: dict[str, Counter] = defaultdict(Counter)

    for r in records:
        if not r.is_focus:
            continue
        root_to_sources[r.root].add(r.source_file)

        symbol = r.imported_name or "<module>"
        root_to_symbols[r.root][symbol] += 1

    for root_name in sorted(FOCUS_ROOTS):
        sources = sorted(root_to_sources[root_name])
        print(f"\n[{root_name}]")
        print(f"  importing files : {len(sources)}")
        for src in sources:
            print(f"    - {src}")

        if root_to_symbols[root_name]:
            print("  imported names  :")
            for symbol, count in root_to_symbols[root_name].most_common():
                print(f"    - {symbol}: {count}")


def print_local_dependency_graph(
    records: list[ImportRecord],
):
    header("LOCAL DEPENDENCY GRAPH")

    edges: dict[str, set[str]] = defaultdict(set)

    for r in records:
        if r.local_target and r.local_target != r.source_file:
            edges[r.source_file].add(r.local_target)

    if not edges:
        print("No local dependency edges detected.")
        return

    for source in sorted(edges):
        print(f"\n{source}")
        for target in sorted(edges[source]):
            print(f"  -> {target}")


def print_reverse_dependencies(
    records: list[ImportRecord],
):
    header("REVERSE LOCAL DEPENDENCIES")

    incoming: dict[str, set[str]] = defaultdict(set)

    for r in records:
        if r.local_target and r.local_target != r.source_file:
            incoming[r.local_target].add(r.source_file)

    for target in sorted(incoming):
        print(f"\n{target}")
        for source in sorted(incoming[target]):
            print(f"  <- {source}")


def print_file_categories(
    file_infos: list[FileInfo],
):
    header("FILE CATEGORY SUMMARY")

    grouped: dict[str, list[FileInfo]] = defaultdict(list)
    for info in file_infos:
        grouped[info.category].append(info)

    order = [
        "main_release",
        "research_entry",
        "research_model",
        "research_engine",
        "dataset",
        "reference_model",
        "legacy_config",
        "legacy_utils",
        "legacy_loss",
        "script",
        "tool",
        "historical_entry",
        "other",
    ]

    for category in order:
        items = grouped.get(category, [])
        if not items:
            continue

        print(f"\n[{category}] ({len(items)})")
        for info in sorted(items, key=lambda x: x.path):
            entry = " entry-like" if info.is_entry_like else ""
            print(
                f"  - {info.path}"
                f" | imports={info.import_count}"
                f" | focus={info.focus_import_count}"
                f"{entry}"
            )


def print_unreferenced_local_files(
    py_files: list[Path],
    root: Path,
    records: list[ImportRecord],
):
    header("FILES WITH NO INCOMING LOCAL IMPORTS")

    all_files = {rel(p, root) for p in py_files}
    incoming = {
        r.local_target
        for r in records
        if r.local_target and r.local_target != r.source_file
    }

    unreferenced = sorted(all_files - incoming)

    print(
        "These files are not statically imported by another local Python file.\n"
        "This is informational only: entry scripts are normally unreferenced.\n"
    )

    for path_rel in unreferenced:
        kind = "ENTRY-LIKE" if entry_like(path_rel) else "unreferenced"
        print(f"  [{kind:<11}] {path_rel}")


def write_csv_outputs(
    out_dir: Path,
    file_infos: list[FileInfo],
    records: list[ImportRecord],
    parse_errors: dict[str, str],
):
    out_dir.mkdir(parents=True, exist_ok=True)

    imports_csv = out_dir / "imports.csv"
    with imports_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "source_file",
            "lineno",
            "kind",
            "module",
            "root",
            "imported_name",
            "alias",
            "statement",
            "is_focus",
            "local_target",
        ])
        for r in records:
            w.writerow([
                r.source_file,
                r.lineno,
                r.kind,
                r.module,
                r.root,
                r.imported_name,
                r.alias,
                r.statement,
                int(r.is_focus),
                r.local_target,
            ])

    files_csv = out_dir / "files.csv"
    with files_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "path",
            "category",
            "is_entry_like",
            "import_count",
            "focus_import_count",
        ])
        for info in file_infos:
            w.writerow([
                info.path,
                info.category,
                int(info.is_entry_like),
                info.import_count,
                info.focus_import_count,
            ])

    deps_csv = out_dir / "local_dependencies.csv"
    seen = set()
    with deps_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_file", "target_file"])
        for r in records:
            if not r.local_target or r.local_target == r.source_file:
                continue
            edge = (r.source_file, r.local_target)
            if edge in seen:
                continue
            seen.add(edge)
            w.writerow(edge)

    errors_json = out_dir / "parse_errors.json"
    errors_json.write_text(
        json.dumps(parse_errors, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"CSV/JSON outputs written to: {out_dir}")
    print(f"  - {imports_csv.name}")
    print(f"  - {files_csv.name}")
    print(f"  - {deps_csv.name}")
    print(f"  - {errors_json.name}")


def main() -> int:
    args = parse_args()
    root = detect_root(args.root)

    py_files = discover_python_files(root, args.include_hidden)
    local_index = build_local_module_index(py_files, root)

    records: list[ImportRecord] = []
    parse_errors: dict[str, str] = {}

    for path in py_files:
        recs, err = parse_file_imports(
            path,
            root,
            local_index,
        )
        records.extend(recs)
        if err:
            parse_errors[rel(path, root)] = err

    by_source: dict[str, list[ImportRecord]] = defaultdict(list)
    for r in records:
        by_source[r.source_file].append(r)

    file_infos = []
    for path in py_files:
        path_rel = rel(path, root)
        recs = by_source[path_rel]
        file_infos.append(
            FileInfo(
                path=path_rel,
                category=categorize_file(path_rel),
                is_entry_like=entry_like(path_rel),
                import_count=len(recs),
                focus_import_count=sum(r.is_focus for r in recs),
            )
        )

    print("=" * 100)
    print("GDF-Restormer repository import dependency audit")
    print("=" * 100)
    print(f"Repository root      : {root}")
    print(f"Python files scanned : {len(py_files)}")
    print(f"Import records       : {len(records)}")
    print(f"Parse errors         : {len(parse_errors)}")
    print(f"Focus roots          : {', '.join(sorted(FOCUS_ROOTS))}")

    if parse_errors:
        header("PARSE ERRORS")
        for path_rel, err in sorted(parse_errors.items()):
            print(f"[ERROR] {path_rel}: {err}")

    duplicate_top_level_report(root)
    print_file_categories(file_infos)
    print_focus_root_summary(records)
    print_focus_dependencies(records, args.all_imports)
    print_local_dependency_graph(records)
    print_reverse_dependencies(records)
    print_unreferenced_local_files(py_files, root, records)

    header("MAIN RELEASE PATH QUICK VIEW")
    main_sources = {
        "train.py",
        "engines/engine_strict_global_degfield.py",
        "models/Res_Strict.py",
        "models/Resbase.py",
        "data/dataset_GlobalMeta_FullCanvas.py",
        "gdf_loss.py",
    }

    for source in sorted(main_sources):
        print(f"\n[{source}]")
        recs = [
            r for r in records
            if r.source_file == source and (r.is_focus or r.local_target)
        ]
        if not recs:
            print("  no focus/local imports detected")
            continue

        shown = set()
        for r in recs:
            key = (r.statement, r.local_target)
            if key in shown:
                continue
            shown.add(key)
            target = f" -> {r.local_target}" if r.local_target else ""
            print(f"  L{r.lineno:<4} {r.statement}{target}")

    header("INTERPRETATION NOTES")
    print(
        "- A duplicate top-level name is not automatically an error.\n"
        "- A historical file with no incoming imports is not automatically removable.\n"
        "- train/run/eval/test scripts are commonly entry points and therefore unreferenced.\n"
        "- MAIN RELEASE files should be evaluated first when deciding whether a duplicate\n"
        "  module name can affect the default training path.\n"
        "- Historical architectures can be retained even if unused by the main path.\n"
        "- The strongest cleanup candidates are files that are both unreferenced and not\n"
        "  obvious entry points, but they still require manual review."
    )

    if args.csv_dir is not None:
        out_dir = args.csv_dir
        if not out_dir.is_absolute():
            out_dir = root / out_dir
        write_csv_outputs(
            out_dir.resolve(),
            file_infos,
            records,
            parse_errors,
        )

    header("AUDIT COMPLETE")
    if parse_errors:
        print(
            f"[WARN] Completed with {len(parse_errors)} parse error(s). "
            "Review them before relying on dependency conclusions."
        )
    else:
        print("[PASS] Static import audit completed without parse errors.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
