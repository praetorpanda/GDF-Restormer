#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Read-only import-resolution audit for the GDF-Restormer repository.

Checks:
- config.py vs config/
- utils.py vs utils/
- loss.py vs loss/ vs gdf_loss.py
- actual Python resolution
- required mainline symbols
- relevant imports in train.py and the main engine
- whether the main engine uses gdf_loss instead of legacy loss

Usage:
    python scripts/check_import_resolution.py
    python scripts/check_import_resolution.py --strict
    python scripts/check_import_resolution.py --root /path/to/open_source
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

AMBIGUOUS_NAMES = ("config", "utils", "loss", "gdf_loss")
MAIN_FILES = (
    "train.py",
    "engines/engine_strict_global_degfield.py",
)
RELEVANT_IMPORT_ROOTS = {
    "config", "utils", "loss", "gdf_loss",
    "models", "engines", "data",
}


@dataclass
class Finding:
    level: str
    message: str


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def detect_root(explicit):
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
        "Cannot auto-detect repository root. "
        "Run from repo root or pass --root."
    )


def rel(path, root):
    if path is None:
        return "None"
    try:
        p = Path(path).resolve()
        return str(p.relative_to(root))
    except Exception:
        return str(path)


def header(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def candidates(root, name):
    return [
        ("module", root / f"{name}.py"),
        ("package", root / name / "__init__.py"),
        ("directory", root / name),
    ]


def inspect_resolution(root, name):
    findings = []
    print(f"\n[{name}]")

    for kind, path in candidates(root, name):
        print(
            f"  repo {kind:<9}: "
            f"{'YES' if path.exists() else 'no ':<3}  "
            f"{rel(path, root)}"
        )

    module_exists = (root / f"{name}.py").is_file()
    package_exists = (root / name / "__init__.py").is_file()

    if module_exists and package_exists:
        findings.append(Finding(
            "WARN",
            f"Both {name}.py and {name}/__init__.py exist."
        ))

    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:
        print(f"  find_spec       : ERROR: {exc}")
        findings.append(Finding("FAIL", f"find_spec({name}) failed: {exc}"))
        return findings

    if spec is None:
        print("  find_spec       : NOT FOUND")
        findings.append(Finding("WARN", f"{name!r} is not resolvable."))
        return findings

    print(f"  resolved origin : {rel(spec.origin, root)}")
    if spec.submodule_search_locations is None:
        print("  resolved type   : module")
    else:
        print("  resolved type   : package")
        print(
            "  package paths   : "
            + ", ".join(rel(p, root) for p in spec.submodule_search_locations)
        )

    return findings


def import_symbol(root, module_name, symbols):
    findings = []
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        print(f"  import test     : FAIL: {type(exc).__name__}: {exc}")
        findings.append(Finding(
            "FAIL",
            f"Cannot import {module_name!r}: {type(exc).__name__}: {exc}"
        ))
        return findings

    print(f"  imported file   : {rel(getattr(module, '__file__', None), root)}")

    for symbol in symbols:
        ok = hasattr(module, symbol)
        print(f"  symbol {symbol:<18}: {'PASS' if ok else 'MISSING'}")
        if not ok:
            findings.append(Finding(
                "FAIL",
                f"{module_name!r} lacks required symbol {symbol!r}."
            ))
            continue

        obj = getattr(module, symbol)
        print(f"    defined module: {getattr(obj, '__module__', None)}")
        try:
            src = inspect.getsourcefile(obj)
        except Exception:
            src = None
        if src:
            print(f"    source file   : {rel(src, root)}")

    return findings


def scan_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_name = alias.name.split(".", 1)[0]
                if root_name in RELEVANT_IMPORT_ROOTS:
                    s = f"import {alias.name}"
                    if alias.asname:
                        s += f" as {alias.asname}"
                    found.append((node.lineno, s))

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root_name = module.split(".", 1)[0] if module else ""
            if root_name in RELEVANT_IMPORT_ROOTS:
                names = ", ".join(
                    f"{a.name} as {a.asname}" if a.asname else a.name
                    for a in node.names
                )
                found.append(
                    (node.lineno, f"from {'.' * node.level}{module} import {names}")
                )

    return sorted(set(found))


def audit_main_files(root):
    findings = []
    header("MAINLINE IMPORTS")

    for relpath in MAIN_FILES:
        path = root / relpath
        print(f"\n[{relpath}]")
        if not path.is_file():
            print("  MISSING")
            findings.append(Finding("FAIL", f"Missing main file: {relpath}"))
            continue

        imports = scan_imports(path)
        if not imports:
            print("  No relevant imports found.")
        for lineno, statement in imports:
            print(f"  L{lineno:<4} {statement}")

    engine = root / "engines" / "engine_strict_global_degfield.py"
    if engine.is_file():
        text = engine.read_text(encoding="utf-8")
        uses_gdf = "from gdf_loss import SSIMLoss" in text
        uses_legacy = (
            "from loss import SSIMLoss" in text
            or "\nimport loss" in text
        )

        print("\n[Main engine loss isolation]")
        print(
            "  from gdf_loss import SSIMLoss : "
            + ("PASS" if uses_gdf else "MISSING")
        )
        print(
            "  legacy loss import present    : "
            + ("YES" if uses_legacy else "no")
        )

        if not uses_gdf:
            findings.append(Finding(
                "FAIL",
                "Main engine does not explicitly import SSIMLoss from gdf_loss."
            ))
        if uses_legacy:
            findings.append(Finding(
                "FAIL",
                "Main engine still imports legacy `loss`."
            ))

    return findings


def main():
    args = parse_args()
    root = detect_root(args.root)

    root_str = str(root)
    sys.path[:] = [p for p in sys.path if p != root_str]
    sys.path.insert(0, root_str)

    print("=" * 88)
    print("GDF-Restormer import-resolution audit")
    print("=" * 88)
    print(f"Repository root : {root}")
    print(f"Python          : {sys.version.split()[0]}")
    print(f"sys.path[0]     : {sys.path[0]}")
    print(f"Mode            : {'STRICT' if args.strict else 'REPORT'}")

    findings = []

    header("AMBIGUOUS TOP-LEVEL NAMES")
    for name in AMBIGUOUS_NAMES:
        findings.extend(inspect_resolution(root, name))

    header("MAINLINE SYMBOL RESOLUTION")

    print("\n[config -> Config]")
    findings.extend(import_symbol(root, "config", ("Config",)))

    print("\n[utils -> seed_everything]")
    findings.extend(import_symbol(root, "utils", ("seed_everything",)))

    print("\n[gdf_loss -> SSIMLoss]")
    findings.extend(import_symbol(root, "gdf_loss", ("SSIMLoss",)))

    print("\n[legacy loss (informational)]")
    try:
        spec = importlib.util.find_spec("loss")
    except Exception as exc:
        print(f"  resolution      : ERROR: {exc}")
    else:
        if spec is None:
            print("  resolution      : not present")
        else:
            print(f"  resolved origin : {rel(spec.origin, root)}")
            print(
                "  status          : historical code is allowed; "
                "main engine must use gdf_loss"
            )

    findings.extend(audit_main_files(root))

    header("SUMMARY")
    warns = [x for x in findings if x.level == "WARN"]
    fails = [x for x in findings if x.level == "FAIL"]

    print("\nWarnings:")
    if warns:
        for x in warns:
            print(f"  [WARN] {x.message}")
    else:
        print("  none")

    print("\nMainline failures:")
    if fails:
        for x in fails:
            print(f"  [FAIL] {x.message}")
    else:
        print("  none")

    print()
    if fails:
        print("[FAIL] Main public training import path needs attention.")
        return 1 if args.strict else 0

    print("[PASS] Main public training imports are resolvable.")
    if warns:
        print(
            "[INFO] Historical duplicate names exist, "
            "but no mainline failure was detected."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
