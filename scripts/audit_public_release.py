"""Check candidate/staged public files; does not upload, modify or delete anything."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {".py", ".md", ".json", ".toml", ".txt", ".yml", ".yaml", ".ps1", ".sh"}
SPECIAL = {".gitignore", ".gitattributes", "LICENSE", "NOTICE"}
PRIVATE_DIRS = {"results", "outputs", "checkpoints", "cache", "backups"}
SKIP = {".git", ".venv", ".venv_gpu", "venv", "env", "__pycache__", ".pytest_cache", "build", "dist"}
SECRETS = [re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
           re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
           re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
           re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b")]


def inspect(path, content):
    p = Path(path)
    reasons = []
    if p.is_absolute() or ".." in p.parts:
        reasons.append("unexpected path")
    if set(p.parts) & PRIVATE_DIRS:
        reasons.append("private output/cache directory")
    if p.parts and p.parts[0] == "data" and p.as_posix() not in {"data/README.md", "data/raw/README.md"}:
        reasons.append("data directory accepts documentation only")
    if p.suffix.lower() not in ALLOWED and p.name not in SPECIAL:
        reasons.append("binary/data/archive or unreviewed file type")
    if p.name.startswith(".env") or p.suffix.lower() in {".pem", ".key"}:
        reasons.append("environment/credential file")
    if any(pattern.search(content) for pattern in SECRETS):
        reasons.append("possible credential; value suppressed")
    if b"\x00" in content:
        reasons.append("unexpected binary content")
    if len(content) > 2 * 1024 * 1024:
        reasons.append("unexpectedly large source file")
    return reasons


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL)


def candidates(staged=False):
    if staged:
        paths = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").split(b"\0")
        return [(p.decode(), git("show", ":" + p.decode())) for p in paths if p]
    try:
        git("rev-parse", "--show-toplevel")
        paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0")
        names = sorted({p.decode() for p in paths if p})
        return [(p, (ROOT / p).read_bytes()) for p in names if (ROOT / p).is_file()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [(p.relative_to(ROOT).as_posix(), p.read_bytes()) for p in ROOT.rglob("*")
                if p.is_file() and not set(p.relative_to(ROOT).parts) & SKIP
                and not any(x.endswith(".egg-info") for x in p.relative_to(ROOT).parts)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Inspect the exact index blobs that would be committed.")
    args = parser.parse_args()
    try:
        items = candidates(args.staged)
    except (FileNotFoundError, subprocess.CalledProcessError):
        parser.error("Staged audit requires Git and an initialized local repository.")
    failed = False
    for path, content in items:
        reasons = inspect(path, content)
        if reasons:
            failed = True
            print(path + ": " + "; ".join(reasons))
    print(f"{'FAIL' if failed else 'PASS'}: reviewed {len(items)} files.")
    print("Content checks are heuristic. Review the file list before committing.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
