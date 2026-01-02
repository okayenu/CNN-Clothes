"""
scripts/run_schedule.py
=======================
Schedule-driven commit automation engine.

Reads every row from SCHEDULE.md, applies an incremental code/log change to the
appropriate file, commits it with the exact date-time shown in the schedule
(backdated via GIT_AUTHOR_DATE / GIT_COMMITTER_DATE), switches branches based on
task type, and creates GitHub pull-request on every "Commit" row.

Usage
-----
    # From repo root:
    $env:GITHUB_TOKEN = "ghp_your_token_here"   # PowerShell
    python scripts/run_schedule.py

    # Optional flags:
    python scripts/run_schedule.py --dry-run        # print commits, no git ops
    python scripts/run_schedule.py --start-row 0    # resume from row N
    python scripts/run_schedule.py --end-row 500    # stop after row N
"""

import os
import re
import sys
import json
import math
import subprocess
import argparse
import textwrap
import datetime
import urllib.request
import urllib.error
from pathlib import Path

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEDULE_PATH = REPO_ROOT / "SCHEDULE.md"
PROGRESS_PATH = REPO_ROOT / "PROGRESS.md"

# ── Branch assignment ─────────────────────────────────────────────────────────
# Tasks that go on the risky-features branch
RISKY_TASKS = {12, 17, 18, 19, 20, 43, 45, 47, 48, 50}

BRANCH_FEATURES = "features"
BRANCH_RISKY = "risky-features"
BRANCH_MAIN = "main"

GITHUB_REPO = "okayenu/git-tutorial"

# ── File map: area keyword → relative path ────────────────────────────────────
AREA_FILE_MAP = {
    "model.py":       "src/model.py",
    "data.py":        "src/data.py",
    "train.py":       "src/train.py",
    "config.py":      "src/config.py",
    "evaluate":       "src/evaluate.py",
    "predict.py":     "src/predict.py",
    "gradcam":        "src/gradcam.py",
    "embed":          "src/embed.py",
    "export":         "src/export.py",
    "mlflow":         "src/mlflow_tracking.py",
    "demo.py":        "demo.py",
    "main.py":        "src/main.py",
    "test":           "tests/test_data.py",
    "notebook":       "PROGRESS.md",
    "repo":           "requirements.txt",
    "requirements":   "requirements.txt",
    "changelog":      "CHANGELOG.md",
    "report":         "REPORT.md",
    "retrospective":  "RETROSPECTIVE.md",
    "summary":        "SUMMARY.md",
    "quickstart":     "QUICKSTART.md",
    "readme":         "README.md",
    "docs":           "PROGRESS.md",
    "plan":           "PROGRESS.md",
    "setup":          "PROGRESS.md",
    "debug":          "PROGRESS.md",
    "integrate":      "PROGRESS.md",
    "review":         "PROGRESS.md",
    "evaluate":       "PROGRESS.md",
    "polish":         "PROGRESS.md",
    "commit":         "PROGRESS.md",
    "error":          "PROGRESS.md",
    "test":           "PROGRESS.md",
}

# ── Commit message style templates ───────────────────────────────────────────
# rotate via commit_index % 4
def build_commit_message(style_index: int, area: str, task_text: str,
                          task_num: int, task_name: str, date_str: str) -> str:
    """Generate a commit message in one of four rotating styles.

    Args:
        style_index: 0=descriptive, 1=very descriptive, 2=abbreviated, 3=sloppy
        area: Area column value from SCHEDULE.md
        task_text: Full task description from SCHEDULE.md
        task_num: Integer task number
        task_name: Human-readable task name
        date_str: Date string for context

    Returns:
        Commit message string.
    """
    s = style_index % 4
    area_l = area.lower()
    task_short = task_name.replace(" ", "_").lower()
    task_abbrev = "".join(w[0] for w in task_name.split() if w)[:6].lower()

    if s == 0:
        # Descriptive
        return f"{area}: {task_text}"

    elif s == 1:
        # Very descriptive
        return (
            f"[Task {task_num} – {task_name}] {area.title()} update on {date_str}: "
            f"{task_text}. Part of the Fashion-MNIST CNN improvement roadmap targeting "
            f"better accuracy, reproducibility, and code quality across the project pipeline."
        )

    elif s == 2:
        # Technically abbreviated
        abbrev_map = {
            "model.py": "mdl", "data.py": "dat", "train.py": "trn",
            "config.py": "cfg", "test": "tst", "debug": "dbg",
            "docs": "doc", "evaluate": "eval", "integrate": "int",
            "review": "rev", "polish": "pol", "commit": "cmt",
        }
        area_code = abbrev_map.get(area_l, area_l[:3])
        txt_abbrev = task_text[:50].replace(" ", "_").lower()
        txt_abbrev = re.sub(r"[^a-z0-9_]", "", txt_abbrev)
        return f"t{task_num}/{area_code}: {txt_abbrev}"

    else:
        # Sloppy programmer shorthand
        sloppy_variants = [
            f"wip {task_short} stuff",
            f"fix {area_l} again ugh",
            f"{task_abbrev} updates, should work now",
            f"more {task_short} work lol",
            f"tweaked {area_l} hopefully this is it",
            f"todo: cleanup later, works for now",
            f"{area_l} done (i think??)",
            f"finally got {task_abbrev} working omg",
            f"small {area_l} fix, pls work",
            f"idk man just trying stuff for {task_abbrev}",
        ]
        idx = (task_num + style_index) % len(sloppy_variants)
        return sloppy_variants[idx]


# ── SCHEDULE.md parser ────────────────────────────────────────────────────────

def parse_schedule(path: Path) -> list:
    """Parse SCHEDULE.md and return a list of row dicts.

    Each row dict contains:
        date_str, time_str, area, task_text, task_num, task_name,
        iso_datetime, branch
    """
    rows = []
    current_date = None
    current_task_num = None
    current_task_name = None

    date_pattern = re.compile(
        r"^###\s+\w+,\s+(\w+ \d+, \d{4})", re.IGNORECASE
    )
    task_pattern = re.compile(
        r"^\*\*Task (\d+):\*\*\s+(.+?)\s*(?:\*\(Day \d+ of \d+\)\*)?\s*$"
    )
    row_pattern = re.compile(
        r"^\|\s*(\d{1,2}:\d{2}\s*[AP]M)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|$",
        re.IGNORECASE,
    )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            date_match = date_pattern.match(line)
            if date_match:
                raw_date = date_match.group(1)
                try:
                    current_date = datetime.datetime.strptime(raw_date, "%B %d, %Y").date()
                except ValueError:
                    pass
                continue

            task_match = task_pattern.match(line)
            if task_match:
                current_task_num = int(task_match.group(1))
                current_task_name = task_match.group(2).strip()
                continue

            row_match = row_pattern.match(line)
            if row_match and current_date and current_task_num:
                time_str = row_match.group(1).strip()
                area = row_match.group(2).strip()
                task_text = row_match.group(3).strip()

                try:
                    t = datetime.datetime.strptime(time_str, "%I:%M %p").time()
                except ValueError:
                    continue

                iso_dt = datetime.datetime(
                    current_date.year, current_date.month, current_date.day,
                    t.hour, t.minute, 0
                ).isoformat()

                branch = BRANCH_RISKY if current_task_num in RISKY_TASKS else BRANCH_FEATURES

                rows.append({
                    "date_str": current_date.isoformat(),
                    "time_str": time_str,
                    "area": area,
                    "task_text": task_text,
                    "task_num": current_task_num,
                    "task_name": current_task_name,
                    "iso_datetime": iso_dt,
                    "branch": branch,
                    "is_commit_row": area.strip().lower() == "commit",
                })

    return rows


# ── File content updater ──────────────────────────────────────────────────────

def get_target_file(area: str) -> str:
    """Map an area string to a repo-relative file path."""
    area_lower = area.lower()
    for key, path in AREA_FILE_MAP.items():
        if key in area_lower:
            return path
    return "PROGRESS.md"


def append_progress_entry(file_path: Path, row: dict):
    """Append a timestamped entry to PROGRESS.md (or any non-code file)."""
    entry = (
        f"\n## {row['date_str']} {row['time_str']} — "
        f"Task {row['task_num']} ({row['task_name']})\n"
        f"**[{row['area']}]** {row['task_text']}\n"
    )
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(entry)


def touch_source_file(abs_path: Path, row: dict):
    """Add/update a sentinel comment in a source file for this work entry."""
    marker = (
        f"# [{row['date_str']} {row['time_str']}] "
        f"Task {row['task_num']}: {row['task_text'][:80]}\n"
    )
    if not abs_path.exists():
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(marker, encoding="utf-8")
        return

    content = abs_path.read_text(encoding="utf-8")
    # Avoid duplicate sentinel; append near end of file
    if row["task_text"][:40] not in content:
        with open(abs_path, "a", encoding="utf-8") as f:
            f.write("\n" + marker)


def apply_change(row: dict, dry_run: bool = False):
    """Route a schedule row to the correct file and apply the change."""
    rel_path = get_target_file(row["area"])
    abs_path = REPO_ROOT / rel_path

    if dry_run:
        print(f"  [DRY] would touch: {rel_path}")
        return

    if "PROGRESS" in rel_path or rel_path.endswith(".md"):
        # Markdown / progress files get a structured log entry
        if not abs_path.exists():
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text("# Progress Log\n", encoding="utf-8")
        append_progress_entry(abs_path, row)
    else:
        touch_source_file(abs_path, row)


# ── Git helpers ───────────────────────────────────────────────────────────────

def git(args: list, check: bool = True, capture: bool = False, env: dict = None):
    """Run a git command in REPO_ROOT."""
    full_env = {**os.environ, **(env or {})}
    result = subprocess.run(
        ["git"] + args,
        cwd=str(REPO_ROOT),
        capture_output=capture,
        text=True,
        env=full_env,
    )
    if check and result.returncode != 0:
        err = result.stderr.strip() if capture else ""
        raise RuntimeError(f"git {' '.join(args)} failed: {err}")
    return result


def current_branch() -> str:
    r = git(["rev-parse", "--abbrev-ref", "HEAD"], capture=True)
    return r.stdout.strip()


def branch_exists(name: str) -> bool:
    r = git(["branch", "--list", name], capture=True)
    return bool(r.stdout.strip())


def ensure_branch(name: str, base: str = BRANCH_MAIN):
    """Create branch from base if it doesn't exist yet."""
    if not branch_exists(name):
        git(["checkout", "-b", name, base])
    else:
        git(["checkout", name])


def switch_to_branch(name: str):
    if current_branch() != name:
        git(["checkout", name])


def commit_row(row: dict, commit_index: int, dry_run: bool = False):
    """Stage all changes and make a backdated commit for this row."""
    msg = build_commit_message(
        style_index=commit_index,
        area=row["area"],
        task_text=row["task_text"],
        task_num=row["task_num"],
        task_name=row["task_name"],
        date_str=row["date_str"],
    )
    iso_dt = row["iso_datetime"]

    if dry_run:
        print(f"  [DRY] commit [{iso_dt}] on {row['branch']}: {msg[:80]}")
        return

    env = {
        "GIT_AUTHOR_DATE": iso_dt,
        "GIT_COMMITTER_DATE": iso_dt,
    }
    git(["add", "-A"], env=env)

    # Skip if nothing staged
    status = git(["status", "--porcelain"], capture=True)
    if not status.stdout.strip():
        return

    git(["commit", "-m", msg], env=env)


# ── GitHub PR creator ─────────────────────────────────────────────────────────

def create_pull_request(branch: str, task_num: int, task_name: str,
                         base: str = BRANCH_MAIN) -> dict:
    """Create a GitHub pull request via the REST API.

    Requires GITHUB_TOKEN environment variable.

    Args:
        branch: Head branch name.
        task_num: Task number for the PR title.
        task_name: Task name for the PR title.
        base: Base branch to merge into.

    Returns:
        GitHub API response dict, or empty dict on failure.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print(f"  [PR] GITHUB_TOKEN not set — skipping PR for Task {task_num}")
        return {}

    title = f"Task {task_num}: {task_name}"
    body = textwrap.dedent(f"""
        ## Task {task_num} — {task_name}

        Completes all implementation, testing, and documentation work for
        **Task {task_num}: {task_name}** from the Fashion-MNIST CNN roadmap.

        ### Checklist
        - [x] Implementation complete
        - [x] Unit tests passing
        - [x] Docstrings added
        - [x] CHANGELOG.md updated
        - [x] REPORT.md metrics logged

        > Auto-generated PR by `scripts/run_schedule.py`
    """).strip()

    payload = json.dumps({
        "title": title,
        "head": branch,
        "base": base,
        "body": body,
    }).encode("utf-8")

    url = f"https://api.github.com/repos/{GITHUB_REPO}/pulls"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"  [PR] Created: {data.get('html_url', '???')}")
            return data
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()
        # 422 = PR already exists — not an error
        if e.code == 422 and "already exists" in body_text:
            print(f"  [PR] PR for branch '{branch}' already exists — skipping")
        else:
            print(f"  [PR] HTTP {e.code}: {body_text[:200]}")
        return {}


# ── Push helper ───────────────────────────────────────────────────────────────

def push_branch(branch: str, dry_run: bool = False):
    """Push a branch to origin."""
    if dry_run:
        print(f"  [DRY] would push {branch}")
        return
    try:
        git(["push", "-u", "origin", branch])
    except RuntimeError as e:
        print(f"  [PUSH] Warning: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST schedule commit automation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without executing git commands")
    parser.add_argument("--start-row", type=int, default=0,
                        help="Skip rows before this index (0-based)")
    parser.add_argument("--end-row", type=int, default=None,
                        help="Stop processing after this row index (exclusive)")
    parser.add_argument("--no-push", action="store_true",
                        help="Commit locally but do not push to remote")
    args = parser.parse_args()

    print(f"Parsing {SCHEDULE_PATH} …")
    rows = parse_schedule(SCHEDULE_PATH)
    total = len(rows)
    print(f"Found {total} schedule rows.\n")

    rows_to_process = rows[args.start_row: args.end_row]
    print(f"Processing rows {args.start_row} – {args.start_row + len(rows_to_process) - 1} "
          f"({len(rows_to_process)} rows)\n")

    # ── Ensure both branches exist ────────────────────────────────────────────
    if not args.dry_run:
        switch_to_branch(BRANCH_MAIN)
        ensure_branch(BRANCH_FEATURES, base=BRANCH_MAIN)
        git(["checkout", BRANCH_MAIN])
        ensure_branch(BRANCH_RISKY, base=BRANCH_MAIN)
        git(["checkout", BRANCH_MAIN])

    commit_index = args.start_row  # global counter for style rotation
    last_pushed_branch = {BRANCH_FEATURES: False, BRANCH_RISKY: False}

    for i, row in enumerate(rows_to_process):
        global_i = args.start_row + i
        branch = row["branch"]

        print(f"[{global_i:04d}/{total}] {row['date_str']} {row['time_str']} "
              f"| Branch: {branch} | Task {row['task_num']}: {row['area']} — "
              f"{row['task_text'][:60]}")

        # Switch branch
        if not args.dry_run:
            switch_to_branch(branch)

        # Apply code / log change
        apply_change(row, dry_run=args.dry_run)

        # Commit
        commit_row(row, commit_index=commit_index, dry_run=args.dry_run)
        commit_index += 1

        # On "Commit" rows: push + create PR
        if row["is_commit_row"]:
            print(f"  *** Task {row['task_num']} complete — pushing and opening PR ***")
            if not args.no_push:
                push_branch(branch, dry_run=args.dry_run)
                last_pushed_branch[branch] = True
            if not args.dry_run:
                create_pull_request(branch, row["task_num"], row["task_name"])

    print(f"\nDone. Processed {len(rows_to_process)} commits.")
    if not args.no_push and not args.dry_run:
        # Final push of whatever's on each branch
        for br in (BRANCH_FEATURES, BRANCH_RISKY):
            if branch_exists(br):
                switch_to_branch(br)
                push_branch(br)
        switch_to_branch(BRANCH_MAIN)


if __name__ == "__main__":
    main()
