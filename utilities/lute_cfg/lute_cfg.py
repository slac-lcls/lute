"""Utility to reconstruct LUTE config YAML files and restore repositories to their run state."""

import argparse
import base64
import json
import os
import subprocess
import sqlite3
import yaml
import zlib
from typing import Any, Dict, List, Optional


def find_version_location(schema_dict: Dict[str, Any]) -> Optional[str]:
    """Search for version_location const in schema definition.

    Args:
        schema_dict (Dict[str, Any]): A TaskParameters schema.

    Returns:
        version_location (Optional[str]): The location where version info was taken
            from, if provided. (E.g. a git repository clone).
    """
    for defs_key in ["definitions", "$defs"]:
        if defs_key in schema_dict:
            defs = schema_dict[defs_key]
            for config_key in ["Config", "ParameterConfig"]:
                if config_key in defs:
                    props = defs[config_key].get("properties", {})
                    if "version_location" in props:
                        return props["version_location"].get("const")
    return None


def restore_repo(repo_path: str, sha: str, diff_b64: Optional[str]) -> None:
    """Restore a git repository to a specific commit and apply a diff if provided.

    Args:
        repo_path (str): The location of the repository to restore the state of.

        sha (str): The commit hash to checkout.

        diff_b64 (Optional[str]): A diff to apply on top of the commit checkout
            if provided. This would be the base64 compressed version stored in
            the database.
    """
    if not os.path.exists(repo_path):
        print(
            f"Warning: Repository path '{repo_path}' does not exist. Cannot restore version."
        )
        return

    print(f"\nRestoring repository at '{repo_path}':")

    # Stash any uncommitted changes before making modifications
    status_proc = subprocess.run(
        ["git", "status", "--porcelain"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=repo_path,
        text=True,
    )
    stashed: bool = False
    if status_proc.stdout.strip():
        print("Uncommitted changes detected. Stashing...")
        subprocess.run(["git", "stash", "-u"], cwd=repo_path, check=True)
        stashed = True

    try:
        # Checkout the requested commit
        print(f"Checking out commit {sha}...")
        subprocess.run(["git", "checkout", sha], cwd=repo_path, check=True)

        # If a diff was also recorded, apply the diff now
        if diff_b64:
            compressed: bytes = base64.b64decode(diff_b64)
            diff_bytes: bytes = zlib.decompress(compressed)
            print(f"Will apply the following diff:\n{diff_bytes.decode('utf-8')}")

            print("Applying recorded git diff...")
            apply_proc: subprocess.Popen = subprocess.Popen(
                ["git", "apply", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=repo_path,
            )
            out: bytes
            err: bytes
            out, err = apply_proc.communicate(input=diff_bytes)
            if apply_proc.returncode != 0:
                print(
                    f"Warning: Failed to apply diff: {err.decode('utf-8', errors='ignore')}"
                )
            else:
                print("Diff successfully applied.")
    except Exception as e:
        print(f"Error restoring version: {e}")
        if stashed:
            print("Attempting to restore stashed changes...")
            subprocess.run(["git", "stash", "pop"], cwd=repo_path)
    else:
        if stashed:
            print(
                "Note: Your uncommitted changes are stashed. "
                "Use `git stash pop` to recover them if you switch branches back."
            )


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Reconstruct LUTE config YAML and restore git repositories to their "
            "state during a specific run."
        )
    )
    parser.add_argument(
        "-d",
        "--database",
        required=True,
        type=str,
        help="Path to the LUTE database.",
    )
    parser.add_argument(
        "-r",
        "--run",
        required=True,
        type=str,
        help="Run number to reconstruct config for.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Location to output the new config YAML (directory or file path).",
    )

    args: argparse.Namespace = parser.parse_args()

    # Resolve database path
    db_dir: str = args.database
    if os.path.isdir(db_dir):
        db_path: str = os.path.join(db_dir, "lute.db")
    else:
        db_path = db_dir
        db_dir = os.path.dirname(db_path)

    if not os.path.exists(db_path):
        print(f"Error: Database file not found at '{db_path}'")
        return

    # Get the latest execution for each Task for the requested run
    query = """
    WITH latest_executions AS (
        SELECT e.task_id, MAX(e.id) AS execution_id
        FROM executions e
        JOIN config c ON e.config_id = c.id
        WHERE c.run = ? OR c.run = ?
        GROUP BY e.task_id
    )
    SELECT
        t.name AS task_name,
        c.title AS config_title,
        c.experiment AS config_experiment,
        c.run AS config_run,
        c.date AS config_date,
        c.lute_version AS config_lute_version,
        c.task_timeout AS config_task_timeout,
        v.version_specifier AS version_specifier,
        v.version_info AS version_info,
        p.name AS param_name,
        p.value AS param_val,
        pt.definition AS param_schema
    FROM latest_executions le
    JOIN executions e ON le.execution_id = e.id
    JOIN config c ON e.config_id = c.id
    JOIN tasks t ON e.task_id = t.id
    LEFT JOIN task_version v ON e.task_version_id = v.id
    JOIN parameters p ON p.execution_id = e.id
    JOIN parameter_types pt ON e.parameter_type_id = pt.id
    """

    run_str: str = str(args.run)
    run_int: Optional[int]  # Check its a valid number
    try:
        run_int = int(args.run)
    except ValueError:
        run_int = None

    con: sqlite3.Connection = sqlite3.connect(db_path)
    cur: sqlite3.Cursor = con.cursor()
    cur.execute(query, (run_str, run_str if run_int is None else run_int))
    rows: List[Any] = cur.fetchall()

    if not rows:
        print(
            f"Error: No executions found for run '{args.run}' in database '{db_path}'."
        )
        con.close()
        return

    # Extract configuration headers
    config_title: Optional[str] = None
    config_experiment: Optional[str] = None
    config_run: Optional[str] = None
    config_date: Optional[str] = None
    config_lute_version: Optional[str] = None
    config_task_timeout: Optional[str] = None

    tasks_params: Dict[str, Dict[str, Any]] = {}
    tasks_versions: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        (
            task_name,
            title,
            experiment,
            run,
            date,
            lute_version,
            task_timeout,
            version_specifier,
            version_info_str,
            param_name,
            param_val_str,
            param_schema,
        ) = row

        if config_title is None:
            config_title = title
            config_experiment = experiment
            config_run = run
            config_date = date
            config_lute_version = lute_version
            config_task_timeout = task_timeout

        if task_name not in tasks_params:
            tasks_params[task_name] = {}

        try:
            param_val = json.loads(param_val_str)
        except Exception:
            param_val = param_val_str
        tasks_params[task_name][param_name] = param_val

        # Store version metadata for repository restoration
        if task_name not in tasks_versions and version_info_str:
            try:
                version_info_dict: Dict[str, str] = json.loads(version_info_str)
            except Exception:
                version_info_dict = {}

            try:
                schema_dict: Dict[str, Any] = json.loads(param_schema)
            except Exception:
                schema_dict = {}

            tasks_versions[task_name] = {
                "version_info": version_info_dict,
                "schema": schema_dict,
            }

    con.close()

    # Reconstruct the config YAML
    header_doc = {
        "title": config_title,
        "experiment": config_experiment,
        "run": config_run,
        "date": config_date,
        "lute_version": config_lute_version,
        "task_timeout": config_task_timeout,
        "work_dir": os.path.abspath(db_dir),
    }

    output_path = args.output
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, f"reconstructed_run_{args.run}.yaml")

    with open(output_path, "w") as f:
        yaml.safe_dump_all([header_doc, tasks_params], f, default_flow_style=False)

    print(f"Successfully reconstructed config YAML at '{output_path}'")

    # Finally, restore any git changes (commit checkout and diff application)
    for task_name, version_data in tasks_versions.items():
        banner: str = f"++++ Looking at version information for: {task_name} ++++"
        blen: int = len(banner)
        divider: str = "+" * blen
        print(f"{banner}\n{divider}")

        version_info: Dict[str, str] = version_data["version_info"]
        schema: Dict[str, Any] = version_data["schema"]

        sha: Optional[str] = version_info.get("git-sha")
        diff_b64: Optional[str] = version_info.get("git-diff")
        if sha:
            repo_path: Optional[str] = version_info.get("version-location")
            if not repo_path:
                repo_path = find_version_location(schema)
            else:
                # For safety this is JSON encoded individually
                repo_path = json.loads(repo_path)

            if repo_path:
                restore_repo(repo_path, sha, diff_b64)
            else:
                print(
                    f"Task '{task_name}' recorded version SHA {sha} but no "
                    "version_location was found in its schema."
                )

        lute_sha: Optional[str] = version_info.get("lute-version")
        if lute_sha:
            lute_dir = os.getenv("LUTE_PATH")
            if lute_dir is None:
                try:
                    import lute

                    lute_dir = os.path.abspath(
                        os.path.join(os.path.dirname(lute.__file__), "..")
                    )
                except ImportError:
                    pass
            if lute_dir:
                restore_repo(lute_dir, lute_sha, None)


if __name__ == "__main__":
    main()
