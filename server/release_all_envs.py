#!/usr/bin/env python3
"""
Release all environments from the AndroidEnv HTTP server.

- Fetches `/v1/server_status` to list env IDs
- Calls `POST /v1/envs/<env_id>/release` for each

Usage examples:
  python android_env/server/release_all_envs.py --host localhost --port 5000
  python android_env/server/release_all_envs.py --only-busy false
  python android_env/server/release_all_envs.py --concurrency 8
  python android_env/server/release_all_envs.py --dry-run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from typing import Dict, List, Tuple

import requests


def build_base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1"


def fetch_server_status(base_url: str, timeout_s: int = 20) -> Dict:
    url = f"{base_url}/server_status"
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    return response.json()


def list_envs(status_payload: Dict) -> List[Tuple[str, Dict]]:
    environments = status_payload.get("environments", {})
    return list(environments.items())


def release_one(base_url: str, env_id: str, timeout_s: int = 60) -> Tuple[str, bool, str]:
    url = f"{base_url}/envs/{env_id}/release"
    try:
        response = requests.post(url, timeout=timeout_s)
        # 204 No Content indicates success; 200 with JSON message indicates already idle
        if response.status_code == 204:
            return env_id, True, "released"
        if response.status_code == 200:
            try:
                msg = response.json().get("message", "already idle")
            except Exception:
                msg = "already idle"
            return env_id, True, msg
        # 425 Too Early can happen if env is still being created
        if response.status_code == 425:
            return env_id, False, "creating (skipped)"
        # Other non-2xx
        return env_id, False, f"HTTP {response.status_code}: {response.text.strip()}"
    except requests.RequestException as exc:
        return env_id, False, f"request error: {exc}"


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Release all envs on AndroidEnv server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--only-busy", type=str, default="true", help="Only release BUSY envs (true/false)")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent release requests")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    args = parser.parse_args(argv)

    only_busy = str(args.only_busy).lower() in {"1", "true", "yes", "y"}
    base_url = build_base_url(args.host, args.port)

    print(f"Fetching server status from {base_url} ...")
    try:
        status = fetch_server_status(base_url)
    except Exception as exc:
        print(f"Failed to fetch server status: {exc}")
        return 2

    env_items = list_envs(status)
    if not env_items:
        print("No environments found.")
        return 0

    # Filter by BUSY if requested
    target_env_ids: List[str] = []
    for env_id, info in env_items:
        env_status = info.get("status", "UNKNOWN")
        if only_busy and env_status != "BUSY":
            continue
        target_env_ids.append(env_id)

    if not target_env_ids:
        print("No target environments to release (filter may have excluded all).")
        return 0

    print(f"Found {len(target_env_ids)} env(s) to release: {[e[:8] for e in target_env_ids]}")

    if args.dry_run:
        print("Dry run: no requests sent.")
        return 0

    start_time = time.time()
    successes = 0
    results: List[Tuple[str, bool, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        future_to_env = {
            executor.submit(release_one, base_url, env_id): env_id for env_id in target_env_ids
        }
        for future in concurrent.futures.as_completed(future_to_env):
            env_id = future_to_env[future]
            try:
                env_id, ok, msg = future.result()
                results.append((env_id, ok, msg))
                if ok:
                    successes += 1
                    print(f"[OK]  {env_id} -> {msg}")
                else:
                    print(f"[ERR] {env_id} -> {msg}")
            except Exception as exc:
                print(f"[ERR] {env_id} -> exception: {exc}")

    elapsed = time.time() - start_time
    print(f"Done. Released {successes}/{len(target_env_ids)} env(s) in {elapsed:.2f}s.")
    # Non-zero exit if none succeeded and we attempted at least one
    return 0 if successes > 0 or len(target_env_ids) == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


