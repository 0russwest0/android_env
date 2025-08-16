import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests


def now_s() -> float:
    return time.perf_counter()


def http_post(url: str, payload: Dict[str, Any], timeout: float = 120.0) -> Tuple[int, Dict[str, Any], float]:
    start = now_s()
    resp = requests.post(url, json=payload, timeout=timeout)
    elapsed = now_s() - start
    try:
        data = resp.json()
    except Exception:
        data = {}
    return resp.status_code, data, elapsed


def http_post_no_payload(url: str, timeout: float = 120.0) -> Tuple[int, Dict[str, Any], float]:
    start = now_s()
    resp = requests.post(url, timeout=timeout)
    elapsed = now_s() - start
    try:
        data = resp.json()
    except Exception:
        data = {}
    return resp.status_code, data, elapsed


def powers_of_two_up_to(n: int) -> List[int]:
    stages = []
    p = 1
    while p <= n:
        stages.append(p)
        p *= 2
    return stages


class PerfTester:
    def __init__(
        self,
        base_url: str,
        config_name: str,
        max_envs: int,
        action: Dict[str, Any],
        results_csv: Optional[str],
        request_timeout_s: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.config_name = config_name
        self.max_envs = max_envs
        self.action = action
        self.results_csv = results_csv
        self.request_timeout_s = request_timeout_s

        self.env_ids: List[str] = []
        self.create_times_s: List[float] = []
        self.rows: List[List[Any]] = []

    # --------------------------
    # Environment lifecycle
    # --------------------------
    def create_env(self) -> Tuple[Optional[str], float, Optional[str]]:
        url = f"{self.base_url}/v1/envs"
        status, data, elapsed = http_post(url, {"config_name": self.config_name}, timeout=self.request_timeout_s)
        if status == 200:
            return data.get("env_id"), elapsed, None
        return None, elapsed, data.get("error") or f"HTTP {status}"

    def step_env(self, env_id: str) -> Tuple[int, float]:
        url = f"{self.base_url}/v1/envs/{env_id}/step"
        status, _data, elapsed = http_post(url, {"action": self.action}, timeout=self.request_timeout_s)
        return status, elapsed

    def release_env(self, env_id: str) -> Tuple[int, float]:
        url = f"{self.base_url}/v1/envs/{env_id}/release"
        status, _data, elapsed = http_post_no_payload(url, timeout=self.request_timeout_s)
        return status, elapsed

    # --------------------------
    # Concurrency helpers
    # --------------------------
    def run_concurrent(self, funcs: List[Tuple[str, Any]]) -> Tuple[float, List[Tuple[str, int, float]]]:
        start = now_s()
        results: List[Tuple[str, int, float]] = []
        with ThreadPoolExecutor(max_workers=len(funcs)) as executor:
            futures = {}
            for key, func in funcs:
                futures[executor.submit(func)] = key
            for future in as_completed(futures):
                key = futures[future]
                try:
                    status, elapsed = future.result()
                except Exception:
                    status, elapsed = 599, float("nan")
                results.append((key, status, elapsed))
        wall = now_s() - start
        return wall, results

    def reacquire_many_concurrent(self, count: int) -> Tuple[float, List[str], List[float]]:
        start = now_s()
        ids: List[str] = []
        durations: List[float] = []
        with ThreadPoolExecutor(max_workers=count) as executor:
            futures = [executor.submit(self.create_env) for _ in range(count)]
            for future in as_completed(futures):
                try:
                    env_id, elapsed, _err = future.result()
                except Exception:
                    env_id, elapsed = None, float("nan")
                durations.append(elapsed)
                if env_id is not None:
                    ids.append(env_id)
        wall = now_s() - start
        return wall, ids, durations

    # --------------------------
    # Metrics recording
    # --------------------------
    def record(self, stage_envs: int, test_type: str, concurrency: int, wall_s: float, per_req: List[float]) -> None:
        avg_req = sum(per_req) / len(per_req) if per_req else float("nan")
        row = [stage_envs, test_type, concurrency, wall_s, avg_req]
        self.rows.append(row)

    def write_results(self) -> None:
        if not self.results_csv:
            return
        header = ["stage_envs", "test_type", "concurrency", "wall_s", "avg_req_s"]
        try:
            with open(self.results_csv, "w", encoding="utf-8") as f:
                f.write(",".join(header) + "\n")
                for row in self.rows:
                    f.write(",".join(str(x) for x in row) + "\n")
        except Exception as e:
            print(f"Failed to write results CSV: {e}")

    # --------------------------
    # Test routines
    # --------------------------
    def run(self) -> None:
        print(f"Server: {self.base_url} | config: {self.config_name} | max_envs: {self.max_envs}")
        stages = powers_of_two_up_to(self.max_envs)

        total_create_start = now_s()
        next_stage_index = 0

        for i in range(1, self.max_envs + 1):
            # Repeatedly call create until a NEW env_id is returned.
            attempts = 0
            new_env_id: Optional[str] = None
            total_elapsed_this_creation = 0.0
            while attempts < self.max_envs * 10:
                env_id, elapsed, err = self.create_env()
                attempts += 1
                total_elapsed_this_creation += elapsed
                if env_id is None:
                    print(f"[CREATE {i}] FAILED in {elapsed:.2f}s: {err}. Stopping further creation.")
                    new_env_id = None
                    break
                if env_id not in self.env_ids:
                    new_env_id = env_id
                    break
                # If reused an existing env, this call made it BUSY; keep trying until a new one is created
            if new_env_id is None:
                print(f"[CREATE {i}] Could not obtain a new environment after {attempts} attempts. Stop.")
                break
            self.create_times_s.append(total_elapsed_this_creation)
            self.env_ids.append(new_env_id)
            print(f"[CREATE {i}] env_id={new_env_id} attempts={attempts} total_time={total_elapsed_this_creation:.2f}s")

            # If we reached a stage boundary (1,2,4,8,16,32...), run tests
            if next_stage_index < len(stages) and i == stages[next_stage_index]:
                stage_envs = i
                stage_total_time = now_s() - total_create_start
                # Record creation summary for this stage
                self.record(stage_envs, "create_total", 1, stage_total_time, self.create_times_s[:])
                print(f"-- Stage {stage_envs}: created in total {stage_total_time:.2f}s")

                # Step tests: concurrency 1..stage_envs
                print(f"-- Stage {stage_envs}: step tests (1..{stage_envs})")
                for c in range(1, stage_envs + 1):
                    subset = self.env_ids[:c]
                    funcs = [(eid, (lambda eid=eid: self.step_env(eid))) for eid in subset]
                    wall, results = self.run_concurrent(funcs)
                    per_req = [r[2] for r in results if not math.isnan(r[2])]  # durations
                    self.record(stage_envs, "step", c, wall, per_req)
                    avg = (sum(per_req)/len(per_req)) if per_req else float('nan')
                    print(f"   step x{c}: wall={wall:.3f}s avg={avg:.3f}s")

                # Make all first stage_envs environments IDLE before release tests
                # Do this sequentially (not measuring here) to avoid affecting release metrics
                for eid in self.env_ids[:stage_envs]:
                    try:
                        self.release_env(eid)
                    except Exception:
                        pass

                # Release tests: only perform at full concurrency = stage_envs
                print(f"-- Stage {stage_envs}: release tests (concurrency={stage_envs})")
                reacq_wall, reacquired_ids, reacq_durations = self.reacquire_many_concurrent(stage_envs)
                self.record(stage_envs, "reacquire", stage_envs, reacq_wall, reacq_durations)
                reacq_avg = (sum(reacq_durations)/len(reacq_durations)) if reacq_durations else float('nan')
                print(f"   reacquire x{stage_envs}: wall={reacq_wall:.3f}s avg={reacq_avg:.3f}s (count={len(reacquired_ids)})")
                if len(reacquired_ids) < stage_envs:
                    print(f"   release x{stage_envs}: only reacquired {len(reacquired_ids)}; skipping")
                else:
                    funcs = [(eid, (lambda eid=eid: self.release_env(eid))) for eid in reacquired_ids]
                    wall, results = self.run_concurrent(funcs)
                    per_req = [r[2] for r in results if not math.isnan(r[2])]
                    self.record(stage_envs, "release", stage_envs, wall, per_req)
                    avg = (sum(per_req)/len(per_req)) if per_req else float('nan')
                    print(f"   release x{stage_envs}: wall={wall:.3f}s avg={avg:.3f}s (reacquired={len(reacquired_ids)})")

                # Prepare for next stage: set first stage_envs back to BUSY without recording metrics
                reacq_all_wall, reacq_all_ids, _ = self.reacquire_many_concurrent(stage_envs)
                print(f"-- Stage {stage_envs}: prepared next stage (reacquired {len(reacq_all_ids)}/{stage_envs}) in {reacq_all_wall:.3f}s")

                next_stage_index += 1

        # Finalize
        self.write_results()
        print("Done. Results stored." if self.results_csv else "Done.")


def parse_action_json(s: Optional[str]) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        print(f"Invalid --action-json. Using empty dict. Error: {e}")
        return {}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Performance tests for android_env server")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:5000", help="Base URL of the server")
    parser.add_argument("--config-name", type=str, default="default", help="Server config name to use")
    parser.add_argument("--max-envs", type=int, default=32, help="Maximum number of environments to create (sequential)")
    parser.add_argument("--action-json", type=str, default="", help="JSON string for the action passed to step API")
    parser.add_argument("--results-csv", type=str, default="/root/android_env/server/perf_results_2.csv", help="Path to write CSV results")
    parser.add_argument("--timeout-s", type=float, default=300.0, help="Per-request timeout in seconds")

    args = parser.parse_args(argv)

    if args.action_json == "":
        args.action_json = json.dumps({
            "action": "click",
            "coordinate": [540, 1170]
        })

    tester = PerfTester(
        base_url=args.base_url,
        config_name=args.config_name,
        max_envs=args.max_envs,
        action=parse_action_json(args.action_json),
        results_csv=args.results_csv,
        request_timeout_s=args.timeout_s,
    )
    tester.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())


