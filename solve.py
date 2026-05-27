"""Inference: load model.pt, run A* (with V as heuristic), write CSV (parallel)."""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

import gym
import common
from common import state_key
from model import ValueNet
from search import solve_astar


TIME_LIMIT_DEFAULT = 25 * 60   # 25 minutes total for 1000 instances
SAFETY_MARGIN = 30
MODEL_PATH = "model.pt"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = ValueNet()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def make_v_fn(env, model):
    if model is None:
        return lambda states: np.zeros(len(states), dtype=np.float32)

    def v_fn(states):
        tokens = np.stack([common.encode_tokens(env, s) for s in states])
        B, N, _ = tokens.shape
        parts = common.split_token_features(tokens.reshape(B * N, -1))
        dense = torch.from_numpy(parts["dense"].reshape(B, N, -1))
        cv = torch.from_numpy(parts["content_value"].reshape(B, N))
        tv = torch.from_numpy(parts["target_value"].reshape(B, N))
        with torch.no_grad():
            return model(dense, cv, tv).cpu().numpy()

    return v_fn


def solve_one(env, solved_k, v_fn, inst, deadline):
    iid = inst["instance_id"]
    try:
        sol = solve_astar(env, inst["state"], solved_k, v_fn, deadline)
        actions = sol or []
        return iid, " ".join(actions), len(actions) > 0
    except Exception as e:
        print(f"  {iid} failed: {repr(e)}")
        return iid, "", False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input_states.jsonl")
    parser.add_argument("--output", default="output_actions.csv")
    parser.add_argument("--time_limit", type=int,
                        default=int(os.environ.get("SOLVE_TIME_LIMIT", TIME_LIMIT_DEFAULT)))
    parser.add_argument("--parallel", type=int, default=8)
    args = parser.parse_args()

    start = time.time()
    deadline = start + args.time_limit - SAFETY_MARGIN
    torch.set_num_threads(1)  # Avoid oversubscription in threads

    env = gym.make_env()
    instances = load_jsonl(args.input)
    print(f"loaded {len(instances)} instances")

    model = load_model()
    print(f"model loaded: {model is not None}")

    env.reset()
    solved_k = state_key(env.get_state())
    v_fn = make_v_fn(env, model)

    n = len(instances)
    solved = 0
    results = [None] * n

    # Distribute deadlines: each instance gets at least 0.5s, and the remaining time is split
    remaining_instances = n
    time_per_instance = max(1.0, (deadline - time.time()) / remaining_instances)

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for i, inst in enumerate(instances):
            inst_deadline = time.time() + time_per_instance
            future = executor.submit(
                solve_one, env, solved_k, v_fn, inst, inst_deadline
            )
            futures.append((i, future))
            # Update remaining dynamic (approximate)
            remaining_instances -= 1
            if remaining_instances > 0:
                time_per_instance = max(0.5, (deadline - time.time()) / remaining_instances)

        for i, future in futures:
            iid, actions_str, ok = future.result()
            results[i] = (iid, actions_str)
            if ok:
                solved += 1
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{n} solved={solved} elapsed={time.time()-start:.0f}s")

    # Write output CSV
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance_id", "actions"])
        writer.writeheader()
        for iid, actions_str in results:
            writer.writerow({"instance_id": iid, "actions": actions_str})

    print(f"final: solved {solved}/{n}, time {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()