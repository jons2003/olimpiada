"""Forward A* with V(s) as heuristic, batched child evaluation, and cache."""

import heapq
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from common import state_key, to_jsonable


def solve_astar(
    env,
    initial_state: Any,
    solved_key: str,
    v_fn: Optional[Callable[[List[Any]], np.ndarray]],
    deadline: float,
    max_nodes: int = 100_000,
    expand_batch: int = 32,
    h_cache: Optional[Dict[str, float]] = None,
) -> Optional[List[str]]:
    """A* with f = g + V(s). Returns action list or None."""
    start = to_jsonable(initial_state)
    start_k = state_key(start)
    if start_k == solved_key:
        return []

    if h_cache is None:
        h_cache = {}

    parents: Dict[str, Tuple[Optional[str], Optional[str]]] = {start_k: (None, None)}
    g_score: Dict[str, int] = {start_k: 0}

    # Heuristic for start
    if start_k in h_cache:
        h0 = h_cache[start_k]
    else:
        h0 = float(v_fn([start])[0]) if v_fn else 0.0
        h_cache[start_k] = h0

    counter = 0
    open_heap = [(h0, counter, 0, start_k, start)]
    expanded = 0

    while open_heap:
        if time.time() >= deadline or expanded >= max_nodes:
            return None

        # Pop a batch
        batch = []
        while open_heap and len(batch) < expand_batch:
            batch.append(heapq.heappop(open_heap))

        # Collect unique children
        children = []  # (child_key, child_state, parent_key, action, g)
        seen = set()
        for _f, _c, g, sk, state in batch:
            if sk == solved_key:
                return _reconstruct(sk, parents)
            try:
                env.set_state(state)
                valid = env.valid_actions()
            except Exception:
                continue
            for a in valid:
                try:
                    env.set_state(state)
                    env.step(a)
                    ns = to_jsonable(env.get_state())
                except Exception:
                    continue
                nsk = state_key(ns)
                ng = g + 1
                if g_score.get(nsk, 1 << 30) <= ng or nsk in seen:
                    continue
                seen.add(nsk)
                children.append((nsk, ns, sk, a, ng))

        if not children:
            continue

        # Check goal among children
        for nsk, ns, psk, a, ng in children:
            if nsk == solved_key:
                parents[nsk] = (psk, a)
                return _reconstruct(nsk, parents)

        /# Batched V over unseen children
        states_to_eval = []
        idx_to_child = []
        for i, (nsk, ns, psk, a, ng) in enumerate(children):
            if nsk in h_cache:
                h_val = h_cache[nsk]
            else:
                states_to_eval.append(ns)
                idx_to_child.append(i)
        if states_to_eval:
            h_vals = v_fn(states_to_eval) if v_fn else np.zeros(len(states_to_eval), np.float32)
            for ii, h in zip(idx_to_child, h_vals):
                nsk = children[ii][0]
                h_cache[nsk] = float(h)

        # Update open set
        for i, (nsk, ns, psk, a, ng) in enumerate(children):
            h_val = h_cache[nsk]
            parents[nsk] = (psk, a)
            g_score[nsk] = ng
            counter += 1
            heapq.heappush(open_heap, (ng + h_val, counter, ng, nsk, ns))
            expanded += 1
            if expanded >= max_nodes:
                break

    return None


def _reconstruct(end_k: str, parents) -> List[str]:
    actions = []
    cur = end_k
    while True:
        pk, a = parents.get(cur, (None, None))
        if pk is None or a is None:
            break
        actions.append(a)
        cur = pk
    actions.reverse()
    return actions