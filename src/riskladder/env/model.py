"""Exact model of a map as sparse matrices over decision states (docs/env_spec.md section 6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from riskladder.env.dynamics import (
    ACTIONS,
    FAIL,
    N_ACTIONS,
    N_BRANCHES,
    RUNNING,
    SUCCESS,
    base_cost,
    branches,
    move_many,
)
from riskladder.env.spec import MapSpec

# Values of `branch_next` for transitions into the absorbing states.
NEXT_SUCCESS, NEXT_FAIL = -1, -2


@dataclass(frozen=True, eq=False)
class SparseMDP:
    """Transitions among the n decision states, plus absorption into SUCCESS / FAIL.

    Per (state, action, branch) arrays keep the exact outcome of every branch, which the sampler and
    Monte Carlo rollouts use. The aggregated arrays are what solvers and the oracle use:

    - `P[a]`: n x n CSR matrix, probability of moving between decision states (substochastic);
    - `p_success[s, a]`, `p_fail[s, a]`: probability of absorbing in one step;
    - `cost[s, a]`: expected one-step cost (includes `c_fail` on failure).

    For every (s, a): `P[a][s].sum() + p_success[s, a] + p_fail[s, a] == 1`.
    """

    spec: MapSpec
    cells: np.ndarray  # (n, 2) row, col of each state
    index: np.ndarray  # (rows, cols) state index, -1 for non-decision cells
    branch_prob: np.ndarray  # (A, 3)
    branch_next: np.ndarray  # (n, A, 3) state index or NEXT_SUCCESS / NEXT_FAIL
    branch_cell: np.ndarray  # (n, A, 3, 2) cell the rover ends in
    branch_cost: np.ndarray  # (n, A, 3)
    P: tuple[sparse.csr_matrix, ...]
    p_success: np.ndarray  # (n, A)
    p_fail: np.ndarray  # (n, A)
    cost: np.ndarray  # (n, A)

    @property
    def n_states(self) -> int:
        return len(self.cells)

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def start(self) -> int:
        r, c = self.spec.layout.start
        return int(self.index[r, c])

    def to_grid(self, values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """Map a per-state vector onto the (rows, cols) grid."""
        grid = np.full(self.index.shape, fill, dtype=np.result_type(values, type(fill)))
        grid[self.cells[:, 0], self.cells[:, 1]] = values
        return grid


def build_model(spec: MapSpec) -> SparseMDP:
    layout = spec.layout
    cells = np.array(layout.passable_cells, dtype=np.int64)
    n = len(cells)
    index = np.full((layout.rows, layout.cols), -1, dtype=np.int64)
    index[cells[:, 0], cells[:, 1]] = np.arange(n)

    branch_prob = np.zeros((N_ACTIONS, N_BRANCHES))
    branch_next = np.zeros((n, N_ACTIONS, N_BRANCHES), dtype=np.int32)
    branch_cell = np.zeros((n, N_ACTIONS, N_BRANCHES, 2), dtype=np.int32)
    branch_cost = np.zeros((n, N_ACTIONS, N_BRANCHES))

    for a, act in enumerate(ACTIONS):
        for b, (direction, prob) in enumerate(branches(spec, a)):
            r, c, extra, status = move_many(spec, cells[:, 0], cells[:, 1], direction, act.cells)
            branch_prob[a, b] = prob
            branch_cell[:, a, b, 0], branch_cell[:, a, b, 1] = r, c
            branch_cost[:, a, b] = base_cost(spec, a) + extra
            nxt = np.where(status == RUNNING, index[r, c], 0)
            nxt[status == SUCCESS] = NEXT_SUCCESS
            nxt[status == FAIL] = NEXT_FAIL
            branch_next[:, a, b] = nxt

    P = []
    p_success = np.zeros((n, N_ACTIONS))
    p_fail = np.zeros((n, N_ACTIONS))
    rows = np.repeat(np.arange(n), N_BRANCHES)
    for a in range(N_ACTIONS):
        nxt = branch_next[:, a, :].ravel()
        prob = np.tile(branch_prob[a], n)
        inside = nxt >= 0
        P.append(
            sparse.csr_matrix((prob[inside], (rows[inside], nxt[inside])), shape=(n, n))
        )  # duplicate entries are summed
        p_success[:, a] = (branch_prob[a] * (branch_next[:, a, :] == NEXT_SUCCESS)).sum(axis=1)
        p_fail[:, a] = (branch_prob[a] * (branch_next[:, a, :] == NEXT_FAIL)).sum(axis=1)
    cost = (branch_prob[None, :, :] * branch_cost).sum(axis=2)

    for arr in (
        cells,
        index,
        branch_prob,
        branch_next,
        branch_cell,
        branch_cost,
        p_success,
        p_fail,
        cost,
    ):
        arr.flags.writeable = False
    return SparseMDP(
        spec=spec,
        cells=cells,
        index=index,
        branch_prob=branch_prob,
        branch_next=branch_next,
        branch_cell=branch_cell,
        branch_cost=branch_cost,
        P=tuple(P),
        p_success=p_success,
        p_fail=p_fail,
        cost=cost,
    )
