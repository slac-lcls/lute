"""Task for CPU-based crystal lattice indexing.

This module contains a completely self-contained implementation of the crystallographic
indexer. All functions are embedded directly - no external dependencies on indexer_cpu.py.

Classes:
    IndexerCPU: Perform crystallographic indexing using differential evolution.
"""

__all__ = ["IndexerCPU"]
__author__ = "Marc Nasser"

import math
from functools import partial
from typing import cast, Dict, Any, Tuple, List, Optional

import numpy as np
import gemmi
from numba import njit
from scipy.optimize import differential_evolution

from lute.execution.ipc import Message
from lute.io.models.indexer_cpu import IndexerCPUParameters
from lute.tasks.task import Task

# ==============================================================================
# Quaternion Operations
# ==============================================================================


def quat_normalize(q):
    """Normalize a quaternion to unit length."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return q / n


def quat_to_R(q):
    """Convert a single quaternion to a 3x3 rotation matrix."""
    q = quat_normalize(q)
    q0, q1, q2, q3 = q
    return np.array(
        [
            [
                1 - 2 * (q2 * q2 + q3 * q3),
                2 * (q1 * q2 - q0 * q3),
                2 * (q1 * q3 + q0 * q2),
            ],
            [
                2 * (q1 * q2 + q0 * q3),
                1 - 2 * (q1 * q1 + q3 * q3),
                2 * (q2 * q3 - q0 * q1),
            ],
            [
                2 * (q1 * q3 - q0 * q2),
                2 * (q2 * q3 + q0 * q1),
                1 - 2 * (q1 * q1 + q2 * q2),
            ],
        ],
        dtype=np.float64,
    )


@njit(fastmath=True)
def _quat_to_R_batch_core(quats):
    """JIT-compiled core function for batch quaternion to rotation matrix conversion."""
    B = quats.shape[0]
    R = np.empty((B, 3, 3), np.float64)

    for i in range(B):
        w = quats[i, 0]
        x = quats[i, 1]
        y = quats[i, 2]
        z = quats[i, 3]

        n = math.sqrt(w * w + x * x + y * y + z * z)
        if n > 0.0:
            w /= n
            x /= n
            y /= n
            z /= n

        xx = x * x
        yy = y * y
        zz = z * z
        wx = w * x
        wy = w * y
        wz = w * z
        xy = x * y
        xz = x * z
        yz = y * z

        R[i, 0, 0] = 1.0 - 2.0 * (yy + zz)
        R[i, 0, 1] = 2.0 * (xy - wz)
        R[i, 0, 2] = 2.0 * (xz + wy)

        R[i, 1, 0] = 2.0 * (xy + wz)
        R[i, 1, 1] = 1.0 - 2.0 * (xx + zz)
        R[i, 1, 2] = 2.0 * (yz - wx)

        R[i, 2, 0] = 2.0 * (xz - wy)
        R[i, 2, 1] = 2.0 * (yz + wx)
        R[i, 2, 2] = 1.0 - 2.0 * (xx + yy)

    return R


def quat_to_R_batch(quats):
    """Convert batch of quaternions to rotation matrices."""
    q = np.asarray(quats, dtype=np.float64)
    if q.ndim == 1:
        if q.shape[0] != 4:
            raise ValueError(f"Expected quaternion of length 4, got shape {q.shape}")
        q = q[None, :]
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quat_to_R_batch expects shape (B,4) or (4,), got {q.shape}")
    return _quat_to_R_batch_core(q)


def R_to_quat(R):
    """Convert a 3x3 rotation matrix to a quaternion."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t + 1.0) * 2.0
        q0 = 0.25 * S
        q1 = (R[2, 1] - R[1, 2]) / S
        q2 = (R[0, 2] - R[2, 0]) / S
        q3 = (R[1, 0] - R[0, 1]) / S
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q0 = (R[2, 1] - R[1, 2]) / S
            q1 = 0.25 * S
            q2 = (R[0, 1] + R[1, 0]) / S
            q3 = (R[0, 2] + R[2, 0]) / S
        elif i == 1:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q0 = (R[0, 2] - R[2, 0]) / S
            q1 = (R[0, 1] + R[1, 0]) / S
            q2 = 0.25 * S
            q3 = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q0 = (R[1, 0] - R[0, 1]) / S
            q1 = (R[0, 2] + R[2, 0]) / S
            q2 = (R[1, 2] + R[2, 1]) / S
            q3 = 0.25 * S
    return quat_normalize(np.array([q0, q1, q2, q3], dtype=np.float64))


# ==============================================================================
# Lattice Reduction and Basis Operations
# ==============================================================================


def _shortest_vector_enum_vec(B, Hmax=6):
    """Find shortest lattice vector using enumeration over a grid."""
    r = np.arange(-Hmax, Hmax + 1)
    H = np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3)
    H = H[np.any(H != 0, axis=1)]
    V = B @ H.T
    n2 = np.einsum("ij,ij->j", V, V)
    k = int(np.argmin(n2))
    return H[k], V[:, k], float(n2[k])


def _gauss_reduce_2d(B2):
    """Perform 2D Gaussian lattice reduction."""
    G = B2.copy().astype(np.float64)
    U2 = np.eye(2, dtype=int)
    while True:
        denom = max(np.dot(G[:, 0], G[:, 0]), 1e-18)
        mu = np.dot(G[:, 0], G[:, 1]) / denom
        k = int(np.rint(mu))
        if k != 0:
            G[:, 1] -= k * G[:, 0]
            U2[:, 1] -= k * U2[:, 0]
        if np.dot(G[:, 1], G[:, 1]) < np.dot(G[:, 0], G[:, 0]) - 1e-18:
            G[:, [0, 1]] = G[:, [1, 0]]
            U2[:, [0, 1]] = U2[:, [1, 0]]
        else:
            break
    return G, U2


def kz_reduce_integer_3d(B, Hmax=6):
    """Perform Korkine-Zolotarev reduction on a 3D integer lattice basis."""
    B = np.asarray(B, np.float64)
    h1, b1, _ = _shortest_vector_enum_vec(B, Hmax=Hmax)
    U = np.eye(3, dtype=int)
    U[:, 0] = h1
    if np.linalg.matrix_rank(U) < 3:
        U[:, 1] = np.array([0, 1, 0])
        U[:, 2] = np.array([0, 0, 1])
        if np.linalg.matrix_rank(U) < 3:
            U[:, 1] = np.array([1, 0, 0])
            U[:, 2] = np.array([0, 1, 0])
    for j in (1, 2):
        b_j = B @ U[:, j]
        denom = max(np.dot(b1, b1), 1e-18)
        q = int(np.rint(np.dot(b_j, b1) / denom))
        U[:, j] -= q * U[:, 0]
    B1 = B @ U
    G2, U2 = _gauss_reduce_2d(B1[:, 1:3])
    U[:, 1:3] = U[:, 1:3] @ U2
    B_kz = B @ U
    return B_kz, U


@njit(fastmath=True)
def _babai_nearest_batched_core(R_B, M):
    """JIT-compiled Babai nearest plane algorithm for batch lattice rounding."""
    r00 = R_B[0, 0]
    r11 = R_B[1, 1]
    r22 = R_B[2, 2]
    r01 = R_B[0, 1]
    r02 = R_B[0, 2]
    r12 = R_B[1, 2]

    K = M.shape[0]
    N = M.shape[2]
    H = np.empty((K, 3, N), np.int64)

    for k in range(K):
        for n in range(N):
            y2 = M[k, 2, n]
            h2 = int(np.rint(y2 / r22))

            y1 = M[k, 1, n] - r12 * h2
            y0 = M[k, 0, n] - r02 * h2

            h1 = int(np.rint(y1 / r11))
            y0 = y0 - r01 * h1

            h0 = int(np.rint(y0 / r00))

            H[k, 0, n] = h0
            H[k, 1, n] = h1
            H[k, 2, n] = h2

    return H


def babai_nearest_batched(R_B, M):
    """Batch lattice rounding using Babai nearest plane algorithm."""
    R_B = np.asarray(R_B, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 3 or M.shape[1] != 3:
        raise ValueError(
            f"babai_nearest_batched expects M with shape (K,3,N), got {M.shape}"
        )
    return _babai_nearest_batched_core(R_B, M)


# ==============================================================================
# Crystal Symmetry Operations
# ==============================================================================


def get_rots_mats_symm(unit_cell):
    """Determine lattice symmetry operations for a unit cell."""
    uc = gemmi.UnitCell(*unit_cell)
    ops = gemmi.find_lattice_symmetry(uc, "P", 3.0)
    B = np.array(uc.frac.mat).T
    B_inv = np.array(uc.orth.mat).T
    rot_list = []
    for op in ops:
        R_frac = np.array(op.rot, dtype=np.float64) / op.DEN
        R_recip = B @ R_frac.T @ B_inv
        rot_list.append(R_recip.astype(np.float64))
    if not rot_list:
        rot_list = [np.eye(3, dtype=np.float64)]
    return np.stack(rot_list, axis=0)


# ==============================================================================
# Loss Functions
# ==============================================================================


def loss_trimmed_batched(
    quats, Q, B_red, Q_B_T, R_B, S_ops_stack, PB_ops_stack, kappa=0.40
):
    """Compute trimmed mean loss for batch of quaternions with symmetry."""
    quats = np.ascontiguousarray(quats, dtype=np.float64)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12

    Rk = quat_to_R_batch(quats)
    A = np.einsum("bji,jn->bin", Rk, Q, optimize=True)

    Bsz, _, N = A.shape
    S = S_ops_stack.shape[0]

    A_sym = np.einsum("sij,bjn->sbin", S_ops_stack, A, optimize=True)
    A_resh = A_sym.reshape(S * Bsz, 3, N)
    H_sym = babai_nearest_batched(R_B, A_resh).reshape(S, Bsz, 3, N)
    Y_sym = np.einsum("sij,sbjn->sbin", PB_ops_stack, H_sym, optimize=True)

    diff = A[None, :, :, :] - Y_sym
    r2 = np.einsum("sbin,sbin->sbn", diff, diff, optimize=True)
    r2_min = np.min(r2, axis=0)

    h = int(np.ceil(kappa * N))
    part = np.partition(r2_min, h - 1, axis=1)[:, :h]
    t2_vec = np.quantile(part, 0.90, axis=1)
    clipped = np.minimum(part, t2_vec[:, None])
    out = clipped.mean(axis=1)
    return out


def loss_mse_batched(quats, Q, B_red, Q_B_T, R_B, S_ops_stack, PB_ops_stack):
    """Compute mean squared error loss for batch of quaternions with symmetry."""
    quats = np.ascontiguousarray(quats, dtype=np.float64)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12
    Q_sq_mean = float(np.mean(np.sum(Q * Q, axis=0)))

    Rk = quat_to_R_batch(quats)
    A = np.einsum("bji,jn->bin", Rk, Q, optimize=True)

    best = np.full(len(quats), np.inf, dtype=np.float64)
    for s in range(S_ops_stack.shape[0]):
        A_sym = np.einsum("ij,bjn->bin", S_ops_stack[s], A, optimize=True)
        Hp_sym = babai_nearest_batched(R_B, A_sym)
        Y_sym = np.einsum("ij,bjn->bin", PB_ops_stack[s], Hp_sym, optimize=True)
        YY_mean = np.mean(np.sum(Y_sym * Y_sym, axis=1), axis=1)
        AY_mean = np.mean(np.sum(A * Y_sym, axis=1), axis=1)
        val = Q_sq_mean + YY_mean - 2.0 * AY_mean
        best = np.minimum(best, val)
    return best


def loss_small_n_batched(
    quats, Q, B_red, Q_B_T, R_B, S_ops_stack, PB_ops_stack, D, PB_D, kappa=0.40, delta=1
):
    """Loss function with target-space neighbor search for small datasets."""
    quats = np.ascontiguousarray(quats, dtype=np.float64)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True) + 1e-12
    Rk = quat_to_R_batch(quats)
    A = np.einsum("bji,jn->bin", Rk, Q, optimize=True)
    Bsz, _, N = A.shape
    S = S_ops_stack.shape[0]
    A_sym = np.einsum("sij,bjn->sbin", S_ops_stack, A, optimize=True)
    A_resh = A_sym.reshape(S * Bsz, 3, N)
    H_babai = babai_nearest_batched(R_B, A_resh).reshape(S, Bsz, 3, N)
    Y0 = np.einsum("sij,sbjn->sbin", PB_ops_stack, H_babai, optimize=True)
    E = A[None, :, :, :] - Y0
    E2 = np.einsum("sbin,sbin->sbn", E, E, optimize=True)
    dY2 = np.einsum("sik,sik->sk", PB_D, PB_D, optimize=True)
    dot = np.einsum("sbin,sik->sbkn", E, PB_D, optimize=True)
    r2_k = E2[:, :, None, :] + dY2[:, None, :, None] - 2.0 * dot
    r2_best_per_sym = np.min(r2_k, axis=2)
    r2_min = np.min(r2_best_per_sym, axis=0)
    Bsz, N = r2_min.shape
    h = int(np.ceil(kappa * N))
    part = np.partition(r2_min, h - 1, axis=1)[:, :h]
    out = part.mean(axis=1).astype(np.float64)
    return out


# ==============================================================================
# Polishing and Refinement
# ==============================================================================


def polish_R_smallN_tsn(
    R_init,
    Q,
    consts,
    *,
    kappa=0.40,
    delta=1,
    winsor_q=None,
    fit_alpha=True,
    alpha_clip=(0.97, 1.03),
    steps=2,
    indices=None,
):
    """Polish rotation matrix using target-space neighbor search (TSN)."""
    Q = np.asarray(Q, np.float64)
    N = Q.shape[1]
    I = np.arange(N, dtype=int) if indices is None else np.asarray(indices, dtype=int)
    h = int(np.ceil(kappa * len(I)))

    R_B = consts["R_B"]
    S_ops = consts["S_ops_stack"]
    PB_ops = consts["PB_ops_stack"]
    S = S_ops.shape[0]

    R = np.asarray(R_init, np.float64)
    alpha = 1.0
    H_out = None
    s_best = 0

    for _ in range(max(1, int(steps))):
        A = R.T @ Q
        A_I = A[:, I]
        A_sym_I = np.einsum("sij,jn->sin", S_ops, A_I, optimize=True)
        H_babai_I = babai_nearest_batched(R_B, A_sym_I)
        Y0_I = np.einsum("sij,sjn->sin", PB_ops, H_babai_I, optimize=True)
        E_I = A_sym_I - Y0_I
        E2_I = np.einsum("sin,sin->sn", E_I, E_I, optimize=True)

        if delta > 0:
            D_tsn = consts.get("D_tsn", None)
            PB_D = consts.get("PB_D", None)
            dY2 = consts.get("dY2", None)
            if (
                D_tsn is None
                or PB_D is None
                or dY2 is None
                or consts.get("delta_tsn", None) != delta
            ):
                offs = np.arange(-delta, delta + 1, dtype=int)
                D_tsn = (
                    np.stack(np.meshgrid(offs, offs, offs, indexing="ij"), axis=-1)
                    .reshape(-1, 3)
                    .T
                )
                PB_D = np.einsum("sij,jk->sik", PB_ops, D_tsn, optimize=True)
                dY2 = np.einsum("sik,sik->sk", PB_D, PB_D, optimize=True)

            dot = np.einsum("sin,sik->skn", E_I, PB_D, optimize=True)
            r2_k = E2_I[:, None, :] + dY2[:, :, None] - 2.0 * dot
            k_star_I = np.argmin(r2_k, axis=1)
            r2_s_I = np.min(r2_k, axis=1)
        else:
            k_star_I = None
            r2_s_I = E2_I

        S_, M = r2_s_I.shape
        assert S_ == S

        h = int(np.ceil(kappa * len(I)))
        part = np.partition(r2_s_I, h - 1, axis=1)[:, :h]

        if winsor_q is not None and 0.0 < winsor_q < 1.0 and h >= 3:
            t2_vec = np.quantile(part, winsor_q, axis=1)
            clipped_all = np.minimum(part, t2_vec[:, None])
        else:
            clipped_all = part

        loss_s = clipped_all.mean(axis=1)
        s_best = int(np.argmin(loss_s))

        r2_best = r2_s_I[s_best]
        idx_h_sub = np.argpartition(r2_best, h - 1)[:h]
        sel = r2_best[idx_h_sub]
        if winsor_q is not None and 0.0 < winsor_q < 1.0 and h >= 3:
            t2 = float(np.quantile(sel, winsor_q))
            sel = np.minimum(sel, t2)
        best_loss = float(np.mean(sel))

        idx_h = I[idx_h_sub]

        A_sym_all = S_ops[s_best] @ A
        H_all = babai_nearest_batched(R_B, A_sym_all[None, ...])[0]

        if delta > 0:
            offs = np.arange(-delta, delta + 1, dtype=int)
            D = (
                np.stack(np.meshgrid(offs, offs, offs, indexing="ij"), axis=-1)
                .reshape(-1, 3)
                .T
            )
            ks = k_star_I[s_best]
            H_all[:, I] = (H_all[:, I] + D[:, ks]).astype(int)

        Y_all = PB_ops[s_best] @ H_all

        if fit_alpha:
            num = float(np.sum(Q[:, idx_h] * (R @ Y_all[:, idx_h])))
            den = float(np.sum(Y_all[:, idx_h] * Y_all[:, idx_h])) + 1e-18
            alpha = num / den
            lo, hi = alpha_clip
            alpha = float(np.minimum(hi, np.maximum(lo, alpha)))
        else:
            alpha = 1.0

        Sxy = Q[:, idx_h] @ (alpha * Y_all[:, idx_h]).T
        U, _, Vt = np.linalg.svd(Sxy, full_matrices=False)
        R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt

        H_out = H_all

    return R, H_out, s_best, alpha


def hardest_peak_indices(R, Q, consts, K):
    """Identify the K peaks with largest indexing residuals."""
    S_ops, PB_ops, R_B = consts["S_ops_stack"], consts["PB_ops_stack"], consts["R_B"]
    A = R.T @ Q
    A_sym = np.einsum("sij,jn->sin", S_ops, A, optimize=True)
    H_b = babai_nearest_batched(R_B, A_sym)
    Y0 = np.einsum("sij,sjn->sin", PB_ops, H_b, optimize=True)
    diff = A[None, :, :] - Y0
    r2 = np.einsum("sin,sin->sn", diff, diff, optimize=True)
    r2_min = np.min(r2, axis=0)
    K = int(min(K, r2_min.shape[0]))
    return np.argpartition(r2_min, -K)[-K:]


# ==============================================================================
# Setup and Preparation
# ==============================================================================


def prepare_constants(B, unit_cell, Hmax=6, delta_tsn=1):
    """Precompute all constants needed for efficient indexing."""
    B = np.ascontiguousarray(B, dtype=np.float64)
    B_red, U = kz_reduce_integer_3d(B, Hmax=Hmax)
    Q_B, R_B = np.linalg.qr(B_red)
    s = np.sign(np.diag(R_B))
    s[s == 0] = 1
    D = np.diag(s)
    Q_B = Q_B @ D
    R_B = D @ R_B
    Q_B_T = Q_B.T
    rot_mats = get_rots_mats_symm(unit_cell)
    order = np.argsort([float(np.trace(R)) for R in rot_mats])[::-1]
    rot_mats = rot_mats[order]
    S_ops_stack = np.stack([Q_B_T @ R_sym.T for R_sym in rot_mats], axis=0)
    PB_ops_stack = np.stack([R_sym @ B_red for R_sym in rot_mats], axis=0)
    if delta_tsn is not None and delta_tsn > 0:
        offs = np.arange(-delta_tsn, delta_tsn + 1, dtype=int)
        D_tsn = (
            np.stack(np.meshgrid(offs, offs, offs, indexing="ij"), axis=-1)
            .reshape(-1, 3)
            .T
        )
        PB_D = np.einsum("sij,jk->sik", PB_ops_stack, D_tsn, optimize=True)
        dY2 = np.einsum("sik,sik->sk", PB_D, PB_D, optimize=True)
    else:
        D_tsn, PB_D, dY2 = None, None, None
    return dict(
        B_red=B_red,
        U=U,
        Q_B_T=Q_B_T,
        R_B=R_B,
        S_ops_stack=S_ops_stack,
        PB_ops_stack=PB_ops_stack,
        rot_mats=rot_mats,
        delta_tsn=delta_tsn,
        D_tsn=D_tsn,
        PB_D=PB_D,
        dY2=dY2,
    )


# ==============================================================================
# Main Optimization Function
# ==============================================================================


def global_optimize_via_de_prepared(
    Q,
    consts,
    obj="mse_symm_trimmed_auto",
    kappa=0.40,
    tol=3e-4,
    maxiter=1500,
    popsize=21,
    polish=True,
    updating="deferred",
    workers=1,
    strategy="randtobest1bin",
    smallN_tsn_polish=True,
    smallN_threshold=25,
    callback_every=25,
    tsn_delta=1,
    tsn_steps=1,
    tsn_winsor_q=None,
    tsn_fit_alpha=False,
    midN_threshold=80,
    midN_tsn_topK=48,
    callback_every_largeN=75,
    **kwargs,
):
    """Global optimization of crystal orientation using differential evolution."""
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    B_red, U, Q_B_T, R_B = consts["B_red"], consts["U"], consts["Q_B_T"], consts["R_B"]
    S_ops_stack, PB_ops_stack = consts["S_ops_stack"], consts["PB_ops_stack"]
    rot_mats = consts["rot_mats"]
    N = Q.shape[1]
    use_smallN = smallN_tsn_polish and (N <= smallN_threshold)
    use_midN = (N > smallN_threshold) and (N <= midN_threshold)

    if obj == "mse_symm_trimmed_auto":
        batched_loss = partial(
            loss_trimmed_batched,
            Q=Q,
            B_red=B_red,
            Q_B_T=Q_B_T,
            R_B=R_B,
            S_ops_stack=S_ops_stack,
            PB_ops_stack=PB_ops_stack,
            kappa=kappa,
        )
    elif obj == "mse_symm":
        batched_loss = partial(
            loss_mse_batched,
            Q=Q,
            B_red=B_red,
            Q_B_T=Q_B_T,
            R_B=R_B,
            S_ops_stack=S_ops_stack,
            PB_ops_stack=PB_ops_stack,
        )
    elif obj == "mse_small_n":
        D_tsn = consts["D_tsn"]
        PB_D = consts["PB_D"]
        batched_loss = partial(
            loss_small_n_batched,
            Q=Q,
            B_red=B_red,
            Q_B_T=Q_B_T,
            R_B=R_B,
            S_ops_stack=S_ops_stack,
            PB_ops_stack=PB_ops_stack,
            D=D_tsn,
            PB_D=PB_D,
            kappa=kappa,
        )
    else:
        raise ValueError(f"Unknown objective type: {obj}")

    init = "latinhypercube"

    def objective_vec(pop):
        pop = np.asarray(pop, dtype=np.float64)
        if pop.ndim == 1:
            pop = pop[None, :]
        elif pop.ndim == 2:
            if pop.shape[1] == 4:
                pass
            elif pop.shape[0] == 4:
                pop = pop.T
            else:
                raise AssertionError(
                    f"Unexpected shape {pop.shape}; expected (NP,4) or (4,NP)"
                )
        else:
            raise AssertionError(f"Unexpected ndim {pop.ndim}")
        vals = batched_loss(pop)
        return vals

    best_q_shadow = None
    best_val_shadow = np.inf
    gen_counter = {"g": 0}
    best_tracker = {
        "best_val": np.inf,
        "last_improve_gen": 0,
    }
    patience = 600
    target_loss = 1e-7

    def de_callback(xk, convergence):
        """Callback function called during DE iterations for refinement and early stopping."""
        gen_counter["g"] += 1

        if use_smallN:
            if (callback_every is None) or (
                gen_counter["g"] % max(1, int(callback_every)) != 0
            ):
                return False
        elif use_midN:
            if (callback_every_largeN is None) or (
                gen_counter["g"] % max(1, int(callback_every_largeN)) != 0
            ):
                return False
        else:
            if (callback_every_largeN is None) or (
                gen_counter["g"] % max(1, int(callback_every_largeN)) != 0
            ):
                return False
        qk = np.asarray(xk, dtype=np.float64)
        qk /= np.linalg.norm(qk) + 1e-12
        valk = float(batched_loss(qk[None, :])[0])
        Rk = quat_to_R(qk)
        candidates = [(qk, valk)]

        if use_smallN:
            for fit_alpha in (False, tsn_fit_alpha):
                Rp, _, _, _ = polish_R_smallN_tsn(
                    Rk,
                    Q,
                    consts,
                    kappa=kappa,
                    delta=tsn_delta,
                    winsor_q=tsn_winsor_q,
                    fit_alpha=fit_alpha,
                    steps=tsn_steps,
                    indices=None,
                )
                qp = R_to_quat(Rp)
                vp = float(batched_loss(qp[None, :])[0])
                candidates.append((qp, vp))
        elif use_midN:
            Ihard = hardest_peak_indices(Rk, Q, consts, midN_tsn_topK)
            for fit_alpha in (False, tsn_fit_alpha):
                Rp, _, _, _ = polish_R_smallN_tsn(
                    Rk,
                    Q,
                    consts,
                    kappa=kappa,
                    delta=tsn_delta,
                    winsor_q=None,
                    fit_alpha=fit_alpha,
                    steps=2,
                    indices=Ihard,
                )
                qp = R_to_quat(Rp)
                vp = float(batched_loss(qp[None, :])[0])
                candidates.append((qp, vp))
        else:
            Rp, _, _, _ = polish_R_smallN_tsn(
                Rk,
                Q,
                consts,
                kappa=kappa,
                delta=0,
                winsor_q=None,
                fit_alpha=tsn_fit_alpha,
                steps=1,
                indices=None,
            )
            qp = R_to_quat(Rp)
            vp = float(batched_loss(qp[None, :])[0])
            candidates.append((qp, vp))
        best_local_q, best_local_v = min(candidates, key=lambda t: t[1])
        nonlocal best_q_shadow, best_val_shadow
        if best_local_v < best_tracker["best_val"] - 1e-9:
            best_tracker["best_val"] = best_local_v
            best_tracker["last_improve_gen"] = gen_counter["g"]
        if best_local_v < best_val_shadow - 1e-12:
            best_q_shadow, best_val_shadow = best_local_q, best_local_v

        if (target_loss is not None) and (best_tracker["best_val"] <= target_loss):
            return True
        if gen_counter["g"] - best_tracker["last_improve_gen"] >= patience:
            return True
        return False

    result = differential_evolution(
        objective_vec,
        bounds=[(-1, 1)] * 4,
        tol=tol,
        maxiter=maxiter,
        popsize=popsize,
        polish=polish,
        updating=updating,
        workers=workers,
        vectorized=True,
        strategy=strategy,
        init=init,
        callback=de_callback,
        recombination=0.7,
        mutation=(0.5, 1),
    )

    q_best_de = np.array(result.x, dtype=np.float64)
    q_best_de /= np.linalg.norm(q_best_de) + 1e-12
    val_best_de = float(batched_loss(q_best_de[None, :])[0])
    if best_q_shadow is not None and best_val_shadow < val_best_de - 1e-12:
        q_best = best_q_shadow
        best_val = best_val_shadow
    else:
        q_best = q_best_de
        best_val = val_best_de

    R_best = quat_to_R(q_best)
    M_final = Q_B_T @ (R_best.T @ Q)
    Hp_final = babai_nearest_batched(R_B, M_final[None, :, :])[0]
    H_best = consts["U"] @ Hp_final
    return R_best, H_best, best_val, result.nit, result.message, result.success


def crystfel_check_single_lattice(Q, R, B_inv, delta=0.25):
    """CrystFEL-style lattice check."""
    n_feat = int(Q.shape[0])
    if n_feat == 0:
        return False, 0.0, 0, 0

    U = (B_inv @ (R.T @ Q.T)).T
    D = np.abs(U - np.rint(U))
    ok = np.all(D < delta, axis=1)

    n_sane = int(ok.sum())
    frac = n_sane / n_feat
    return (frac >= 0.4), frac, n_sane, n_feat


def unit_cell_to_B_and_Binv(unit_cell):
    """Convert unit cell to basis matrices."""
    uc = gemmi.UnitCell(*unit_cell)
    B = np.array(uc.frac.mat).T.astype(np.float64)
    B_inv = np.array(uc.orth.mat).T.astype(np.float64)
    return B, B_inv


def run_indexing_with_retries(
    Q,
    unit_cell,
    kappas,
    strategies,
    *,
    obj,
    n_tries=6,
    delta=0.25,
    tol=3e-2,
    maxiter=1500,
    popsize=21,
    polish=False,
    updating="deferred",
    workers=1,
    **kwargs,
):
    """Run indexing with multiple retry attempts."""
    if n_tries <= 0:
        raise ValueError("n_tries must be positive")
    if len(kappas) != n_tries:
        raise ValueError("Number of kappas must equal n_tries")
    if len(strategies) != n_tries:
        raise ValueError("Number of strategies must equal n_tries")
    if len(obj) != n_tries:
        raise ValueError("Number of obj entries must equal n_tries")

    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[1] != 3:
        raise ValueError(f"Q must have shape (N,3), got {Q.shape}")

    Q_rows = Q
    Q_cols = Q.T

    B, B_inv = unit_cell_to_B_and_Binv(unit_cell)
    consts = prepare_constants(B, unit_cell)

    attempts = []

    for try_idx in range(n_tries):
        kappa = float(kappas[try_idx])
        strategy = str(strategies[try_idx])
        obj_c = str(obj[try_idx])

        R_est, H_est, err, nit, message, de_success = global_optimize_via_de_prepared(
            Q_cols,
            consts,
            obj=obj_c,
            kappa=kappa,
            tol=tol,
            maxiter=maxiter,
            popsize=popsize,
            polish=polish,
            updating=updating,
            workers=workers,
            strategy=strategy,
            **kwargs,
        )

        passed, frac, n_sane, n_feat = crystfel_check_single_lattice(
            Q_rows, R_est, B_inv, delta=delta
        )

        rec = {
            "try_idx": try_idx,
            "kappa": kappa,
            "strategy": strategy,
            "passed": bool(passed),
            "frac": float(frac),
            "n_sane": int(n_sane),
            "n_feat": int(n_feat),
            "err": float(err),
            "nit": int(nit),
            "message": str(message),
            "de_success": bool(de_success),
            "R_est": R_est.astype(np.float64),
            "H_est": H_est.astype(np.int32),
        }
        attempts.append(rec)

        print(
            f"[try {try_idx + 1}/{n_tries}] "
            f"strategy={strategy} kappa={kappa:.6f} "
            f"passed={passed} frac={frac:.4f} "
            f"n_sane={n_sane}/{n_feat} err={err:.6e} "
            f"nit={nit} de_success={de_success}"
        )

        if passed:
            return {
                "accepted": True,
                "accepted_try": try_idx,
                "result": rec,
                "attempts": attempts,
            }

    return {
        "accepted": False,
        "accepted_try": None,
        "result": attempts[-1] if attempts else None,
        "attempts": attempts,
    }


# ==============================================================================
# LUTE Task Class
# ==============================================================================


class IndexerCPU(Task):
    """CPU-based crystal lattice indexer using differential evolution.

    This task performs crystallographic indexing of diffraction peaks to determine
    crystal orientation. It uses quaternion-based rotations, lattice reduction,
    crystal symmetry, and differential evolution optimization.

    All indexing functions are self-contained in this module - no external dependencies.
    """

    def __init__(self, *, params: IndexerCPUParameters) -> None:
        super().__init__(params=params)

    def _run(self) -> None:
        """Execute the indexing algorithm."""
        self._task_parameters = cast(IndexerCPUParameters, self._task_parameters)

        # Load Q vectors
        msg: Message = Message(
            contents=f"Loading Q vectors from {self._task_parameters.q_path}"
        )
        self._report_to_executor(msg)

        Q: np.ndarray = np.load(self._task_parameters.q_path)

        msg = Message(
            contents=(
                f"Loaded {Q.shape[0]} peaks. "
                f"Unit cell: {self._task_parameters.unit_cell}"
            )
        )
        self._report_to_executor(msg)

        # Run indexing with retry schedule
        msg = Message(
            contents=(
                f"Starting indexing with {self._task_parameters.n_tries} attempts. "
                f"Kappas: {self._task_parameters.kappas}, "
                f"Strategies: {self._task_parameters.strategies}"
            )
        )
        self._report_to_executor(msg)

        result: Dict[str, Any] = run_indexing_with_retries(
            Q=Q,
            unit_cell=self._task_parameters.unit_cell,
            kappas=self._task_parameters.kappas,
            strategies=self._task_parameters.strategies,
            obj=self._task_parameters.obj,
            n_tries=self._task_parameters.n_tries,
            delta=self._task_parameters.delta,
            tol=self._task_parameters.tol,
            maxiter=self._task_parameters.maxiter,
            popsize=self._task_parameters.popsize,
            polish=self._task_parameters.polish,
            updating=self._task_parameters.updating,
            workers=self._task_parameters.workers,
            init_mode=self._task_parameters.init_mode,
            seed_factor=0,
            niche_radius_deg=self._task_parameters.niche_radius_deg,
            elite_per_niche=self._task_parameters.elite_per_niche,
            immigrants_frac=self._task_parameters.immigrants_frac,
            jitter_floor_deg=self._task_parameters.jitter_floor_deg,
            jitter_scale=self._task_parameters.jitter_scale,
            callback_every=self._task_parameters.callback_every,
        )

        # Report results
        if result["accepted"]:
            rec = result["result"]
            msg = Message(
                contents=(
                    f"Indexing SUCCEEDED on attempt {rec['try_idx'] + 1}/"
                    f"{self._task_parameters.n_tries}: "
                    f"strategy={rec['strategy']}, kappa={rec['kappa']:.4f}, "
                    f"fraction indexed={rec['frac']:.4f} "
                    f"({rec['n_sane']}/{rec['n_feat']} peaks)"
                )
            )
            self._report_to_executor(msg)

            # Save results
            save_prefix: str = self._task_parameters.save_prefix
            np.save(f"{save_prefix}_U.npy", rec["R_est"], allow_pickle=False)
            np.save(f"{save_prefix}_H.npy", rec["H_est"], allow_pickle=False)

            msg = Message(
                contents=(
                    f"Saved rotation matrix to {save_prefix}_U.npy and "
                    f"Miller indices to {save_prefix}_H.npy"
                )
            )
            self._report_to_executor(msg)

        else:
            # Indexing failed all attempts
            rec = result["result"]
            msg = Message(
                contents=(
                    f"Indexing FAILED after {self._task_parameters.n_tries} attempts. "
                    f"Best result: frac={rec['frac']:.4f} "
                    f"({rec['n_sane']}/{rec['n_feat']} peaks) "
                    f"with strategy={rec['strategy']}, kappa={rec['kappa']:.4f}"
                )
            )
            self._report_to_executor(msg)

            # Still save the best attempt
            save_prefix = self._task_parameters.save_prefix
            if rec is not None:
                np.save(f"{save_prefix}_U.npy", rec["R_est"], allow_pickle=False)
                np.save(f"{save_prefix}_H.npy", rec["H_est"], allow_pickle=False)

        # Write attempt log
        save_prefix = self._task_parameters.save_prefix
        with open(f"{save_prefix}_attempts.txt", "w") as f:
            for rec_i in result["attempts"]:
                f.write(
                    f"try={rec_i['try_idx'] + 1} "
                    f"strategy={rec_i['strategy']} "
                    f"kappa={rec_i['kappa']:.6f} "
                    f"passed={int(rec_i['passed'])} "
                    f"frac={rec_i['frac']:.6f} "
                    f"n_sane={rec_i['n_sane']} "
                    f"n_feat={rec_i['n_feat']} "
                    f"err={rec_i['err']:.12e} "
                    f"nit={rec_i['nit']} "
                    f"de_success={int(rec_i['de_success'])} "
                    f"message={rec_i['message']}\n"
                )

        msg = Message(contents=f"Wrote attempt log to {save_prefix}_attempts.txt")
        self._report_to_executor(msg)
