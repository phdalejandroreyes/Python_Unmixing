
import numpy as np
from scipy.linalg import pinv, svd, sqrtm
from joblib import Parallel, delayed
from unmixing.endmember_extraction import vca, SVMAX, NFINDR


def initPo(Yo, Ym, initcond, N):
    L = Yo.shape[0]
    Po = np.zeros((L, N), dtype=np.float64)

    if initcond == 1:
        Yt = Ym.copy()
        Po[:, 0] = np.mean(Ym, axis=1)

        for index in range(N - 1):
            Pn = Po[:, :index + 1]
            num = Yt.T @ Pn
            den = np.linalg.norm(Yt, axis=0)[:, None] * np.linalg.norm(Pn, axis=0)[None, :]
            e = num / (den + 1e-12)
            scores = np.min(np.abs(e), axis=1)

            duplicate = np.zeros(Yt.shape[1], dtype=bool)
            for j in range(index + 1):
                duplicate |= np.all(Yt == Po[:, j:j + 1], axis=0)

            scores[duplicate] = np.inf
            idx = np.argmin(scores)
            Po[:, index + 1] = Yt[:, idx]
            Yt = np.delete(Yt, idx, axis=1)

    elif initcond == 2:
        energy = np.sum(np.abs(Yo), axis=0)
        Imax, Imin = np.argmax(energy), np.argmin(energy)
        Po[:, 0], Po[:, 1] = Yo[:, Imax], Yo[:, Imin]
        Yt = np.delete(Yo, [Imax, Imin], axis=1)

        for index in range(1, N - 1):
            Pn = Po[:, :index + 1]
            num = Yt.T @ Pn
            den = np.linalg.norm(Yt, axis=0)[:, None] * np.linalg.norm(Pn, axis=0)[None, :]
            e = num / np.maximum(den, 1e-12)
            idx = np.argmin(np.min(np.abs(e), axis=1))
            Po[:, index + 1] = Yt[:, idx]
            Yt = np.delete(Yt, idx, axis=1)

    elif initcond == 3:
        _, _, V = svd(Ym.T, full_matrices=False)
        W = V[:N].T
        Po = W * np.sign(W.T @ np.ones((L, 1))).T

    elif initcond == 4:
        Yom = np.mean(Ym, axis=1, keepdims=True)
        Yon = Ym - Yom
        _, S, V = svd(Yon.T, full_matrices=False)
        Yo_w = pinv(sqrtm(np.diag(S))) @ V @ Ym.T
        U, _, _ = svd((Yo_w ** 2).sum(axis=1, keepdims=True) * Yo_w @ Yo_w.T)
        W = V.T @ sqrtm(np.diag(S)) @ U[:N].T
        Po = W * np.sign(W.T @ np.ones((L, 1))).T

    elif initcond == 5:
        Po = NFINDR(Ym, N)
    elif initcond == 6:
        Po = vca(Ym, N)
    elif initcond == 7:
        Po = SVMAX(Ym, N)
    else:
        raise ValueError(f"initcond={initcond} no es válido. Valores permitidos: 1, 2, 3, 4, 5, 6, 7.")

    Po = np.asarray(Po, dtype=np.float64)
    if Po.shape != (L, N):
        raise ValueError(f"initPo(): se esperaba una matriz {(L, N)}, se obtuvo {Po.shape}.")

    return Po


def normalize_endmembers(P, fallback=None, reference=None, eps=1e-12):
    P = np.asarray(P, dtype=np.float64).copy()

    if P.ndim != 2 or 0 in P.shape:
        raise ValueError("normalize_endmembers(): P debe ser una matriz no vacía.")

    L, N = P.shape
    P[~np.isfinite(P)] = 0.0
    P = np.maximum(P, 0.0)

    reference_spectrum = None
    if reference is not None:
        reference = np.asarray(reference, dtype=np.float64)
        if reference.ndim == 2 and reference.shape[0] == L and reference.shape[1] > 0:
            reference = np.maximum(np.where(np.isfinite(reference), reference, 0.0), 0.0)
            candidate = np.mean(reference, axis=1)
            total = np.sum(candidate)
            if np.isfinite(total) and total > eps:
                reference_spectrum = candidate / total

    if fallback is not None:
        fallback = np.asarray(fallback, dtype=np.float64)
        if fallback.shape != P.shape:
            fallback = None
        else:
            fallback = np.maximum(np.where(np.isfinite(fallback), fallback, 0.0), 0.0)

    for j in range(N):
        total = np.sum(P[:, j])

        if not np.isfinite(total) or total <= eps:
            recovered = False

            if fallback is not None:
                candidate = fallback[:, j]
                candidate_sum = np.sum(candidate)
                if np.all(np.isfinite(candidate)) and candidate_sum > eps:
                    P[:, j] = candidate / candidate_sum
                    recovered = True

            if not recovered and reference_spectrum is not None:
                P[:, j] = reference_spectrum
                recovered = True

            if not recovered:
                raise ValueError(
                    f"normalize_endmembers(): no se pudo recuperar el endmember {j}."
                )
        else:
            P[:, j] /= total

    if not np.all(np.isfinite(P)) or np.any(P < 0.0):
        raise ValueError("normalize_endmembers(): P contiene valores inválidos.")

    if not np.allclose(P.sum(axis=0), 1.0, atol=1e-10):
        raise ValueError("normalize_endmembers(): las columnas de P no suman uno.")

    return P


def abundance(Y, P, Lambda, parallel):
    if not np.all(np.isfinite(Y)) or not np.all(np.isfinite(P)):
        raise ValueError("abundance(): Y o P contienen NaN o Inf.")

    _, K = Y.shape
    N = P.shape[1]
    eps, tol_negative, tol_sum = 1e-10, 1e-10, 1e-6

    Go = P.T @ P
    eigvals = np.linalg.eigvalsh(Go)
    lmin = np.min(eigvals)
    G = Go - np.eye(N) * lmin * Lambda + eps * np.eye(N)
    c = np.ones(N)

    try:
        Gi = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        Gi = np.linalg.pinv(G)

    if not np.all(np.isfinite(Gi)):
        raise ValueError("abundance(): la inversa de G contiene NaN o Inf.")

    T1 = Gi @ c
    T2 = c @ T1
    if not np.isfinite(T2) or abs(T2) < eps:
        raise ValueError("abundance(): denominador sum-to-one demasiado pequeño.")

    def solve_constrained_pixel(b, initial_a):
        a = np.asarray(initial_a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        active = set()

        for _ in range(N):
            negative = [i for i in np.flatnonzero(a < -tol_negative) if i not in active]
            if not negative:
                break

            active.add(min(negative, key=lambda i: a[i]))
            if len(active) == N:
                a[:] = 1.0 / N
                break

            nactive = len(active)
            Gamma = np.zeros((N + nactive, N + nactive), dtype=float)
            Beta = np.zeros(N + nactive, dtype=float)
            Gamma[:N, :N] = G
            beta_eq = (c @ Gi @ b - 1.0) / T2
            Beta[:N] = b - c * beta_eq

            for q, idx_active in enumerate(sorted(active), start=N):
                Gamma[q, idx_active] = 1.0
                Gamma[idx_active, q] = 1.0

            try:
                solution = np.linalg.solve(Gamma, Beta)
            except np.linalg.LinAlgError:
                solution = np.linalg.pinv(Gamma) @ Beta

            a_new = solution[:N]
            if not np.all(np.isfinite(a_new)):
                break
            a = a_new

        a[~np.isfinite(a)] = 0.0
        a[a < 0.0] = 0.0
        a[np.abs(a) < tol_negative] = 0.0
        total = np.sum(a)

        if total <= eps:
            a[:] = 1.0 / N
        else:
            a /= total

        return a

    def process_block(start, end):
        B = P.T @ Y[:, start:end]
        D = (T1 @ B - 1.0) / T2
        Ab = Gi @ (B - c[:, None] * D[None, :])

        for j in np.flatnonzero(np.any(Ab < -tol_negative, axis=0)):
            Ab[:, j] = solve_constrained_pixel(B[:, j], Ab[:, j])

        Ab[~np.isfinite(Ab)] = 0.0
        Ab[Ab < 0.0] = 0.0
        Ab[np.abs(Ab) < tol_negative] = 0.0
        sums = Ab.sum(axis=0)
        valid = sums > eps
        Ab[:, valid] /= sums[valid][None, :]

        if np.any(~valid):
            Ab[:, ~valid] = 1.0 / N

        return Ab

    block_size = 4096
    blocks = [(start, min(start + block_size, K)) for start in range(0, K, block_size)]

    if parallel:
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(process_block)(start, end) for start, end in blocks
        )
    else:
        results = [process_block(start, end) for start, end in blocks]

    A = np.hstack(results)
    if not np.all(np.isfinite(A)) or np.min(A) < 0.0:
        raise ValueError("abundance(): A contiene valores inválidos.")

    sums = A.sum(axis=0)
    if not np.allclose(sums, 1.0, atol=tol_sum):
        raise ValueError(
            f"abundance(): las abundancias no suman uno por píxel. "
            f"Mínimo={sums.min()}, máximo={sums.max()}."
        )

    return A


def endmember(Y, A, rho, P_previous=None):
    N, K = A.shape
    eps = 1e-10
    Ksum = np.sum(Y ** 2, axis=0)
    W = 1.0 / (K * np.maximum(Ksum, eps))
    M = (W[None, :] * A) @ A.T
    M_regularized = M + rho * np.eye(N)

    try:
        Q = np.linalg.inv(M_regularized)
    except np.linalg.LinAlgError:
        Q = np.linalg.pinv(M_regularized)

    if not np.all(np.isfinite(Q)):
        raise ValueError("endmember(): Q contiene NaN o Inf.")

    P = ((W[None, :] * Y) @ A.T) @ Q
    return normalize_endmembers(P, fallback=P_previous, reference=Y)


def ebeae(Yo, N=2, parameters=None, Po=None, oea=0):
    initcond, rho, Lambda, epsilon, maxiter = 1, 0.1, 0, 1e-3, 20
    downsampling, parallel, display = 0.5, 1, 0

    if parameters is not None:
        initcond, rho, Lambda, epsilon, maxiter, downsampling, parallel, display = parameters
        initcond, maxiter = int(initcond), int(maxiter)

    Yo = np.asarray(Yo, dtype=np.float64)

    if Yo.ndim != 2 or not np.all(np.isfinite(Yo)):
        raise ValueError("ebeae(): Yo debe ser una matriz 2D con valores finitos.")
    if np.any(Yo < 0.0):
        raise ValueError("ebeae(): la imagen contiene valores negativos.")
    if N < 1:
        raise ValueError("ebeae(): N debe ser mayor o igual que uno.")
    if not 0.0 <= downsampling < 1.0:
        raise ValueError("ebeae(): downsampling debe estar en [0, 1).")

    L, K = Yo.shape
    if L == 0 or K == 0:
        raise ValueError("ebeae(): Yo no puede tener dimensiones vacías.")

    I = np.arange(K)
    Kdown = max(1, min(int(np.round(K * (1 - downsampling))), K))
    Is = np.sort(np.random.choice(K, Kdown, replace=False))
    Y = Yo[:, Is]

    mYm = np.sum(Y, axis=0)
    mYmo = np.sum(Yo, axis=0)
    eps = 1e-12
    Ym = Y / np.maximum(mYm, eps)
    Ymo = Yo / np.maximum(mYmo, eps)

    if Po is None:
        Po = initPo(Yo, Ym, initcond, N)
    else:
        Po = np.asarray(Po, dtype=np.float64).copy()

    if Po.shape != (L, N):
        raise ValueError(f"ebeae(): Po debe tener forma {(L, N)}, pero tiene forma {Po.shape}.")

    P = normalize_endmembers(Po, reference=Ym)
    iteration, J, Jp, Ji = 1, 1e5, 1e6, []
    Am = None

    while (Jp - J) / Jp >= epsilon and iteration <= maxiter and oea == 0:
        Am = abundance(Ym, P, Lambda, parallel)
        P_previous = P.copy()
        P = endmember(Ym, Am, rho, P_previous)
        Jp = J
        J = np.linalg.norm(Ym - P @ Am, "fro")
        Ji.append(abs(Jp - J) / Jp)

        if J > Jp:
            P = P_previous
            break

        if display:
            print(f"Iteration {iteration}: J = {J:.8f}")

        iteration += 1

    if Am is None:
        Am = abundance(Ym, P, Lambda, parallel)

    if downsampling:
        Ins = np.setdiff1d(I, Is)
        Ams = abundance(Ymo[:, Ins], P, Lambda, parallel)
        A = np.empty((N, K), dtype=np.float64)
        A[:, Is], A[:, Ins] = Am, Ams
    else:
        A = Am.copy()

    if oea:
        A = abundance(Ymo, P, Lambda, parallel)

    if not np.all(np.isfinite(A)) or np.any(A < 0.0):
        raise ValueError("ebeae(): A contiene valores inválidos.")

    abundance_sums = A.sum(axis=0)
    if not np.allclose(abundance_sums, 1.0, atol=1e-6):
        raise ValueError(
            f"ebeae(): las abundancias no suman uno. "
            f"Mínimo={abundance_sums.min()}, máximo={abundance_sums.max()}."
        )

    S = mYmo
    Yh = P @ (A * mYmo)

    if not np.all(np.isfinite(Yh)):
        raise ValueError("ebeae(): Yh contiene NaN o Inf.")

    return P, A, S, Yh, Ji
