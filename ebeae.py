import numpy as np
from scipy.linalg import pinv, svd, sqrtm
from joblib import Parallel, delayed
from endmember_extraction import vca, SVMAX, NFINDR


# ============================================================
# INITIALIZATION
# ============================================================

def initPo(Yo, Ym, initcond, N):

    L = Yo.shape[0]
    Po = np.zeros((L, N))

    if initcond == 1:

        index = 0
        Yt = Ym.copy()

        # First endmember: mean spectrum
        Po[:, 0] = np.mean(Yo, axis=1)

        # Select remaining endmembers
        while index < N - 1:

            Pn = Po[:, :index + 1]

            num = Yt.T @ Pn

            den = (
                np.linalg.norm(Yt, axis=0)[:, None]
                * np.linalg.norm(Pn, axis=0)[None, :]
            )

            eps = 1e-12
            e = num / (den + eps)


            ymax = np.min(np.abs(e), axis=1)

            # Avoid selecting spectra identical to
            # previously selected endmembers
            duplicate = np.zeros(
                Yt.shape[1],
                dtype=bool
            )

            for j in range(index + 1):

                duplicate |= np.all(
                    Yt == Po[:, j:j + 1],
                    axis=0
                )

            ymax[duplicate] = np.inf

            IImax = np.argmin(ymax)

            Po[:, index + 1] = Yt[:, IImax]

            Yt = np.delete(
                Yt,
                IImax,
                axis=1
            )

            index += 1

    elif initcond == 2:

        energy = np.sum(
            np.abs(Yo),
            axis=0
        )

        Imax = np.argmax(energy)
        Imin = np.argmin(energy)

        Po[:, 0] = Yo[:, Imax]
        Po[:, 1] = Yo[:, Imin]

        Yt = np.delete(
            Yo,
            [Imax, Imin],
            axis=1
        )

        index = 1

        while index < N - 1:

            Pn = Po[:, :index + 1]

            num = Yt.T @ Pn

            den = (
                np.linalg.norm(Yt, axis=0)[:, None]
                * np.linalg.norm(Pn, axis=0)[None, :]
            )

            e = num / den

            ymax = np.min(
                np.abs(e),
                axis=1
            )

            IImax = np.argmin(ymax)

            Po[:, index + 1] = Yt[:, IImax]

            Yt = np.delete(
                Yt,
                IImax,
                axis=1
            )

            index += 1

    elif initcond == 3:

        _, _, V = svd(
            Ym.T,
            full_matrices=False
        )

        W = V[:N].T

        Po = W * np.sign(
            W.T @ np.ones((L, 1))
        ).T

    elif initcond == 4:

        Yom = np.mean(
            Ym,
            axis=1,
            keepdims=True
        )

        Yon = Ym - Yom

        _, S, V = svd(
            Yon.T,
            full_matrices=False
        )

        Yo_w = (
            pinv(sqrtm(np.diag(S)))
            @ V
            @ Ym.T
        )

        U, _, _ = svd(
            (Yo_w ** 2).sum(
                axis=1,
                keepdims=True
            )
            * Yo_w
            @ Yo_w.T
        )

        W = (
            V.T
            @ sqrtm(np.diag(S))
            @ U[:N].T
        )

        Po = W * np.sign(
            W.T @ np.ones((L, 1))
        ).T

    elif initcond == 5:

        Po = NFINDR(
            Ym,
            N
        )

    elif initcond == 6:

        Po = vca(
            Ym,
            N
        )

    elif initcond == 7:

        Po = SVMAX(
            Ym,
            N
        )

    return Po


# ============================================================
# ABUNDANCE
# ============================================================

def abundance(Y, P, Lambda, parallel):

    if not np.all(np.isfinite(Y)):
        raise ValueError(
            "abundance(): Y contiene NaN o Inf"
        )

    if not np.all(np.isfinite(P)):
        raise ValueError(
            "abundance(): P contiene NaN o Inf"
        )

    L, K = Y.shape
    N = P.shape[1]

    c = np.ones((N, 1))

    # --------------------------------------------------------
    # Common matrices
    # --------------------------------------------------------
    Go = P.T @ P
    if not np.all(np.isfinite(Go)):
        raise ValueError(
            "abundance(): Go contiene NaN o Inf"
        )

    # --------------------------------------------------------
    # Numerical stability
    # --------------------------------------------------------

    eigvals = np.linalg.eigvalsh(Go)
    lmin = np.min(eigvals)

    G = (
        Go
        - np.eye(N) * lmin * Lambda
    )

    # Small numerical regularization.
    # Does NOT modify Y or remove zero-valued pixels.
    eps = 1e-10

    G = G + eps * np.eye(N)

    Gi = np.linalg.inv(G)

    T1 = Gi @ c
    T2 = c.T @ T1

    # --------------------------------------------------------
    # Process one block
    # --------------------------------------------------------

    def process_block(start, end):

        Yb = Y[:, start:end]

        # P'Y
        B = P.T @ Yb

        # ||y_k||^2
        by = np.sum(
            Yb ** 2,
            axis=0
        )

        # Unconstrained solution
        D = (
            T1.T @ B - 1
        ) / T2

        Ab = Gi @ (
            B - c @ D
        )

        # ----------------------------------------------------
        # Pixels with negative abundances
        # ----------------------------------------------------

        negative = np.any(
            Ab < 0,
            axis=0
        )

        for j in np.flatnonzero(negative):

            bk = B[:, [j]]
            byk = by[j]
            ak = Ab[:, [j]]

            # ------------------------------------------------
            # Active-set constrained solution
            # ------------------------------------------------

            while np.any(ak < 0):

                Iset = np.flatnonzero(
                    ak[:, 0] < 0
                )

                if len(Iset) == 0:
                    break

                # Remove the most negative abundance
                i = Iset[
                    np.argmin(
                        ak[Iset, 0]
                    )
                ]

                # Fix selected abundance to zero
                active = np.flatnonzero(
                    ak[:, 0] >= 0
                )

                if len(active) == 0:
                    break

                Gamma = np.zeros(
                    (
                        N + len(active),
                        N + len(active)
                    )
                )

                Beta = np.zeros(
                    (
                        N + len(active),
                        1
                    )
                )

                # --------------------------------------------
                # Original quadratic system
                # --------------------------------------------

                Gamma[:N, :N] = G

                Beta[:N, :] = (
                    bk
                    - c * (
                        (
                            c.T
                            @ Gi
                            @ bk
                            - 1
                        )
                        / T2
                    )
                )

                # --------------------------------------------
                # Equality constraints
                # --------------------------------------------

                for q, idx in enumerate(
                    active,
                    start=N
                ):

                    Gamma[
                        q,
                        idx
                    ] = 1

                    Gamma[
                        idx,
                        q
                    ] = 1

                Beta[
                    N:,
                    0
                ] = 0

                # --------------------------------------------
                # Solve
                # --------------------------------------------

                try:

                    solution = np.linalg.solve(
                        Gamma,
                        Beta
                    )

                    ak = solution[
                        :N,
                        :
                    ]

                except np.linalg.LinAlgError:

                    # Fall back to the unconstrained solution
                    ak = Ab[
                        :,
                        j:j + 1
                    ]

                    break

                # Avoid infinite active-set loops
                if np.any(
                    np.isnan(ak)
                ):

                    ak = Ab[
                        :,
                        j:j + 1
                    ]

                    break

                # If the same negative element remains,
                # set it explicitly to zero
                if np.any(
                    ak < 0
                ):

                    ak[
                        ak < 0
                    ] = 0

            Ab[
                :,
                j
            ] = ak[:, 0]

        return Ab

    # --------------------------------------------------------
    # Block size
    # --------------------------------------------------------

    block_size = 4096

    blocks = [
        (
            start,
            min(
                start + block_size,
                K
            )
        )
        for start in range(
            0,
            K,
            block_size
        )
    ]

    # --------------------------------------------------------
    # Parallel block processing
    # --------------------------------------------------------

    if parallel:

        results = Parallel(
            n_jobs=-1,
            backend="loky"
        )(
            delayed(process_block)(
                start,
                end
            )
            for start, end in blocks
        )

    else:

        results = [
            process_block(
                start,
                end
            )
            for start, end in blocks
        ]

    A = np.hstack(
        results
    )

    return A


# ============================================================
# ENDMEMBER UPDATE
# ============================================================

def endmember(Y, A, rho):

    N, K = A.shape
    L = Y.shape[0]

    # --------------------------------------------------------
    # Pixel weights
    # --------------------------------------------------------

    Ksum = np.sum(
        Y ** 2,
        axis=0
    )

    eps = 1e-10

    W = 1.0 / (
        K * np.maximum(Ksum, eps)
    )


    # --------------------------------------------------------
    # Weighted matrices
    # --------------------------------------------------------

    WA = W[None, :] * A

    M = WA @ A.T

    Q = np.linalg.inv(
        M + rho * np.eye(N)
    )

    P = (
        (W[None, :] * Y)
        @ A.T
        @ Q
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    P = P / np.maximum(
        np.sum(
            P,
            axis=0,
            keepdims=True
        ),
        1
    )

    return P


# ============================================================
# EBEAE
# ============================================================

def ebeae(
    Yo,
    N=2,
    parameters=None,
    Po=None,
    oea=0
):

    # --------------------------------------------------------
    # Default parameters
    # --------------------------------------------------------

    initcond = 1
    rho = 0.1
    Lambda = 0
    epsilon = 1e-3
    maxiter = 20
    downsampling = 0.5
    parallel = 1
    display = 0

    if parameters is not None:

        initcond = int(
            parameters[0]
        )

        rho = parameters[1]
        Lambda = parameters[2]
        epsilon = parameters[3]
        maxiter = parameters[4]
        downsampling = parameters[5]
        parallel = parameters[6]
        display = parameters[7]

    L, K = Yo.shape

    # --------------------------------------------------------
    # Downsampling
    # --------------------------------------------------------

    I = np.arange(K)

    Kdown = int(
        np.round(
            K * (1 - downsampling)
        )
    )

    Is = np.sort(
        np.random.choice(
            K,
            Kdown,
            replace=False
        )
    )

    Y = Yo[:, Is]

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    mYm = np.sum(
        Y,
        axis=0
    )

    mYmo = np.sum(
        Yo,
        axis=0
    )

    eps = 1e-12

    Ym = Y / np.maximum(mYm, eps)
    Ymo = Yo / np.maximum(mYmo, eps)
    #Ym = Y / mYm
    #Ymo = Yo / mYmo
    
    # --------------------------------------------------------
    # Initial endmembers
    # --------------------------------------------------------

    Po = (
        initPo(
            Yo,
            Ym,
            initcond,
            N
        )
        if Po is None
        else Po
    )
    
    eps = 1e-12

    Po /= (
        np.sum(
            Po,
            axis=0,
            keepdims=True
        ) + eps
    )


    P = Po

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    iteration = 1
    J = 1e5
    Jp = 1e6
    Ji = []

    while (
        (Jp - J) / Jp >= epsilon
        and iteration <= maxiter
        and oea == 0
    ):

        Am = abundance(
            Ym,
            P,
            Lambda,
            parallel
        )

        P_previous = P.copy()

        P = endmember(
            Ym,
            Am,
            rho
        )

        Jp = J

        J = np.linalg.norm(
            Ym - P @ Am,
            "fro"
        )

        Ji.append(
            abs(Jp - J) / Jp
        )

        if J > Jp:

            P = P_previous
            break

        if display:

            print(
                f"Iteration {iteration}: J = {J}"
            )

        iteration += 1

    # --------------------------------------------------------
    # Remaining pixels
    # --------------------------------------------------------

    if downsampling:

        Ins = np.setdiff1d(
            I,
            Is
        )

        Ams = abundance(
            Ymo[:, Ins],
            P,
            Lambda,
            parallel
        )

        A = np.empty(
            (
                N,
                K
            ),
            dtype=Am.dtype
        )

        A[:, Is] = Am
        A[:, Ins] = Ams

    else:

        A = Am

    # --------------------------------------------------------
    # OEA
    # --------------------------------------------------------

    if oea:

        A = abundance(
            Ymo,
            P,
            Lambda,
            parallel
        )

    # --------------------------------------------------------
    # Reconstruction
    # --------------------------------------------------------

    S = mYmo

    Yh = P @ (
        A * mYmo
    )

    return P, A, S, Yh, Ji