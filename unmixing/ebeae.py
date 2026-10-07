import numpy as np
from scipy.linalg import pinv, svd, sqrtm
from joblib import Parallel, delayed
from unmixing.endmember_extraction import vca, SVMAX, NFINDR

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

            # Avoid selecting spectra identical to previously selected endmembers
            duplicate = np.zeros(
                Yt.shape[1],
                dtype=bool
            )

            for j in range(index + 1):
                duplicate |= np.all(Yt == Po[:, j:j + 1],axis=0)
            
            ymax[duplicate] = np.inf
            IImax = np.argmin(ymax)
            Po[:, index + 1] = Yt[:, IImax]
            Yt = np.delete(Yt,IImax,axis=1)
            index += 1

    elif initcond == 2:

        energy = np.sum(np.abs(Yo),axis=0)
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

            ymax = np.min(np.abs(e),axis=1)
            IImax = np.argmin(ymax)
            Po[:, index + 1] = Yt[:, IImax]
            Yt = np.delete(Yt,IImax,axis=1)
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

        Yom = np.mean(Ym,axis=1,keepdims=True)
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

    # ========================================================
    # Parámetros numéricos
    # ========================================================

    eps = 1e-10
    tol_negative = 1e-10
    tol_sum = 1e-6

    # ========================================================
    # Matriz del problema
    # ========================================================

    Go = P.T @ P

    if not np.all(np.isfinite(Go)):
        raise ValueError(
            "abundance(): Go contiene NaN o Inf"
        )

    # ========================================================
    # Regularización original
    # ========================================================

    eigvals = np.linalg.eigvalsh(Go)

    lmin = np.min(eigvals)

    G = (
        Go
        - np.eye(N) * lmin * Lambda
    )

    # ========================================================
    # Regularización numérica
    # ========================================================

    G = (
        G
        + eps * np.eye(N)
    )

    # ========================================================
    # Vector de la restricción
    #
    #       c^T a = 1
    # ========================================================

    c = np.ones(N)

    # ========================================================
    # Inversa de G
    # ========================================================

    try:

        Gi = np.linalg.inv(G)

    except np.linalg.LinAlgError:

        Gi = np.linalg.pinv(G)

    # ========================================================
    # Términos comunes
    # ========================================================

    T1 = Gi @ c

    T2 = c @ T1

    if abs(T2) < eps:

        raise ValueError(
            "abundance(): denominador de la restricción "
            "sum-to-one demasiado pequeño."
        )

    # ========================================================
    # SOLUCIÓN CON ACTIVE-SET
    # ========================================================

    def solve_constrained_pixel(
        b,
        initial_a
    ):

        # ----------------------------------------------------
        # Forzar vectores 1-D
        # ----------------------------------------------------

        b = np.asarray(
            b,
            dtype=float
        ).reshape(-1)

        a = np.asarray(
            initial_a,
            dtype=float
        ).reshape(-1)

        # ----------------------------------------------------
        # Variables activas:
        #
        # a_i = 0
        # ----------------------------------------------------

        active = set()

        # ----------------------------------------------------
        # Límite de iteraciones
        # ----------------------------------------------------

        max_iterations = N

        for _ in range(
            max_iterations
        ):

            # =================================================
            # Buscar negativos
            # =================================================

            negative = np.flatnonzero(
                a < -tol_negative
            )

            negative = [
                i
                for i in negative
                if i not in active
            ]

            # ------------------------------------------------
            # Ya es factible
            # ------------------------------------------------

            if len(negative) == 0:
                break

            # ------------------------------------------------
            # Variable más negativa
            # ------------------------------------------------

            idx = min(
                negative,
                key=lambda i: a[i]
            )

            # ------------------------------------------------
            # Fijar a cero
            # ------------------------------------------------

            active.add(idx)

            # ------------------------------------------------
            # Variables libres
            # ------------------------------------------------

            free = [
                i
                for i in range(N)
                if i not in active
            ]

            # ------------------------------------------------
            # Caso degenerado
            # ------------------------------------------------

            if len(free) == 0:

                a[:] = 1.0 / N

                break

            # =================================================
            # Sistema KKT
            # =================================================

            nactive = len(active)

            Gamma = np.zeros(
                (
                    N + nactive,
                    N + nactive
                )
            )

            Beta = np.zeros(
                N + nactive
            )

            # ------------------------------------------------
            # Bloque cuadrático
            # ------------------------------------------------

            Gamma[
                :N,
                :N
            ] = G

            # ------------------------------------------------
            # Término de la restricción sum-to-one
            # ------------------------------------------------

            beta_eq = (
                (
                    c @ Gi @ b
                    - 1.0
                )
                / T2
            )

            rhs = (
                b
                - c * beta_eq
            )

            Beta[
                :N
            ] = rhs

            # ------------------------------------------------
            # Restricciones activas
            #
            #       a_i = 0
            # ------------------------------------------------

            for q, idx_active in enumerate(
                sorted(active),
                start=N
            ):

                Gamma[
                    q,
                    idx_active
                ] = 1.0

                Gamma[
                    idx_active,
                    q
                ] = 1.0

                Beta[
                    q
                ] = 0.0

            # =================================================
            # Resolver KKT
            # =================================================

            try:

                solution = np.linalg.solve(
                    Gamma,
                    Beta
                )

            except np.linalg.LinAlgError:

                try:

                    solution = (
                        np.linalg.pinv(Gamma)
                        @ Beta
                    )

                except np.linalg.LinAlgError:

                    solution = None

            # ------------------------------------------------
            # Fallo del solver
            # ------------------------------------------------

            if solution is None:
                break

            solution = np.asarray(
                solution,
                dtype=float
            ).reshape(-1)

            a_new = solution[
                :N
            ]

            # ------------------------------------------------
            # Verificar finitud
            # ------------------------------------------------

            if not np.all(
                np.isfinite(a_new)
            ):

                break

            a = a_new

        # ====================================================
        # Limpieza numérica
        # ====================================================

        # Negativos diminutos -> exactamente cero
        a[
            (a < 0)
            & (a >= -tol_negative)
        ] = 0.0

        # ----------------------------------------------------
        # Seguridad
        # ----------------------------------------------------

        a[
            a < 0
        ] = 0.0

        # ====================================================
        # Normalización sum-to-one
        # ====================================================

        total = np.sum(a)

        if total > eps:

            a /= total

        else:

            # Caso degenerado
            a[:] = 1.0 / N

        # ====================================================
        # Limpiar valores muy pequeños
        # ====================================================

        a[
            np.abs(a) < tol_negative
        ] = 0.0

        # ====================================================
        # Segunda normalización
        # ====================================================

        total = np.sum(a)

        if total > eps:

            a /= total

        else:

            a[:] = 1.0 / N

        return a

    # ========================================================
    # PROCESAR BLOQUE
    # ========================================================

    def process_block(
        start,
        end
    ):

        Yb = Y[
            :,
            start:end
        ]

        # ----------------------------------------------------
        # P'Y
        #
        # B -> (N, Kb)
        # ----------------------------------------------------

        B = P.T @ Yb

        # ----------------------------------------------------
        # Solución inicial con sum-to-one
        #
        # T1 -> (N,)
        # B  -> (N,Kb)
        #
        # T1 @ B -> (Kb,)
        # ----------------------------------------------------

        D = (
            T1 @ B
            - 1.0
        ) / T2

        # ----------------------------------------------------
        # Solución inicial
        #
        # D -> (Kb,)
        #
        # c[:,None] -> (N,1)
        #
        # D[None,:] -> (1,Kb)
        #
        # Resultado -> (N,Kb)
        # ----------------------------------------------------

        Ab = Gi @ (
            B
            - c[:, None] * D[None, :]
        )

        # ====================================================
        # Detectar píxeles negativos
        # ====================================================

        negative = np.any(
            Ab < -tol_negative,
            axis=0
        )

        negative_indices = np.flatnonzero(
            negative
        )

        # ====================================================
        # Active-set solamente para píxeles problemáticos
        # ====================================================

        for j in negative_indices:

            Ab[
                :,
                j
            ] = solve_constrained_pixel(
                B[
                    :,
                    j
                ],
                Ab[
                    :,
                    j
                ]
            )

        # ====================================================
        # Limpieza global
        # ====================================================

        Ab[
            Ab < 0
        ] = 0.0

        # ====================================================
        # Sum-to-one
        # ====================================================

        sums = np.sum(
            Ab,
            axis=0
        )

        valid = (
            sums > eps
        )

        if np.any(valid):

            Ab[
                :,
                valid
            ] /= sums[
                valid
            ][None, :]

        # ====================================================
        # Casos degenerados
        # ====================================================

        invalid = ~valid

        if np.any(invalid):

            Ab[
                :,
                invalid
            ] = 1.0 / N

        # ====================================================
        # Limpieza de residuos numéricos
        # ====================================================

        Ab[
            np.abs(Ab) < tol_negative
        ] = 0.0

        # ====================================================
        # Segunda normalización
        # ====================================================

        sums = np.sum(
            Ab,
            axis=0
        )

        valid = (
            sums > eps
        )

        if np.any(valid):

            Ab[
                :,
                valid
            ] /= sums[
                valid
            ][None, :]

        # ----------------------------------------------------
        # Casos degenerados
        # ----------------------------------------------------

        invalid = ~valid

        if np.any(invalid):

            Ab[
                :,
                invalid
            ] = 1.0 / N

        return Ab

    # ========================================================
    # BLOQUES
    # ========================================================

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

    # ========================================================
    # PROCESAMIENTO
    # ========================================================

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

    # ========================================================
    # UNIR BLOQUES
    # ========================================================

    A = np.hstack(
        results
    )

    # ========================================================
    # VERIFICACIÓN FINITA
    # ========================================================

    if not np.all(
        np.isfinite(A)
    ):

        raise ValueError(
            "abundance(): A contiene NaN o Inf"
        )

    # ========================================================
    # VERIFICACIÓN DE NO NEGATIVIDAD
    # ========================================================

    min_A = np.min(A)

    if min_A < -1e-8:

        raise ValueError(
            "abundance(): "
            "A contiene valores negativos. "
            f"Min = {min_A}"
        )

    # ========================================================
    # VERIFICACIÓN SUM-TO-ONE
    # ========================================================

    sums = np.sum(
        A,
        axis=0
    )

    if not np.allclose(
        sums,
        1.0,
        atol=tol_sum
    ):

        raise ValueError(
            "abundance(): "
            "La restricción sum-to-one "
            "no se cumple.\n"
            f"Min sum = {sums.min()}\n"
            f"Max sum = {sums.max()}\n"
            f"Mean sum = {sums.mean()}"
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