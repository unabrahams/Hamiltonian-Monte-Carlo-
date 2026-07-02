import numpy as np
import time

### Implementación de RWMH

def run_rwmh(logpdf, x0, n_samples, proposal_std=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    x = np.array(x0, dtype=float).reshape(-1)
    dim = x.size

    samples = np.zeros((n_samples, dim))
    samples[0] = x

    accepted = 0
    rejections = 0

    start = time.time()

    for t in range(1, n_samples):
        proposal = x + rng.normal(scale=proposal_std, size=dim)

        log_alpha = logpdf(proposal) - logpdf(x)

        if np.log(rng.random()) < log_alpha:
            x = proposal
            accepted += 1
        else:
            rejections += 1

        samples[t] = x

    elapsed = time.time() - start

    info = {
        "algorithm": "RWMH",
        "accept_rate": accepted / (n_samples - 1),
        "rejections": rejections,
        "time": elapsed,
        "n_samples": n_samples,
        "proposal_std": proposal_std
    }

    return samples, info

### Implementación de HMC

def run_hmc(U, grad_U, epsilon, L, x0, n_samples, rng=None):
    """
    Hamiltonian Monte Carlo

    Parameters
    ----------
    U : callable
        potential energy, U(q) = -log pi(q).
    grad_U : callable
        Gradient of U.
    epsilon : float
        Step size from leapfrog scheme.
    L : int
        Number of leapfrog steps.
    x0 : array-like
        Initial state.
    n_samples : int
        Number of iterations.
    rng : np.random.Generator, opcional
        Random generator.

    Returns
    -------
    samples : ndarray, shape (n_samples, d)
        Generated chain.
    info : dict
        Run information.
    """
    if rng is None:
        rng = np.random.default_rng()

    x_current = np.array(x0, dtype=float).reshape(-1)
    dim = x_current.size

    samples = np.zeros((n_samples, dim))
    samples[0] = x_current

    accepted = 0
    rejections = 0

    start = time.time()

    for t in range(1, n_samples):
        p_current = rng.normal(loc=0.0, scale=1.0, size=dim)

        x_prop = x_current.copy()
        p_prop = p_current.copy()

        # medio paso en p
        p_prop = p_prop - 0.5 * epsilon * grad_U(x_prop)

        # L pasos leapfrog
        for j in range(L):
            x_prop = x_prop + epsilon * p_prop

            if j != L - 1:
                p_prop = p_prop - epsilon * grad_U(x_prop)

        # medio paso final en p
        p_prop = p_prop - 0.5 * epsilon * grad_U(x_prop)

        # reversibilidad
        p_prop = -p_prop

        current_H = U(x_current) + 0.5 * np.sum(p_current**2)
        proposed_H = U(x_prop) + 0.5 * np.sum(p_prop**2)

        log_alpha = current_H - proposed_H

        if np.log(rng.uniform()) < log_alpha:
            x_current = x_prop
            accepted += 1
        else:
            rejections += 1

        samples[t] = x_current

    elapsed = time.time() - start

    info = {
        "algorithm": "HMC",
        "accept_rate": accepted / (n_samples - 1),
        "rejections": rejections,
        "time": elapsed,
        "n_samples": n_samples,
        "step_size": epsilon,
        "n_leapfrog": L
    }

    return samples, info

### Funciones necesarias para NUTS


import numpy as np
import time


def leapfrog_nuts(theta, r, epsilon, grad_L):
    """
        r_tilde = r + (epsilon/2) grad L(theta)
        theta_tilde = theta + epsilon r_tilde
        r_tilde = r_tilde + (epsilon/2) grad L(theta_tilde)
    """
    r_half = r + 0.5 * epsilon * grad_L(theta)
    theta_new = theta + epsilon * r_half
    r_new = r_half + 0.5 * epsilon * grad_L(theta_new)
    return theta_new, r_new


def _joint(theta, r, L):
    val = L(theta)
    val = np.asarray(val, dtype=float).reshape(-1)[0]
    val = float(val)

    return val - 0.5 * np.sum(r**2)


def _stop_criterion(theta_minus, theta_plus, r_minus, r_plus):
    dtheta = theta_plus - theta_minus
    return int(np.dot(dtheta, r_minus) >= 0.0) * int(np.dot(dtheta, r_plus) >= 0.0)


def _build_tree(theta, r, log_u, v, j, epsilon, L, grad_L, Delta_max, rng):
    if j == 0:
        theta_prime, r_prime = leapfrog_nuts(theta, r, v * epsilon, grad_L)
        joint = _joint(theta_prime, r_prime, L)
        n_prime = int(log_u <= joint)
        s_prime = int(joint > log_u - Delta_max)
        return theta_prime, r_prime, theta_prime, r_prime, theta_prime, n_prime, s_prime
    else:
        theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime = \
            _build_tree(theta, r, log_u, v, j - 1, epsilon, L, grad_L, Delta_max, rng)

        if s_prime == 1:
            if v == -1:
                theta_minus2, r_minus2, _, _, theta_prime2, n_prime2, s_prime2 = \
                    _build_tree(theta_minus, r_minus, log_u, v, j - 1, epsilon, L, grad_L, Delta_max, rng)
                theta_minus = theta_minus2
                r_minus = r_minus2
            else:
                _, _, theta_plus2, r_plus2, theta_prime2, n_prime2, s_prime2 = \
                    _build_tree(theta_plus, r_plus, log_u, v, j - 1, epsilon, L, grad_L, Delta_max, rng)
                theta_plus = theta_plus2
                r_plus = r_plus2

            if (n_prime + n_prime2) > 0:
                if rng.uniform() < n_prime2 / (n_prime + n_prime2):
                    theta_prime = theta_prime2

            s_prime = s_prime2 * _stop_criterion(theta_minus, theta_plus, r_minus, r_plus)
            n_prime = n_prime + n_prime2

        return theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime



def run_nuts(L, grad_L, epsilon, x0, M, Delta_max, rng=None):
    """
    NUTS sampler.

    Parameters
    ----------
    L : callable
        Log-density.
    grad_L : callable
        Gradient of L.
    epsilon : float
        Step size.
    x0 : array-like
        Initial state theta^0.
    M : int
        Total number of iterations.
    Delta_max : float
        Delta_max for stop criterion.
    rng : np.random.Generator, opcional
        Random generator.

    Returns
    -------
    samples : ndarray
    info : dict
    """
    if rng is None:
        rng = np.random.default_rng()

    theta_current = np.array(x0, dtype=float).reshape(-1)
    d = theta_current.size
    samples = np.zeros((M, d))
    samples[0] = theta_current
    start = time.time()

    for m in range(1, M):
        r0 = rng.normal(size=d)
        joint0 = _joint(theta_current, r0, L)

        # antes: u = rng.uniform(0.0, np.exp(joint0))
        log_u = joint0 - rng.exponential()

        theta_minus = theta_current.copy()
        theta_plus = theta_current.copy()
        r_minus = r0.copy()
        r_plus = r0.copy()
        j = 0
        theta_m = theta_current.copy()
        n = 1
        s = 1

        while s == 1:
            v = rng.choice([-1, 1])
            if v == -1:
                theta_minus, r_minus, _, _, theta_prime, n_prime, s_prime = \
                    _build_tree(theta_minus, r_minus, log_u, v, j, epsilon, L, grad_L, Delta_max, rng)
            else:
                _, _, theta_plus, r_plus, theta_prime, n_prime, s_prime = \
                    _build_tree(theta_plus, r_plus, log_u, v, j, epsilon, L, grad_L, Delta_max, rng)

            if s_prime == 1:
                if rng.uniform() < min(1.0, n_prime / n):
                    theta_m = theta_prime.copy()

            n = n + n_prime
            s = s_prime * _stop_criterion(theta_minus, theta_plus, r_minus, r_plus)
            j = j + 1

        theta_current = theta_m
        samples[m] = theta_current

    elapsed = time.time() - start
    info = {
        "algorithm": "NUTS",
        "time": elapsed,
        "n_samples": M,
        "step_size": epsilon,
        "Delta_max": Delta_max
    }
    return samples, info

### raHMC


def conformal_leapfrog(grad_U, q, p, Sigma_inv, epsilon, gamma):
    """

        p_tilde <- exp(-gamma * epsilon / 2) * p - (epsilon/2) * grad_U(q)
        q_tilde <- q + epsilon * Sigma^{-1} p_tilde
        p_tilde <- exp(-gamma * epsilon / 2) * (p_tilde - (epsilon/2) * grad_U(q_tilde))

    Parameters
    ----------
    grad_U : callable
    q : ndarray
    p : ndarray
    Sigma_inv : ndarray
    epsilon : float
    gamma : float

    Returns
    -------
    q_new, p_new
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)

    a = np.exp(-gamma * epsilon / 2.0)

    p_tilde = a * p - 0.5 * epsilon * grad_U(q)
    q_tilde = q + epsilon * (Sigma_inv @ p_tilde)
    p_tilde = a * (p_tilde - 0.5 * epsilon * grad_U(q_tilde))

    return q_tilde, p_tilde


def run_rahmc(U, grad_U, Sigma, gamma, epsilon, L, x0, n_samples, rng=None):
    """
    Repelling Attracting Hamiltonian Monte Carlo

    Input:
        Potential U = -log pi
        mass matrix Sigma > 0
        friction gamma > 0
        step size epsilon > 0
        length L > 0
        sample size N

    Output:
        samples, info

    Parameters
    ----------
    U : callable
        Potential energy U(q) = -log pi(q).
    grad_U : callable
        Gradient of U.
    Sigma : ndarray
        Mass matrix.
    gamma : float
    epsilon : float
    L : int
    x0 : array-like
        Initial state q0.
    n_samples : int
    rng : np.random.Generator

    Returns
    -------
    samples : ndarray, shape (n_samples, d)
    info : dict
    """
    if rng is None:
        rng = np.random.default_rng()

    q_current = np.asarray(x0, dtype=float).reshape(-1)
    d = q_current.size

    Sigma = np.asarray(Sigma, dtype=float)
    Sigma_inv = np.linalg.inv(Sigma)

    samples = np.zeros((n_samples, d))
    samples[0] = q_current

    accepted = 0
    rejections = 0

    half_L = L // 2

    start = time.time()

    for n in range(1, n_samples):
        # Sample p_{n-1} ~ N(0, Sigma)
        p_current = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma)

        # Set (q, p) <- (q_{n-1}, p_{n-1})
        q = q_current.copy()
        p = p_current.copy()

        # for i = 1 to floor(L/2): with -gamma
        for _ in range(half_L):
            q, p = conformal_leapfrog(grad_U, q, p, Sigma_inv, epsilon, -gamma)

        # for i = 1 to floor(L/2): with +gamma
        for _ in range(half_L):
            q, p = conformal_leapfrog(grad_U, q, p, Sigma_inv, epsilon, +gamma)

        # Set (q, p) <- (q, -p)
        p = -p

        # Define H(q,p) = U(q) + 1/2 p^T Sigma^{-1} p
        H_current = U(q_current) + 0.5 * (p_current @ Sigma_inv @ p_current)
        H_proposed = U(q) + 0.5 * (p @ Sigma_inv @ p)

        log_alpha = H_current - H_proposed

        if np.log(rng.uniform()) < log_alpha:
            q_current = q
            accepted += 1
        else:
            rejections += 1

        samples[n] = q_current

    elapsed = time.time() - start

    info = {
        "algorithm": "rAHMC",
        "accept_rate": accepted / (n_samples - 1),
        "rejections": rejections,
        "time": elapsed,
        "n_samples": n_samples,
        "step_size": epsilon,
        "length": L,
        "gamma": gamma
    }

    return samples, info


### uHMC with sMC


def uhmc_smc_step(grad_U, x0, T, h, rng=None):
    """
    UHMC transition step with sMC time integration.

    Input:
        x0 : current state en R^d
        T  : duration
        h  : step size

    Output:
        x1 : next state of the chain
    """
    if rng is None:
        rng = np.random.default_rng()

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    d = x0.size

    # 1) Sample xi ~ N(0, I_d)
    V = rng.normal(size=d)

    # 2) Initialize
    Q = x0.copy()
    t = 0.0
    n = int(np.floor(T / h))

    # 3) for i = 0,...,n-1
    for _ in range(n):
        # 4) t_{i+1} = t_i + h
        # 5) u_i ~ Uniform(t_i, t_i + h)
        u = rng.uniform(t, t + h)

        # 6) F_{t_i} = -∇U(Q_{t_i} + (u_i - t_i) V_{t_i})
        F = -np.asarray(grad_U(Q + (u - t) * V), dtype=float).reshape(-1)

        # 7) Q_{t_{i+1}} = Q_{t_i} + h V_{t_i} + (1/2) h^2 F_{t_i}
        Q = Q + h * V + 0.5 * (h ** 2) * F

        # 8) V_{t_{i+1}} = V_{t_i} + h F_{t_i}
        V = V + h * F

        t = t + h

    # Output: X1 = Q_{t_n}
    return Q


def run_uhmc(grad_U, x0, T, h, n_samples, rng=None):
    """
    uHMC using transition step with sMC time integration

    Parameters
    ----------
    grad_U : callable
        Gradient of U.
    x0 : array-like
        Initial state.
    T : float
        Duration.
    h : float
        Step size.
    n_samples : int
        Number of iterations.
    rng : np.random.Generator

    Returns
    -------
    samples : ndarray, shape (n_samples, d)
    info : dict
    """
    if rng is None:
        rng = np.random.default_rng()

    x_current = np.asarray(x0, dtype=float).reshape(-1)
    d = x_current.size

    samples = np.zeros((n_samples, d))
    samples[0] = x_current

    start = time.time()

    for m in range(1, n_samples):
        x_current = uhmc_smc_step(grad_U, x_current, T, h, rng)
        samples[m] = x_current

    elapsed = time.time() - start

    info = {
        "algorithm": "uHMC",
        "time": elapsed,
        "n_samples": n_samples,
        "T": T,
        "h": h,
        "n_steps_per_transition": int(np.floor(T / h))
    }

    return samples, info

### Dual-Averaging versions


def run_hmc_dual_averaging(
    U,
    grad_U,
    epsilon0,
    lam,
    x0,
    n_samples,
    n_adapt,
    delta=0.65,
    rng=None
):
    """
    HMC with dual averaging.

    Parameters
    ----------
    U : callable
        Potential energy U(q) = -log pi(q).
    grad_U : callable
        Gradient of U.
    epsilon0 : float
        Initial step size.
    lam : float
        Approximate integration length used to define
        L_m = max(1, round(lam / epsilon_{m-1})).
    x0 : array-like
        Initial state.
    n_samples : int
        Total number of iterations.
    n_adapt : int
        Number of warm-up/adaptation iterations.
    delta : float, default=0.65
        Target acceptance rate.
    rng : np.random.Generator, optional

    Returns
    -------
    samples : ndarray, shape (n_samples, d)
    info : dict
    """
    if rng is None:
        rng = np.random.default_rng()

    x_current = np.asarray(x0, dtype=float).reshape(-1)
    d = x_current.size

    samples = np.zeros((n_samples, d))
    samples[0] = x_current

    accepted = 0
    rejections = 0

    # Dual averaging initialization (with the values recommended in the original paper)
    epsilon = float(epsilon0)
    mu = np.log(10.0 * epsilon0)
    epsilon_bar = 1.0
    H_bar = 0.0
    gamma_da = 0.05
    t0 = 10.0
    kappa = 0.75

    epsilon_history = np.zeros(n_samples)
    epsilon_bar_history = np.zeros(n_samples)
    accept_prob_history = np.zeros(n_samples)
    leapfrog_history = np.zeros(n_samples, dtype=int)

    epsilon_history[0] = epsilon
    epsilon_bar_history[0] = epsilon_bar
    accept_prob_history[0] = np.nan
    leapfrog_history[0] = max(1, int(np.round(lam / epsilon)))

    start = time.time()

    for m in range(1, n_samples):
        p0 = rng.normal(size=d)

        x_prop = x_current.copy()
        p_prop = p0.copy()

        L_m = max(1, int(np.round(lam / epsilon)))
        leapfrog_history[m] = L_m

        # Leapfrog integration
        p_prop = p_prop - 0.5 * epsilon * grad_U(x_prop)

        for j in range(L_m):
            x_prop = x_prop + epsilon * p_prop

            if j != L_m - 1:
                p_prop = p_prop - epsilon * grad_U(x_prop)

        p_prop = p_prop - 0.5 * epsilon * grad_U(x_prop)
        p_prop = -p_prop

        current_H = U(x_current) + 0.5 * np.sum(p0**2)
        proposed_H = U(x_prop) + 0.5 * np.sum(p_prop**2)

        log_alpha = current_H - proposed_H

        if not np.isfinite(log_alpha):
            alpha = 0.0
        else:
            alpha = np.exp(min(0.0, log_alpha))

        accept_prob_history[m] = alpha

        if np.log(rng.uniform()) < log_alpha:
            x_current = x_prop
            accepted += 1
        else:
            rejections += 1

        samples[m] = x_current

        # Dual averaging update
        if m <= n_adapt:
            eta_H = 1.0 / (m + t0)

            # Safety: make sure alpha is finite and in [0, 1]
            if not np.isfinite(alpha):
                alpha = 0.0

            alpha = np.clip(alpha, 0.0, 1.0)

            H_bar = (1.0 - eta_H) * H_bar + eta_H * (delta - alpha)

            log_epsilon = mu - (np.sqrt(m) / gamma_da) * H_bar

            # Safety bounds
            log_eps_min = np.log(1e-5)
            log_eps_max = np.log(2.0)

            log_epsilon = np.clip(log_epsilon, log_eps_min, log_eps_max)

            log_epsilon_bar = (
                (m ** (-kappa)) * log_epsilon
                + (1.0 - m ** (-kappa)) * np.log(epsilon_bar)
            )

            log_epsilon_bar = np.clip(log_epsilon_bar, log_eps_min, log_eps_max)

            epsilon = np.exp(log_epsilon)
            epsilon_bar = np.exp(log_epsilon_bar)

        else:
            epsilon = epsilon_bar

        epsilon_history[m] = epsilon
        epsilon_bar_history[m] = epsilon_bar

    elapsed = time.time() - start

    info = {
        "algorithm": "HMC-DA",
        "accept_rate": accepted / (n_samples - 1),
        "rejections": rejections,
        "time": elapsed,
        "n_samples": n_samples,
        "n_adapt": n_adapt,
        "delta": delta,
        "lambda": lam,
        "epsilon0": epsilon0,
        "final_epsilon": epsilon,
        "final_epsilon_bar": epsilon_bar,
        "epsilon_history": epsilon_history,
        "epsilon_bar_history": epsilon_bar_history,
        "accept_prob_history": accept_prob_history,
        "leapfrog_history": leapfrog_history
    }

    return samples, info


# NUTS with Dual Averaging


def _build_tree_da(
    theta,
    r,
    log_u,
    v,
    j,
    epsilon,
    L,
    grad_L,
    Delta_max,
    theta0,
    r0,
    rng
):
    """
    BuildTree for NUTS with dual averaging.

    Returns
    -------
    theta_minus, r_minus, theta_plus, r_plus,
    theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime
    """

    if j == 0:
        # Base case: one leapfrog step in direction v
        theta_prime, r_prime = leapfrog_nuts(theta, r, v * epsilon, grad_L)

        joint_prime = _joint(theta_prime, r_prime, L)
        joint0 = _joint(theta0, r0, L)

        # Validity checks
        if not np.isfinite(joint_prime):
            n_prime = 0
            s_prime = 0
            alpha_prime = 0.0
            n_alpha_prime = 1

            return (
                theta_prime, r_prime,
                theta_prime, r_prime,
                theta_prime, n_prime, s_prime,
                alpha_prime, n_alpha_prime
            )

        n_prime = int(log_u <= joint_prime)
        s_prime = int(joint_prime > log_u - Delta_max)

        log_alpha_prime = joint_prime - joint0

        if not np.isfinite(log_alpha_prime):
            alpha_prime = 0.0
        else:
            alpha_prime = np.exp(min(0.0, log_alpha_prime))

        alpha_prime = np.clip(alpha_prime, 0.0, 1.0)
        n_alpha_prime = 1

        return (
            theta_prime, r_prime,
            theta_prime, r_prime,
            theta_prime, n_prime, s_prime,
            alpha_prime, n_alpha_prime
        )

    else:
        # First subtree
        (
            theta_minus, r_minus,
            theta_plus, r_plus,
            theta_prime, n_prime, s_prime,
            alpha_prime, n_alpha_prime
        ) = _build_tree_da(
            theta,
            r,
            log_u,
            v,
            j - 1,
            epsilon,
            L,
            grad_L,
            Delta_max,
            theta0,
            r0,
            rng
        )

        if s_prime == 1:
            if v == -1:
                (
                    theta_minus2, r_minus2,
                    _, _,
                    theta_prime2, n_prime2, s_prime2,
                    alpha_prime2, n_alpha_prime2
                ) = _build_tree_da(
                    theta_minus,
                    r_minus,
                    log_u,
                    v,
                    j - 1,
                    epsilon,
                    L,
                    grad_L,
                    Delta_max,
                    theta0,
                    r0,
                    rng
                )

                theta_minus = theta_minus2
                r_minus = r_minus2

            else:
                (
                    _, _,
                    theta_plus2, r_plus2,
                    theta_prime2, n_prime2, s_prime2,
                    alpha_prime2, n_alpha_prime2
                ) = _build_tree_da(
                    theta_plus,
                    r_plus,
                    log_u,
                    v,
                    j - 1,
                    epsilon,
                    L,
                    grad_L,
                    Delta_max,
                    theta0,
                    r0,
                    rng
                )

                theta_plus = theta_plus2
                r_plus = r_plus2

            # Choose candidate from second subtree with probability proportional to n_prime2
            if (n_prime + n_prime2) > 0:
                if rng.uniform() < n_prime2 / (n_prime + n_prime2):
                    theta_prime = theta_prime2

            alpha_prime = alpha_prime + alpha_prime2
            n_alpha_prime = n_alpha_prime + n_alpha_prime2

            s_prime = (
                s_prime2
                * _stop_criterion(theta_minus, theta_plus, r_minus, r_plus)
            )

            n_prime = n_prime + n_prime2

        return (
            theta_minus, r_minus,
            theta_plus, r_plus,
            theta_prime, n_prime, s_prime,
            alpha_prime, n_alpha_prime
        )


def run_nuts_dual_averaging(
    L,
    grad_L,
    epsilon0,
    x0,
    n_samples,
    n_adapt,
    delta=0.80,
    Delta_max=1000.0,
    rng=None,
    eps_min=1e-5,
    eps_max=2.0,
    max_tree_depth=10
):
    """
    NUTS with dual averaging.

    Parameters
    ----------
    L : callable
        Log-density of the target distribution.
    grad_L : callable
        Gradient of the log-density.
    epsilon0 : float
        Initial step size.
    x0 : array-like
        Initial state.
    n_samples : int
        Total number of iterations.
    n_adapt : int
        Number of warm-up/adaptation iterations.
    delta : float, default=0.80
        Target acceptance statistic for dual averaging.
    Delta_max : float, default=1000.0
        Divergence threshold used in NUTS.
    rng : np.random.Generator, optional
        Random number generator.
    eps_min : float, default=1e-5
        Lower bound for epsilon.
    eps_max : float, default=1.0
        Upper bound for epsilon.
    max_tree_depth : int, default=10
        Maximum tree depth allowed in NUTS.

    Returns
    -------
    samples : ndarray
    info : dict
    """

    if rng is None:
        rng = np.random.default_rng()

    theta_current = np.asarray(x0, dtype=float).reshape(-1)
    d = theta_current.size

    samples = np.zeros((n_samples, d))
    samples[0] = theta_current

    # Dual averaging initialization
    epsilon = float(epsilon0)
    mu = np.log(10.0 * epsilon0)

    epsilon_bar = float(epsilon0)

    H_bar = 0.0
    gamma_da = 0.05
    t0 = 10.0
    kappa = 0.75

    log_eps_min = np.log(eps_min)
    log_eps_max = np.log(eps_max)

    # Clip initial epsilon
    log_epsilon = np.clip(np.log(epsilon), log_eps_min, log_eps_max)
    epsilon = np.exp(log_epsilon)

    log_epsilon_bar = np.clip(np.log(epsilon_bar), log_eps_min, log_eps_max)
    epsilon_bar = np.exp(log_epsilon_bar)

    epsilon_history = np.zeros(n_samples)
    epsilon_bar_history = np.zeros(n_samples)
    h_nuts_history = np.full(n_samples, np.nan)
    alpha_sum_history = np.full(n_samples, np.nan)
    n_alpha_history = np.zeros(n_samples, dtype=int)
    tree_depth_history = np.zeros(n_samples, dtype=int)

    epsilon_history[0] = epsilon
    epsilon_bar_history[0] = epsilon_bar

    start = time.time()

    for m in range(1, n_samples):
        # Sample momentum r0 ~ N(0, I)
        r0 = rng.normal(size=d)

        # Joint log-density
        joint0 = _joint(theta_current, r0, L)

        if not np.isfinite(joint0):
            raise ValueError(
                "Initial joint density is not finite. "
                "Check the initial point x0 and the log-density L."
            )

        # log u = joint0 + log Uniform(0, 1)
        # Equivalent to joint0 - Exponential(1)
        log_u = joint0 - rng.exponential()

        # Initialize tree
        theta_minus = theta_current.copy()
        theta_plus = theta_current.copy()
        r_minus = r0.copy()
        r_plus = r0.copy()

        j = 0
        theta_m = theta_current.copy()
        n = 1
        s = 1

        alpha = 0.0
        n_alpha = 0

        # Build tree until U-turn, divergence, or max tree depth
        while s == 1 and j < max_tree_depth:
            v = rng.choice([-1, 1])

            if v == -1:
                (
                    theta_minus, r_minus,
                    _, _,
                    theta_prime, n_prime, s_prime,
                    alpha_prime, n_alpha_prime
                ) = _build_tree_da(
                    theta_minus,
                    r_minus,
                    log_u,
                    v,
                    j,
                    epsilon,
                    L,
                    grad_L,
                    Delta_max,
                    theta_current,
                    r0,
                    rng
                )

            else:
                (
                    _, _,
                    theta_plus, r_plus,
                    theta_prime, n_prime, s_prime,
                    alpha_prime, n_alpha_prime
                ) = _build_tree_da(
                    theta_plus,
                    r_plus,
                    log_u,
                    v,
                    j,
                    epsilon,
                    L,
                    grad_L,
                    Delta_max,
                    theta_current,
                    r0,
                    rng
                )

            # Candidate selection
            if s_prime == 1:
                if n > 0:
                    prob = min(1.0, n_prime / n)
                else:
                    prob = 0.0

                if rng.uniform() < prob:
                    theta_m = theta_prime.copy()

            n = n + n_prime

            s = (
                s_prime
                * _stop_criterion(theta_minus, theta_plus, r_minus, r_plus)
            )

            j = j + 1

            alpha = alpha_prime
            n_alpha = n_alpha_prime

        theta_current = theta_m
        samples[m] = theta_current

        # NUTS acceptance statistic for dual averaging
        if n_alpha > 0 and np.isfinite(alpha):
            h_nuts = alpha / n_alpha
        else:
            h_nuts = 0.0

        h_nuts = np.clip(h_nuts, 0.0, 1.0)

        h_nuts_history[m] = h_nuts
        alpha_sum_history[m] = alpha
        n_alpha_history[m] = n_alpha
        tree_depth_history[m] = j

        # Dual averaging update
        if m <= n_adapt:
            eta_H = 1.0 / (m + t0)

            if not np.isfinite(h_nuts):
                h_nuts = 0.0

            h_nuts = np.clip(h_nuts, 0.0, 1.0)

            H_bar = (1.0 - eta_H) * H_bar + eta_H * (delta - h_nuts)

            log_epsilon = mu - (np.sqrt(m) / gamma_da) * H_bar

            # Safety bounds for epsilon
            log_epsilon = np.clip(log_epsilon, log_eps_min, log_eps_max)

            log_epsilon_bar = (
                (m ** (-kappa)) * log_epsilon
                + (1.0 - m ** (-kappa)) * np.log(epsilon_bar)
            )

            log_epsilon_bar = np.clip(
                log_epsilon_bar,
                log_eps_min,
                log_eps_max
            )

            epsilon = np.exp(log_epsilon)
            epsilon_bar = np.exp(log_epsilon_bar)

        else:
            epsilon = epsilon_bar

        epsilon_history[m] = epsilon
        epsilon_bar_history[m] = epsilon_bar

    elapsed = time.time() - start

    info = {
        "algorithm": "NUTS-DA",
        "time": elapsed,
        "n_samples": n_samples,
        "n_adapt": n_adapt,
        "delta": delta,
        "Delta_max": Delta_max,
        "epsilon0": epsilon0,
        "eps_min": eps_min,
        "eps_max": eps_max,
        "max_tree_depth": max_tree_depth,
        "final_epsilon": epsilon,
        "final_epsilon_bar": epsilon_bar,
        "epsilon_history": epsilon_history,
        "epsilon_bar_history": epsilon_bar_history,
        "h_nuts_history": h_nuts_history,
        "alpha_sum_history": alpha_sum_history,
        "n_alpha_history": n_alpha_history,
        "tree_depth_history": tree_depth_history
    }

    return samples, info

### raHMC con SA


def run_rahmc_dual_averaging(
    U,
    grad_U,
    Sigma,
    epsilon0,
    gamma0,
    T,
    x0,
    n_samples,
    n_adapt,
    delta=0.60,
    rng=None,
    eps_min=1e-5,
    eps_max=1.0,
    gamma_min=1e-4,
    gamma_max=2.0
):
    if rng is None:
        rng = np.random.default_rng()

    q_current = np.asarray(x0, dtype=float).reshape(-1)
    d = q_current.size

    Sigma = np.asarray(Sigma, dtype=float)
    Sigma_inv = np.linalg.inv(Sigma)

    samples = np.zeros((n_samples, d))
    samples[0] = q_current

    accepted = 0
    rejections = 0

    # Dual averaging in x = (log epsilon, log gamma)
    x = np.array([np.log(epsilon0), np.log(gamma0)], dtype=float)
    mu = np.array([np.log(10.0 * epsilon0), np.log(10.0 * gamma0)], dtype=float)

    # Important: initialize the averaged log-parameters at the initial values
    x_bar = x.copy()

    H_bar = np.zeros(2, dtype=float)

    omega = 0.05
    t0 = 10.0
    kappa = 0.75

    log_lower = np.array([np.log(eps_min), np.log(gamma_min)])
    log_upper = np.array([np.log(eps_max), np.log(gamma_max)])

    # Clip initial values too
    x = np.clip(x, log_lower, log_upper)
    x_bar = np.clip(x_bar, log_lower, log_upper)

    epsilon_history = np.zeros(n_samples)
    gamma_history = np.zeros(n_samples)
    epsilon_bar_history = np.zeros(n_samples)
    gamma_bar_history = np.zeros(n_samples)
    accept_prob_history = np.full(n_samples, np.nan)
    leapfrog_history = np.zeros(n_samples, dtype=int)

    epsilon_history[0] = np.exp(x[0])
    gamma_history[0] = np.exp(x[1])
    epsilon_bar_history[0] = np.exp(x_bar[0])
    gamma_bar_history[0] = np.exp(x_bar[1])

    L0 = max(2, int(np.round(T / np.exp(x[0]))))
    if L0 % 2 == 1:
        L0 += 1
    leapfrog_history[0] = L0

    start = time.time()

    for m in range(1, n_samples):
        epsilon = np.exp(x[0])
        gamma = np.exp(x[1])

        L_m = max(2, int(np.round(T / epsilon)))

        # Make L_m even so that repelling and attracting phases are balanced
        if L_m % 2 == 1:
            L_m += 1

        leapfrog_history[m] = L_m
        half_L = L_m // 2

        p_current = rng.multivariate_normal(mean=np.zeros(d), cov=Sigma)

        q = q_current.copy()
        p = p_current.copy()

        # Repelling phase
        for _ in range(half_L):
            q, p = conformal_leapfrog(grad_U, q, p, Sigma_inv, epsilon, -gamma)

        # Attracting phase
        for _ in range(half_L):
            q, p = conformal_leapfrog(grad_U, q, p, Sigma_inv, epsilon, +gamma)

        # Momentum flip
        p = -p

        H_current = U(q_current) + 0.5 * (p_current @ Sigma_inv @ p_current)
        H_proposed = U(q) + 0.5 * (p @ Sigma_inv @ p)

        log_alpha = H_current - H_proposed

        # Safety for invalid proposals
        if not np.isfinite(log_alpha):
            log_alpha = -np.inf
            alpha = 0.0
        else:
            alpha = np.exp(min(0.0, log_alpha))

        alpha = np.clip(alpha, 0.0, 1.0)
        accept_prob_history[m] = alpha

        if np.log(rng.uniform()) < log_alpha:
            q_current = q
            accepted += 1
        else:
            rejections += 1

        samples[m] = q_current

        # Dual averaging update
        if m <= n_adapt:
            f_m = delta - alpha
            g_m = np.array([f_m, f_m], dtype=float)

            eta_H = 1.0 / (m + t0)
            H_bar = (1.0 - eta_H) * H_bar + eta_H * g_m

            x = mu - (np.sqrt(m) / omega) * H_bar

            # Safety: clip log epsilon and log gamma
            x = np.clip(x, log_lower, log_upper)

            x_bar = (
                (m ** (-kappa)) * x
                + (1.0 - m ** (-kappa)) * x_bar
            )

            # Safety: clip averaged log-parameters too
            x_bar = np.clip(x_bar, log_lower, log_upper)

        else:
            x = x_bar.copy()

        epsilon_history[m] = np.exp(x[0])
        gamma_history[m] = np.exp(x[1])
        epsilon_bar_history[m] = np.exp(x_bar[0])
        gamma_bar_history[m] = np.exp(x_bar[1])

    elapsed = time.time() - start

    info = {
        "algorithm": "rAHMC-DA",
        "accept_rate": accepted / (n_samples - 1),
        "rejections": rejections,
        "time": elapsed,
        "n_samples": n_samples,
        "n_adapt": n_adapt,
        "delta": delta,
        "T": T,
        "epsilon0": epsilon0,
        "gamma0": gamma0,
        "final_epsilon": np.exp(x[0]),
        "final_gamma": np.exp(x[1]),
        "final_epsilon_bar": np.exp(x_bar[0]),
        "final_gamma_bar": np.exp(x_bar[1]),
        "epsilon_history": epsilon_history,
        "gamma_history": gamma_history,
        "epsilon_bar_history": epsilon_bar_history,
        "gamma_bar_history": gamma_bar_history,
        "accept_prob_history": accept_prob_history,
        "leapfrog_history": leapfrog_history
    }

    return samples, info

