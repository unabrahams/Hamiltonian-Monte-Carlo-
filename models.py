import numpy as np


# Gamma(5,1)


def logpdf_gamma_5_1(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    z = x[0]

    if z <= 0:
        return -np.inf

    return float(4.0 * np.log(z) - z - np.log(24.0))


def U_gamma_5_1(q):
    """
    Potential U(q) = -log pi(q).
    """
    q = np.asarray(q, dtype=float).reshape(-1)

    if q.size != 1:
        raise ValueError("Gamma(5,1) aquí está implementada como modelo 1-dimensional.")

    x = q[0]

    if x <= 0:
        return np.inf

    return -4.0 * np.log(x) + x + np.log(24.0)


def grad_U_gamma_5_1(q):
    """
    Gradient of U(q) for Gamma(5,1).
    """
    q = np.asarray(q, dtype=float).reshape(-1)

    if q.size != 1:
        raise ValueError("Gamma(5,1) aquí está implementada como modelo 1-dimensional.")

    x = q[0]

    if x <= 0:
        return np.array([np.nan])

    return np.array([1.0 - 4.0 / x])



# Multivariate normal dimension d=100
# stds = 0.01, 0.02, ..., 1.00   si d=100


def make_neal_mvn_model(d=100):
    """

    Varianzas:
        sigma_i = i / d,  i=1,...,d
    de modo que si d=100:
        sigma_i = 0.01, 0.02, ..., 1.00

    Returns a dictionary with:
        - stds
        - vars
        - precision_diag
        - cov
        - cov_inv
        - logpdf
        - U
        - grad_U
    """
    stds = np.arange(1, d + 1, dtype=float) / d
    vars_ = stds**2
    precision_diag = 1.0 / vars_

    cov = np.diag(vars_)
    cov_inv = np.diag(precision_diag)

    def logpdf(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != d:
            raise ValueError(f"Se esperaba vector de dimensión {d}.")
        return -0.5 * np.sum(precision_diag * (x**2))

    def U(q):
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != d:
            raise ValueError(f"Se esperaba vector de dimensión {d}.")
        return 0.5 * np.sum(precision_diag * (q**2))

    def grad_U(q):
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != d:
            raise ValueError(f"Se esperaba vector de dimensión {d}.")
        return precision_diag * q

    return {
        "name": f"neal_mvn_d{d}",
        "dim": d,
        "stds": stds,
        "vars": vars_,
        "precision_diag": precision_diag,
        "cov": cov,
        "cov_inv": cov_inv,
        "logpdf": logpdf,
        "U": U,
        "grad_U": grad_U,
    }



# Helpers

def gamma_initial_point(x0=100.0):
    return np.array([float(x0)])


def neal_mvn_initial_point(d=100, value=7.0):
    return value * np.ones(d, dtype=float)



# Bivariate mixture


def make_bivariate_gaussian_mixture():
    """
    Mezcla:
        0.4 * N(mean1, Sigma1) + 0.6 * N(mean2, Sigma2)

    con
        mean1 = [0, 0]
        mean2 = [5, 5]
        Sigma1 = [[1, 0.5], [0.5, 1]]
        Sigma2 = [[1, -0.3], [-0.3, 1]]
    """
    mean1 = np.array([0.0, 0.0])
    mean2 = np.array([5.0, 5.0])

    Sigma1 = np.array([[1.0, 0.5],
                       [0.5, 1.0]])
    Sigma2 = np.array([[1.0, -0.3],
                       [-0.3, 1.0]])

    Sigma1_inv = np.linalg.inv(Sigma1)
    Sigma2_inv = np.linalg.inv(Sigma2)

    det1 = np.linalg.det(Sigma1)
    det2 = np.linalg.det(Sigma2)

    norm_const1 = 1.0 / (2.0 * np.pi * np.sqrt(det1))
    norm_const2 = 1.0 / (2.0 * np.pi * np.sqrt(det2))

    w1 = 0.4
    w2 = 0.6

    def _gaussian_pdf(q, mean, Sigma_inv, norm_const):
        q = np.asarray(q, dtype=float).reshape(-1)
        diff = q - mean
        exponent = -0.5 * diff @ Sigma_inv @ diff
        return norm_const * np.exp(exponent)

    def _grad_gaussian_pdf(q, mean, Sigma_inv, norm_const):
        q = np.asarray(q, dtype=float).reshape(-1)
        pdf_val = _gaussian_pdf(q, mean, Sigma_inv, norm_const)
        return -pdf_val * (Sigma_inv @ (q - mean))

    def pdf(q):
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != 2:
            raise ValueError("La mezcla gaussiana está implementada en dimensión 2.")
        return (
            w1 * _gaussian_pdf(q, mean1, Sigma1_inv, norm_const1)
            + w2 * _gaussian_pdf(q, mean2, Sigma2_inv, norm_const2)
        )

    def logpdf(q):
        val = pdf(q)
        if val <= 0:
            return -np.inf
        return np.log(val)

    def grad_logpdf(q):
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != 2:
            raise ValueError("La mezcla gaussiana está implementada en dimensión 2.")

        pdf1 = _gaussian_pdf(q, mean1, Sigma1_inv, norm_const1)
        pdf2 = _gaussian_pdf(q, mean2, Sigma2_inv, norm_const2)

        grad1 = -pdf1 * (Sigma1_inv @ (q - mean1))
        grad2 = -pdf2 * (Sigma2_inv @ (q - mean2))

        denom = w1 * pdf1 + w2 * pdf2
        numer = w1 * grad1 + w2 * grad2

        return numer / denom

    def U(q):
        return -logpdf(q)

    def grad_U(q):
        return -grad_logpdf(q)

    return {
        "name": "bivariate_gaussian_mixture",
        "dim": 2,
        "weights": np.array([w1, w2]),
        "mean1": mean1,
        "mean2": mean2,
        "Sigma1": Sigma1,
        "Sigma2": Sigma2,
        "Sigma1_inv": Sigma1_inv,
        "Sigma2_inv": Sigma2_inv,
        "pdf": pdf,
        "logpdf": logpdf,
        "grad_logpdf": grad_logpdf,
        "U": U,
        "grad_U": grad_U,
    }


def gaussian_mixture_initial_point(x0=(0.0, 0.0)):
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.size != 2:
        raise ValueError("El punto inicial debe ser de dimensión 2.")
    return x0




# 8 schools


def make_eight_schools_model(
    y=None,
    kappa=None
):
    """
    Modelo jerárquico:

        q = [mu, tau, eta_1, ..., eta_8]

    with potential

        U(q) =
            0.5 * (mu^2 + tau^2 + sum eta_i^2)
            + 0.5 * sum_i ((y_i - (mu + tau * eta_i))^2 / kappa_i^2)


    """
    if y is None:
        y = np.array([2.8, 0.8, -0.3, 0.7, -0.1, 0.1, 1.8, 1.2], dtype=float)
    else:
        y = np.asarray(y, dtype=float).reshape(-1)

    if kappa is None:
        kappa = np.array([0.8, 0.5, 0.8, 0.6, 0.5, 0.6, 0.5, 0.4], dtype=float)
    else:
        kappa = np.asarray(kappa, dtype=float).reshape(-1)

    if y.size != 8 or kappa.size != 8:
        raise ValueError("Este modelo espera y y kappa de longitud 8.")

    def U(q):
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != 10:
            raise ValueError("Se esperaba q de dimensión 10: [mu, tau, eta_1,...,eta_8].")

        mu = q[0]
        tau = q[1]
        eta = q[2:]

        prior = 0.5 * (mu**2 + tau**2 + np.sum(eta**2))
        resid = y - (mu + tau * eta)
        like = 0.5 * np.sum((resid**2) / (kappa**2))

        return prior + like

    def grad_U(q):
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != 10:
            raise ValueError("Se esperaba q de dimensión 10: [mu, tau, eta_1,...,eta_8].")

        mu = q[0]
        tau = q[1]
        eta = q[2:]

        inv_k2 = 1.0 / (kappa**2)

        dmu = mu + np.sum((mu + tau * eta - y) * inv_k2)
        dtau = tau + np.sum(eta * (mu + tau * eta - y) * inv_k2)
        deta = eta + tau * (mu + tau * eta - y) * inv_k2

        return np.concatenate(([dmu, dtau], deta))

    def logpdf(q):
        return -U(q)

    return {
        "name": "eight_schools_noncentered_tau_free",
        "dim": 10,
        "y": y,
        "kappa": kappa,
        "logpdf": logpdf,
        "U": U,
        "grad_U": grad_U,
    }


def eight_schools_initial_point(mu0=2.0, tau0=2.0, eta0=2.0):
    """
    Punto inicial para q = [mu, tau, eta_1,...,eta_8].
    """
    q0 = np.zeros(10, dtype=float)
    q0[0] = mu0
    q0[1] = tau0
    q0[2:] = eta0
    return q0


# Cauchy standar


def logpdf_cauchy(x):
    """
    Log densidad de Cauchy(0,1)
    """
    x = np.asarray(x, dtype=float)
    return -np.log(np.pi) - np.log1p(x**2)


def U_cauchy(q):
    """
    Potencial U(q) = -log pi(q)
    """
    q = np.asarray(q, dtype=float).reshape(-1)

    if q.size != 1:
        raise ValueError("Cauchy aquí está implementado en dimensión 1.")

    x = q[0]

    return np.log(np.pi) + np.log1p(x**2)


def grad_U_cauchy(q):
    """
    Gradiente de U(q)
    """
    q = np.asarray(q, dtype=float).reshape(-1)

    if q.size != 1:
        raise ValueError("Cauchy aquí está implementado en dimensión 1.")

    x = q[0]

    return np.array([2.0 * x / (1.0 + x**2)])


def cauchy_initial_point(x0=0.0):
    return np.array([float(x0)])