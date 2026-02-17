import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ─── fractional differencing / integration ────────────────────────────────────

def fracdiff_weights(d, n):
    """Pi weights for (1-L)^d X_t = Y_t  (differencing, d > 0 removes memory)"""
    coef = np.zeros(n)
    coef[0] = 1.0
    for i in range(1, n):
        coef[i] = coef[i-1] * (i - 1 - d) / i
    return coef

def fracint_weights(d, n):
    """Pi weights for (1-L)^{-d} eps_t = X_t  (integration, restores memory)"""
    coef = np.zeros(n)
    coef[0] = 1.0
    for i in range(1, n):
        coef[i] = coef[i-1] * (i - 1 + d) / i
    return coef

def fracdiff_multi(X, d_vec):
    n, k = X.shape
    Y = np.zeros((n, k))
    for j in range(k):
        d = d_vec[j]
        if abs(d) < 1e-8:
            Y[:, j] = X[:, j]
        else:
            coef = fracdiff_weights(d, n)
            for t in range(n):
                Y[t, j] = np.dot(coef[:t+1], X[t::-1, j])
    return Y

def fracint_multi(eps, d_vec):
    n, k = eps.shape
    X = np.zeros((n, k))
    for j in range(k):
        d = d_vec[j]
        if abs(d) < 1e-8:
            X[:, j] = eps[:, j]
        else:
            coef = fracint_weights(d, n)
            for t in range(n):
                X[t, j] = np.dot(coef[:t+1], eps[t::-1, j])
    return X

# ─── simulate VARFIMA(1, D, 1) ────────────────────────────────────────────────

def sim_varfima_1d1(n, d, Phi, Theta, Sigma, seed=None):
    rng = np.random.default_rng(seed)
    k = len(d)
    n_total = n + 100

    eps = rng.multivariate_normal(np.zeros(k), Sigma, size=n_total)

    # MA(1): u_t = eps_t + Theta @ eps_{t-1}
    u = np.zeros((n_total, k))
    u[0] = eps[0]
    for t in range(1, n_total):
        u[t] = eps[t] + Theta @ eps[t-1]

    # AR(1) inversion: Y_t = u_t + Phi @ Y_{t-1}
    Y = np.zeros((n_total, k))
    Y[0] = u[0]
    for t in range(1, n_total):
        Y[t] = u[t] + Phi @ Y[t-1]

    # fractional integration
    X = fracint_multi(Y, d)
    return X[-n:]

# ─── log-likelihood for VARFIMA(1, D, 1) ─────────────────────────────────────

def unpack_params(params, k):
    idx = 0
    d     = params[idx:idx+k];           idx += k
    Phi   = params[idx:idx+k*k].reshape(k, k); idx += k*k
    Theta = params[idx:idx+k*k].reshape(k, k); idx += k*k
    # Sigma = L @ L.T  (Cholesky)
    L = np.zeros((k, k))
    L[np.tril_indices(k)] = params[idx:]
    Sigma = L @ L.T
    return d, Phi, Theta, Sigma

def varfima_loglik(params, X):
    try:
        n, k = X.shape
        d, Phi, Theta, Sigma = unpack_params(params, k)

        if np.any(np.abs(d) >= 0.49):
            return 1e10
        if np.max(np.abs(np.linalg.eigvals(Phi))) >= 0.99:
            return 1e10
        eigvals_S = np.linalg.eigvalsh(Sigma)
        if eigvals_S.min() <= 1e-6:
            return 1e10

        Y = fracdiff_multi(X, d)

        eps = np.zeros((n, k))
        eps[0] = Y[0]
        for t in range(1, n):
            eps[t] = Y[t] - Phi @ Y[t-1] - Theta @ eps[t-1]

        if not np.all(np.isfinite(eps)):
            return 1e10

        Sigma_inv = np.linalg.inv(Sigma)
        log_det   = np.log(np.linalg.det(Sigma))
        ll = 0.0
        for t in range(1, n):
            e = eps[t]
            ll -= 0.5 * (log_det + e @ Sigma_inv @ e + k * np.log(2 * np.pi))

        return -ll
    except Exception:
        return 1e10

# ─── estimation ───────────────────────────────────────────────────────────────

def varfima_1d1_estimate(X):
    n, k = X.shape
    n_chol = k * (k + 1) // 2
    n_params = k + 2 * k*k + n_chol

    p0 = np.zeros(n_params)
    p0[:k] = 0.2                                           # d
    p0[k:k+k*k] = (np.eye(k) * 0.2).ravel()               # Phi
    p0[k+k*k:k+2*k*k] = (np.eye(k) * 0.1).ravel()        # Theta
    L0 = np.eye(k)
    p0[k+2*k*k:] = L0[np.tril_indices(k)]                 # Cholesky of Sigma

    lb = np.concatenate([np.full(k, -0.49),
                         np.full(2*k*k, -0.99),
                         np.full(n_chol, -5.0)])
    ub = np.concatenate([np.full(k,  0.49),
                         np.full(2*k*k,  0.99),
                         np.full(n_chol,  5.0)])

    res = minimize(varfima_loglik, p0, args=(X,), method='L-BFGS-B',
                   bounds=list(zip(lb, ub)),
                   options={'maxiter': 1000, 'ftol': 1e-7})

    d, Phi, Theta, Sigma = unpack_params(res.x, k)
    return dict(d=d, Phi=Phi, Theta=Theta, Sigma=Sigma,
                loglik=-res.fun, convergence=res.success)

# ─── Monte Carlo ──────────────────────────────────────────────────────────────

def monte_carlo_varfima_1d1(n_sim=50, n=1000, true_params=None, base_seed=1000):
    k = len(true_params['d'])
    d_est     = np.full((n_sim, k), np.nan)
    Phi_est   = np.full((n_sim, k, k), np.nan)
    Theta_est = np.full((n_sim, k, k), np.nan)

    for sim in range(n_sim):
        X = sim_varfima_1d1(n,
                            true_params['d'],
                            true_params['Phi'],
                            true_params['Theta'],
                            true_params['Sigma'],
                            seed=base_seed + sim)
        try:
            fit = varfima_1d1_estimate(X)
            if fit['convergence']:
                d_est[sim]     = fit['d']
                Phi_est[sim]   = fit['Phi']
                Theta_est[sim] = fit['Theta']
        except Exception:
            pass
        print(f"\rSimulation {sim+1}/{n_sim} completed", end='', flush=True)

    print()
    return dict(d_est=d_est, Phi_est=Phi_est, Theta_est=Theta_est)

# ─── plotting ─────────────────────────────────────────────────────────────────

def plot_histograms(results, true_params, pdf_path="varfima_estimation_histograms.pdf"):
    k = true_params['d'].shape[0]

    # flatten all parameter estimates and true values into lists
    param_labels = []
    param_samples = []
    param_trues = []

    for j in range(k):
        param_labels.append(f"$d_{j+1}$")
        param_samples.append(results['d_est'][:, j])
        param_trues.append(true_params['d'][j])

    for i in range(k):
        for j in range(k):
            param_labels.append(f"$\\Phi_{{{i+1},{j+1}}}$")
            param_samples.append(results['Phi_est'][:, i, j])
            param_trues.append(true_params['Phi'][i, j])

    for i in range(k):
        for j in range(k):
            param_labels.append(f"$\\Theta_{{{i+1},{j+1}}}$")
            param_samples.append(results['Theta_est'][:, i, j])
            param_trues.append(true_params['Theta'][i, j])

    # drop nan
    clean_samples = [s[~np.isnan(s)] for s in param_samples]
    n_params = len(param_labels)

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(max(10, n_params * 0.9), 5))

        bp = ax.boxplot(clean_samples,
                        positions=range(n_params),
                        widths=0.5,
                        patch_artist=True,
                        whis=[2.5, 97.5],      # whiskers at 2.5/97.5 percentile
                        showfliers=False,
                        medianprops=dict(color='black', linewidth=1.5),
                        boxprops=dict(facecolor='white', color='black'),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'),
                        meanprops=dict(marker='o', markerfacecolor='black',
                                       markeredgecolor='black', markersize=4),
                        showmeans=True)

        # true value markers
        for x, tv in enumerate(param_trues):
            ax.scatter(x, tv, marker='x', color='black', s=60,
                       linewidths=1.5, zorder=5, label='True value' if x == 0 else None)

        ax.set_xticks(range(n_params))
        ax.set_xticklabels(param_labels, fontsize=9)
        ax.set_ylabel("Value", fontsize=10)
        ax.set_title("VARFIMA(1,D,1) — Parameter estimates (box: 25/75%, whisker: 2.5/97.5%,  ●: mean,  ×: true)",
                     fontsize=9)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.4)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF saved as: {pdf_path}")

# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    true_params = dict(
        d     = np.array([0.25, 0.35]),
        Phi   = np.array([[0.2, 0.1], [0.05, 0.3]]),
        Theta = np.array([[0.15, 0.05], [0.1, 0.2]]),
        Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    )

    results = monte_carlo_varfima_1d1(n_sim=50, n=5000, true_params=true_params)

    print("Mean estimated d:  ", np.nanmean(results['d_est'], axis=0))
    print("Mean estimated Phi:\n",   np.nanmean(results['Phi_est'], axis=0))
    print("Mean estimated Theta:\n", np.nanmean(results['Theta_est'], axis=0))

    plot_histograms(results, true_params, pdf_path="varfima_estimation_histograms.pdf")
