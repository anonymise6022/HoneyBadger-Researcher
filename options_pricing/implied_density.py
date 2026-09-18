"""
Breeden-Litzenberger (1978) risk-neutral density extraction.

C(K) = e^(-rT) * E^Q[max(S_T-K, 0)]. Differentiating twice w.r.t. K and using
d/dK max(S_T-K,0) = -1{S_T>K} gives d^2C/dK^2 = e^(-rT)*q(K), i.e.:

    q(K) = e^(rT) * d^2C/dK^2

Approximated via the central finite difference on a uniform strike grid
(spacing h): d^2C/dK^2 |_{K_i} ~= [C(K_{i+1}) - 2*C(K_i) + C(K_{i-1})] / h^2
"""

import numpy as np


def extract_density(K, C, T, r):
    """
    K, C: strikes (evenly spaced, ascending) and call prices at each strike.
    T, r: time to expiry (years), risk-free rate.
    Returns dict with interior strikes, density q(K), and the mean/variance/
    skew of the implied distribution (via numerical integration of the
    normalized density).
    """
    K = np.asarray(K, dtype=float)
    C = np.asarray(C, dtype=float)

    h = np.diff(K)
    if not np.allclose(h, h[0]):
        raise ValueError("K must be an evenly spaced, ascending strike grid")
    h = h[0]

    # central second difference at interior points only
    d2C_dK2 = (C[2:] - 2 * C[1:-1] + C[:-2]) / (h ** 2)
    density = np.exp(r * T) * d2C_dK2
    density = np.clip(density, a_min=0.0, a_max=None)  # enforce q(K) >= 0

    strikes = K[1:-1]

    # normalize so density integrates to 1 (raw BL output can be off due to
    # finite-difference / discretization error)
    total_mass = np.trapezoid(density, strikes)
    if total_mass <= 0:
        raise ValueError("computed density has non-positive total mass")
    pdf = density / total_mass

    mean = np.trapezoid(strikes * pdf, strikes)
    variance = np.trapezoid(((strikes - mean) ** 2) * pdf, strikes)
    std = np.sqrt(variance)
    skew = np.trapezoid(((strikes - mean) ** 3) * pdf, strikes) / (std ** 3)

    return {
        "strikes": strikes,
        "density": density,
        "mean": mean,
        "variance": variance,
        "skew": skew,
    }


if __name__ == "__main__":
    from black_scholes import call_price

    S, r, sigma, T = 100.0, 0.03, 0.2, 0.5
    K = np.linspace(60, 140, 401)
    C = call_price(S, K, T, r, sigma)

    result = extract_density(K, C, T, r)
    print("mean:", result["mean"], "(theoretical: ~", S * np.exp(r * T), ")")
    print("variance:", result["variance"])
    print("skew:", result["skew"], "(theoretical: ~0 for lognormal in log-space,"
          " small positive in price-space)")
