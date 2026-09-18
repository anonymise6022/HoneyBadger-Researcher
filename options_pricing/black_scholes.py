"""
Black-Scholes-Merton closed-form option pricing and Greeks.

Model assumes:
    dS = r*S*dt + sigma*S*dW   (risk-neutral GBM, no dividends)

Let tau = T (time to expiry, in years). Define:

    d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

Call price:   C = S*N(d1) - K*e^(-rT)*N(d2)
Put price:    P = K*e^(-rT)*N(-d2) - S*N(-d1)
(Put-call parity: C - P = S - K*e^(-rT))

Greeks (per unit of S, T in years, sigma annualized):
    delta_call = N(d1)
    delta_put  = N(d1) - 1
    gamma      = phi(d1) / (S*sigma*sqrt(T))            (same for call & put)
    vega       = S*phi(d1)*sqrt(T)                       (same for call & put,
                                                           per 1.00 change in sigma)
    theta_call = -S*phi(d1)*sigma/(2*sqrt(T)) - r*K*e^(-rT)*N(d2)   (per year)
    theta_put  = -S*phi(d1)*sigma/(2*sqrt(T)) + r*K*e^(-rT)*N(-d2)  (per year)

where N(.) is the standard normal CDF and phi(.) is the standard normal PDF.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def delta(S, K, T, r, sigma, option_type="call"):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def gamma(S, K, T, r, sigma):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, option_type="call"):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        return term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    return term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)


if __name__ == "__main__":
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    print("call:", call_price(S, K, T, r, sigma))
    print("put:", put_price(S, K, T, r, sigma))
    print("delta (call):", delta(S, K, T, r, sigma, "call"))
    print("gamma:", gamma(S, K, T, r, sigma))
    print("vega:", vega(S, K, T, r, sigma))
    print("theta (call):", theta(S, K, T, r, sigma, "call"))
