"""Unbiased pass@k estimator for code-generation style evaluation."""

from math import comb


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if n<0 or c<0 or c>n or k<1 or k>n: raise ValueError("require 0<=c<=n and 1<=k<=n")
    if n-c<k:return 1.0
    return 1-comb(n-c,k)/comb(n,k)


def macro_pass_at_k(counts, k):
    values=tuple(estimate_pass_at_k(n,c,k) for n,c in counts)
    return sum(values)/len(values) if values else None
