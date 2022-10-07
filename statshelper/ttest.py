import numpy as np
import scipy.stats.distributions as distributions
from collections import namedtuple
import scipy.stats as stats
from scipy.stats.morestats import _get_wilcoxon_distr
from scipy.stats import find_repeats
import warnings


Ttest_indResult = namedtuple(
    "Ttest_indResult",
    (
        "statistic",
        "pvalue",
        "dof",
        "mean1",
        "mean2",
        "var1",
        "var2",
        "tail",
        "cohen_d",
    ),
)

Ttest_relResult = namedtuple(
    "Ttest_relResult",
    (
        "statistic",
        "pvalue",
        "dof",
        "mean",
        "var",
        "tail",
        "cohen_d",
    ),
)

WilcoxonResult = namedtuple(
    "WilcoxonResult",
    (
        "statistic",
        "pvalue",
        "z",
        "tail",
        "cohen_d",
    ),
)


def ttest_rel(x, y, equal_var=True, two_tailed=True, m0=0):

    if len(x) != len(y):
        raise ValueError("unequal length arrays")

    n = float(len(x))
    diff = x - y

    # Compute mean
    mean = np.mean(diff)

    # compute unbiased standard deviation
    std = np.std(diff, ddof=1)

    # degrees of freedom
    dof = n - 1

    # Compute t-statistic
    t = np.sqrt(n) * (mean - m0) / std

    # Compute pvalue
    p = distributions.t.sf(abs(t), dof)

    if two_tailed:
        p *= 2.0
        tail = "two-tailed"

    else:
        tail = "one-tailed"

    if equal_var:
        std_pooled = np.sqrt(0.5 * (np.var(x, ddof=1) + np.var(y, ddof=1)))
    else:
        std_pooled = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / n)
    cohen_d = abs((np.mean(x) - np.mean(y)) / std_pooled)

    return Ttest_relResult(t, p, dof, mean, std ** 2, tail, cohen_d)


def ttest_ind_from_stats(x, m2, two_tailed=True):

    m1 = np.mean(x)
    v1 = np.var(x, ddof=1)
    n1 = float(len(x))

    dof = n1 - 1
    t = np.sqrt(n1) * (m1 - m2) / v1

    cohen_d = abs(m1 - m2) / v1

    p = distributions.t.sf(t, dof)

    if two_tailed:
        p *= 2.0
        tail = "two-tailed"
    else:
        tail = "one-tailed"

    return Ttest_indResult(t, p, dof, m1, m2, v1, v1, tail, cohen_d)


def _ttest_ind_from_stats(m1, m2, v1, v2, sp, norm, dof, two_tailed):

    # Compute t statistic
    t = norm * (m1 - m2) / sp

    # Compute pvalue
    p = distributions.t.sf(abs(t), dof)

    # Cohen's d (effect size)
    cohen_d = abs(m1 - m2) / sp

    # Compute one-tailed or two tailed test
    if two_tailed:
        p *= 2.0
        tail = "two-tailed"
    else:
        tail = "one-tailed"
    return Ttest_indResult(t, p, dof, m1, m2, v1, v2, tail, cohen_d)


def _ttest_unequal_size_equal_var(m1, m2, v1, v2, n1, n2, two_tailed):

    # Standard deviation term
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    # Normalization term
    norm = np.sqrt(n1 * n2 / (n1 + n2))
    # degrees of freedom
    dof = n1 + n2 - 2

    return _ttest_ind_from_stats(m1, m2, v1, v2, sp, norm, dof, two_tailed)


def _ttest_unequal_size_unequal_var(m1, m2, v1, v2, n1, n2, two_tailed):

    v1n = v1 / n1
    v2n = v2 / n2
    sp = np.sqrt(v1n + v2n)
    norm = 1.0

    # degrees of freedom (Welch-Satterthwaite equation)
    dof = (v1n + v2n) ** 2 / ((v1n ** 2) / (n1 - 1) + (v2n ** 2) / (n2 - 1))

    return _ttest_ind_from_stats(m1, m2, v1, v2, sp, norm, dof, two_tailed)


def ttest_ind(x, y, equal_var=True, two_tailed=True):

    m1 = np.mean(x)
    m2 = np.mean(y)

    v1 = np.var(x, ddof=1)
    v2 = np.var(y, ddof=1)

    n1 = float(len(x))
    n2 = float(len(y))

    if equal_var:
        return _ttest_unequal_size_equal_var(m1, m2, v1, v2, n1, n2, two_tailed)
    else:
        return _ttest_unequal_size_unequal_var(m1, m2, v1, v2, n1, n2, two_tailed)


def wilcoxon(
    x,
    y,
    zero_method="wilcox",
    correction=False,
    alternative="two-sided",
    mode="auto",
):
    """
    Calculate the Wilcoxon signed-rank test, a non-parametric version of the
    paired T-test.

    Copied from scipy.stats, but now it returns all statistics

    """

    if mode not in ["auto", "approx", "exact"]:
        raise ValueError("mode must be either 'auto', 'approx' or 'exact'")

    if zero_method not in ["wilcox", "pratt", "zsplit"]:
        raise ValueError(
            "Zero method must be either 'wilcox' " "or 'pratt' or 'zsplit'"
        )

    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError(
            "Alternative must be either 'two-sided', " "'greater' or 'less'"
        )

    if y is None:
        d = np.asarray(x)
        if d.ndim > 1:
            raise ValueError("Sample x must be one-dimensional.")
    else:
        x, y = map(np.asarray, (x, y))
        if x.ndim > 1 or y.ndim > 1:
            raise ValueError("Samples x and y must be one-dimensional.")
        if len(x) != len(y):
            raise ValueError("The samples x and y must have the same length.")
        d = x - y

    if mode == "auto":
        if len(d) <= 25:
            mode = "exact"
        else:
            mode = "approx"

    n_zero = np.sum(d == 0)
    if n_zero > 0 and mode == "exact":
        mode = "approx"
        warnings.warn(
            "Exact p-value calculation does not work if there are "
            "ties. Switching to normal approximation."
        )

    if mode == "approx":
        if zero_method in ["wilcox", "pratt"]:
            if n_zero == len(d):
                raise ValueError(
                    "zero_method 'wilcox' and 'pratt' do not "
                    "work if x - y is zero for all elements."
                )
        if zero_method == "wilcox":
            # Keep all non-zero differences
            d = np.compress(np.not_equal(d, 0), d)

    count = len(d)
    if count < 10 and mode == "approx":
        warnings.warn("Sample size too small for normal approximation.")

    r = stats.rankdata(abs(d))
    r_plus = np.sum((d > 0) * r)
    r_minus = np.sum((d < 0) * r)

    if zero_method == "zsplit":
        r_zero = np.sum((d == 0) * r)
        r_plus += r_zero / 2.0
        r_minus += r_zero / 2.0

    # return min for two-sided test, but r_plus for one-sided test
    # the literature is not consistent here
    # r_plus is more informative since r_plus + r_minus = count*(count+1)/2,
    # i.e. the sum of the ranks, so r_minus and the min can be inferred
    # (If alternative='pratt', r_plus + r_minus = count*(count+1)/2 - r_zero.)
    # [3] uses the r_plus for the one-sided test, keep min for two-sided test
    # to keep backwards compatibility
    if alternative == "two-sided":
        T = min(r_plus, r_minus)
    else:
        T = r_plus

    if mode == "approx":
        mn = count * (count + 1.0) * 0.25
        se = count * (count + 1.0) * (2.0 * count + 1.0)

        if zero_method == "pratt":
            r = r[d != 0]
            # normal approximation needs to be adjusted, see Cureton (1967)
            mn -= n_zero * (n_zero + 1.0) * 0.25
            se -= n_zero * (n_zero + 1.0) * (2.0 * n_zero + 1.0)

        replist, repnum = find_repeats(r)
        if repnum.size != 0:
            # Correction for repeated elements.
            se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

        se = np.sqrt(se / 24)

        # apply continuity correction if applicable
        d = 0
        if correction:
            if alternative == "two-sided":
                d = 0.5 * np.sign(T - mn)
            elif alternative == "less":
                d = -0.5
            else:
                d = 0.5

        # compute statistic and p-value using normal approximation
        z = (T - mn - d) / se
        if alternative == "two-sided":
            prob = 2.0 * distributions.norm.sf(abs(z))
        elif alternative == "greater":
            # large T = r_plus indicates x is greater than y; i.e.
            # accept alternative in that case and return small p-value (sf)
            prob = distributions.norm.sf(z)
        else:
            prob = distributions.norm.cdf(z)
    elif mode == "exact":
        # get frequencies cnt of the possible positive ranksums r_plus
        cnt = _get_wilcoxon_distr(count)
        # note: r_plus is int (ties not allowed), need int for slices below
        r_plus = int(r_plus)
        if alternative == "two-sided":
            if r_plus == (len(cnt) - 1) // 2:
                # r_plus is the center of the distribution.
                prob = 1.0
            else:
                p_less = np.sum(cnt[: r_plus + 1]) / 2 ** count
                p_greater = np.sum(cnt[r_plus:]) / 2 ** count
                prob = 2 * min(p_greater, p_less)
        elif alternative == "greater":
            prob = np.sum(cnt[r_plus:]) / 2 ** count
        else:
            prob = np.sum(cnt[: r_plus + 1]) / 2 ** count

    n = len(x)
    std_pooled = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / n)
    cohen_d = abs((np.mean(x) - np.mean(y)) / std_pooled)

    if alternative == "two_tailed":
        tail = "two-tailed"
    else:
        tail = "one-tailed"
    return WilcoxonResult(
        statistic=T,
        z=z,
        pvalue=prob,
        tail=tail,
        cohen_d=cohen_d,
    )


def pretty_print_results(results, cond1_name="A", cond2_name="B", alpha=0.01):

    # Check for significance
    if results.pvalue < alpha:
        sig = "a significant"
    else:
        sig = "no significant"

    sig += " difference at the $\\alpha = {0:.2f}$ level".format(alpha)

    if results.dof.is_integer():
        dof_s = "{0}".format(int(results.dof))
    else:
        dof_s = "{0:.2f}".format(results.dof)

    if results.pvalue >= 0.01:
        pval_string = f"{results.pvalue:.2f}"
    elif results.pvalue >= 0.001:
        pval_string = f"{results.pvalue:.3f}"
    else:
        pval_string = "<0.001"
    t_res = "$\\text{{t}}({0})={1:.2f}, p={2}, \\text{{Cohen's}}\\ d={3:.2f}$".format(
        dof_s, results.statistic, pval_string, results.cohen_d
    )

    if isinstance(results, Ttest_indResult):
        t_type = "An independent-samples"

        g1_stats = "{2} $(\\text{{mean}}={0:.2f}, \\text{{std}}={1:.2f})$".format(
            results.mean1, np.sqrt(results.var1), cond1_name
        )

        g2_stats = "{2} $(\\text{{mean}}={0:.2f}, \\text{{std}}={1:.2f})$".format(
            results.mean2, np.sqrt(results.var2), cond2_name
        )

        g_stats = "for " + g1_stats + " and " + g2_stats

    elif isinstance(results, Ttest_relResult):
        t_type = "A paired samples"

        g_stats = "$(\\text{{mean}}={0:.2f}, \\text{{std}}={1:.2f})$".format(
            results.mean, np.sqrt(results.var)
        )

    out_str = (
        "{0} {1} t-test was conducted to compare {2} and {3}. "
        "There was {4} in the scores {5}; {6}"
    ).format(t_type, results.tail, cond1_name, cond2_name, sig, g_stats, t_res)

    print(out_str)
    return out_str, t_res, g_stats


if __name__ == "__main__":

    # Compare to scipy.stats example
    rvs1 = stats.norm.rvs(loc=5, scale=10, size=500)
    rvs2 = stats.norm.rvs(loc=5, scale=10, size=500)
    results_scipy = stats.ttest_ind(rvs1, rvs2)
    results_scipy2 = stats.ttest_ind(rvs1, rvs2, equal_var=False)
    results = ttest_ind(rvs1, rvs2)
    results_2 = ttest_ind(rvs1, rvs2, equal_var=False)

    results_rel = stats.ttest_rel(rvs1, rvs2)
    results_rel2 = ttest_rel(rvs1, rvs2)

    pretty_print_results(results_2)

    pretty_print_results(results_rel2)
    # rvs3 = stats.norm.rvs(loc=5, scale=20, size=500)
    # results_scipy = stats.ttest_ind(rvs1, rvs3)

    # results_scipy2 = stats.ttest_ind(rvs1, rvs3, equal_var=False)

    # rvs4 = stats.norm.rvs(loc=5, scale=20, size=100)
    # results_scipy = stats.ttest_ind(rvs1, rvs4)

    # results_scipy2 = stats.ttest_ind(rvs1, rvs4, equal_var=False)

    # rvs5 = stats.norm.rvs(loc=8, scale=20, size=100)
    # results_scipy = stats.ttest_ind(rvs1, rvs5)

    # results_scipy2 = stats.ttest_ind(rvs1, rvs5, equal_var=False)
