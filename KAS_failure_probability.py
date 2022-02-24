from math import *
from functools import reduce
import operator as op
import numpy as np
from sympy.utilities.iterables import multiset_permutations
from mpmath import *


dim = 2048
k = 3
q = 1099511590913
subg_param = sqrt(k / 2)

tau = 11

bound_CC = (q ** 2 / (4 * tau * subg_param))**2  # Bound n*mu+beta to use
# in Cramer-Chernoff Ineq.
n = 2 * dim / 8  # Number of samples in Cramer-Chernoff Ineq.

subg_value = sqrt(bound_CC) * subg_param * tau
p_subg = exp(-tau ** 2 / 2)


# Check each distribution phi_v (far far slower...will take days to run)
# experimentally, they are always (very close to) equal probably due to
# symmetries

# List of type 1 Voronoi-relevant vectors
v1_list = list(multiset_permutations([q/2, q/2, 0, 0, 0, 0, 0, 0])) + \
     list(multiset_permutations([-q/2, q/2, 0, 0, 0, 0, 0, 0])) + \
     list(multiset_permutations([-q/2, -q/2, 0, 0, 0, 0, 0, 0]))

# List of type 2 Voronoi-relevant vectors
v2_list = list(multiset_permutations([q/4, q/4, q/4, q/4, q/4, q/4, q/4,
                                      q/4])) + \
     list(multiset_permutations([-q/4, -q/4, q/4, q/4, q/4, q/4, q/4,
                                 q/4])) + \
     list(multiset_permutations([-q/4, -q/4, -q/4, -q/4, q/4, q/4, q/4,
                                 q/4])) + \
     list(multiset_permutations([-q/4, -q/4, -q/4, -q/4, -q/4, -q/4, q/4,
                                 q/4])) + \
     list(multiset_permutations([-q/4, -q/4, -q/4, -q/4, -q/4, -q/4, -q/4,
                                 -q/4]))

# List of all Voronoi-relevant vectors
v_list = v1_list + v2_list

# Individual type 1 and 2 Voronoi-relevant vectors to try
v_1 = [[q/2, q/2, 0, 0, 0, 0, 0, 0]]
v_2 = [[q/4, q/4, q/4, q/4, q/4, q/4, q/4, q/4]]


# Helps us to do polynomial multiplication (mod y^8+1) with vectors
def rot(v):
    return [-v[7], v[0], v[1], v[2], v[3], v[4], v[5], v[6]]


# Construct the binomial law
supp = range(-k, k + 1)


def binomial(n, r):
    r = min(r, n - r)
    if r == 0: return 1
    numer = reduce(op.mul, range(n, n - r, -1))
    denom = reduce(op.mul, range(1, r + 1))
    return numer // denom


def pdf_binom(x):
    return binomial(2 * k, x + k) / 2. ** (2 * k)

# Working with >=15 most significant digits
print("target bound =", subg_value)
print("q^2/4 =", (q**2/4))
print ("Correct ?", subg_value/10**7 <= (q**2/4)/10**7)
assert subg_value/10**7 <= (q**2/4)/10**7

# Union bound on each application of Chernoff-Cramer
Union_CC = 0.

for v in v_1:
    # Construct a 4*4 matrix associated to v
    v1 = v
    v2 = rot(v1)
    v3 = rot(v2)
    v4 = rot(v3)
    v5 = rot(v4)
    v6 = rot(v5)
    v7 = rot(v6)
    v8 = rot(v7)
    mat = np.matrix([v1, v2, v3, v4, v5, v6, v7, v8])

    # Initialize a table to store the pdf of \varphi_v
    # Analysing the output of certain runs, shows that certain values
    # appear repetitively. Therefore, we take the first
    # 6 most significant digits of the number to classify them.
    # Otherwise we will run out of memory to try and store each value
    # in a table of that size.

    ima = 1000000
    varphi_v_Table = [0 for i in range(ima)]
    values_v_Table = [0 for i in range(ima)]

    # Brute force computation of the pdf of || x*v ||^2
    s = 0.
    for x1 in supp:
        for x2 in supp:
            for x3 in supp:
                for x4 in supp:
                    for x5 in supp:
                        for x6 in supp:
                            for x7 in supp:
                                for x8 in supp:
                                    x = np.array([x1, x2, x3, x4, x5,
                                                  x6, x7, x8])
                                    mx = x * mat
                                    s = mx[0, 0] ** 2 + mx[0, 1] ** 2 \
                                        + mx[0, 2] ** 2 + mx[0, 3] ** 2+\
                                         mx[0, 4] ** 2 + mx[0, 5] ** 2 \
                                        + mx[0, 6] ** 2 + mx[0, 7] ** 2
                                    p = pdf_binom(x1) * pdf_binom(x2) * \
                                        pdf_binom(x3) * pdf_binom(x4) * \
                                        pdf_binom(x5) * pdf_binom(x6) * \
                                        pdf_binom(x7) * pdf_binom(x8)
                                    # To run v2, need to change this to
                                    # (10**21)
                                    values_v_Table[round(s/(10**20))] = s
                                    varphi_v_Table[round(s/(10**20))] += p

    avg = sum([values_v_Table[i] * varphi_v_Table[i] for i in range(ima)])
    print("varphi_v has average ", avg)

    def Moment(t):
        return sum([varphi_v_Table[i] * exp(t * (values_v_Table[i] - avg))
                    for i in range(ima)])


    def ChernoffCramer(n, t, bound):
        beta = bound - n * avg
        p = exp(- t * beta + n * log(Moment(t)))
        return p


    t_CC = 1.19e-43
    p_CC = ChernoffCramer(n, t_CC, bound_CC)

    print("Chernoff-Cramer Bound", p_CC, " = 2^", log(p_CC) / log(2))


    Union_CC = 240 * p_CC

print()
print("####")
print("#### Conclusion of Lemma 5.4 :")
print("####")

print("with parameter k=",k, "and parameter n = ",n,
      "(V_{v,K} has dimension 8n)")
print("The bound ||V_{v,K}||_2^2 <=", bound_CC, " fails with probability "
      "less than 2^", log(Union_CC)/log(2))

print("####")
print("#### Theorem 5.5 :")
print("####")

print("subgaussian parameter sigma = ", subg_param)
print("tailcut parameter tau = ", tau)
print("tailcut value ||V_{v,K}||* sigma * tau = ", subg_value)

print("sub-gaussian tail-bound (Lemma 5.3): 2^", log(p_subg) /log(2))
print("failure proba per bit-agreement and per v")
print()

print("Final bound ||<(~s', ~s), V_{v,K}>|| <=", subg_value )
print ("fails with probability at most 2^",)
print (log(Union_CC + 2*256 * p_subg)/log(2))