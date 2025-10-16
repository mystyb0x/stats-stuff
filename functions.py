import math

def mean_geometric_dist(p: float, support=0) -> float:
    """
    Provides the mean of a geometric distribution.

    :param p: The p-value. Must agree with 0 < p <= 1.
    :type p: float
    :param support: Specifies the parameter of the two forms of geometric distribution. 0, the default, 
        represents the total number of independent Bernoulli trials required to achieve the first success. k = {1, 2, 3, ...}.
        1 represents the number of failures that occur before the first success is achieved. k = {0, 1 , 2, ...}.
    :type support: int
    :returns: The mean of the geometric distribution.
    :rtype: float
    :raises ValueError: If p is not within 0 < p <= 1
    :raises ValueError: If support is not 0 or 1.
    """

    if not (0 < p <= 1):
        raise ValueError("p must agree with 0 < p <= 1.")
    
    if support == 0:
        mean = 1/p
    elif support == 1:
        mean = (1-p)/p
    else:
        raise ValueError("Support can only be 0 or 1.")
    return mean

def pmf_geometric_dist(p: float, k: int, support=0) -> float:
    """
    Performs a probability mass function on a geometric distribution. P(X=k)

    :param p: The p-value. Must agree with 0 < p <= 1.
    :type p: float
    :param k: The number of successes/trials. Must be an integer. k cannot be less than 1 if support is set to 0,
        and cannot be less than 0 if support is set to 1.
    :type k: int
    :param support: Specifies the parameter of the two forms of geometric distribution. 0, the default, 
        represents the total number of independent Bernoulli trials required to achieve the first success. k = {1, 2, 3, ...}.
        1 represents the number of failures that occur before the first success is achieved. k = {0, 1 , 2, ...}.
    :type support: int
    :returns: The probability of P(X=k).
    :rtype: float
    :raises ValueError: If p is not within 0 < p <= 1
    :raises ValueError: if k > 1 when support is 0, or if k > 0 when support is 1.
    :raises ValueError: If support is not 0 or 1.
    """

    if not (0 < p <= 1):
        raise ValueError("p must agree with 0 < p <= 1.")
    
    if not isinstance(k, int):
        raise ValueError("k must be an integer.")
    
    if support == 0:
        if k < 1:
            raise ValueError("k can only be a non-zero positive integer when support is 0.")
        pmf = pow(1-p, k-1) * p
        return pmf
    elif support == 1:
        if k < 0:
            raise ValueError("k can only be a positive integer or 0 when support is 1.")
        pmf = pow(1-p, k) * p
        return pmf
    else:
        raise ValueError("Support can only be 0 and 1.")

def cdf_geometric_dist(p: float, k: int, support=0) -> float:
    """Performs a cumulative density function on a geometric distribution. P=(X<=k)

    :param p: The p-value. Must agree with 0 < p <= 1.
    :type p: float
    :param k: The number of successes/trials. Must be an integer. k cannot be less than 1 if support is set to 0,
        and cannot be less than 0 if support is set to 1.
    :type k: int
    :param support: Specifies the parameter of the two forms of geometric distribution. 0, the default, 
        represents the total number of independent Bernoulli trials required to achieve the first success. k = {1, 2, 3, ...}.
        1 represents the number of failures that occur before the first success is achieved. k = {0, 1 , 2, ...}.
    :type support: int
    :returns: The probability of P(X<=k).
    :rtype: float
    :raises ValueError: If p is not within 0 < p <= 1
    :raises ValueError: if k > 1 when support is 0, or if k > 0 when support is 1.
    :raises ValueError: If support is not 0 or 1.
    """

    if not (0 < p <= 1):
        raise ValueError("p must agree with 0 < p <= 1.")
    
    if not isinstance(k, int):
        raise ValueError("k must be an integer.")
    
    if support == 0:
        if k < 1:
            raise ValueError("k can only be a non-zero positive integer when support is 0.")
        cdf = 1 - pow(1-p, k)
        return cdf
    elif support == 1:
        if k < 0:
            raise ValueError("k can only be a positive integer or 0 when support is 1.")
        cdf = 1 - pow(1-p, k+1)
        return cdf
    else:
        raise ValueError("Support can only be 0 and 1.")

def mean_binomial_dist(n: int, p: float) -> float:
    """Provides the mean of a binomial distribution.

    :param n: The sample size.
    :type n: int
    :param p: The p-value.
    :type p: float
    :returns: The mean of the binomial distribution.
    :rtype: float
    :raises ValueError: if n is not a non-zero positive integer.
    :raises ValueError: if p is not within 0 <= p <= 1.
    """

    if not (0 <= p <= 1):
        raise ValueError("p must agree with 0 <= p <= 1.")
    
    if not isinstance(n, int):
        raise ValueError("n must be a non-zero positive integer.")
    
    if n <= 0:
        raise ValueError("n must be a non-zero positive integer.")
    
    mean = n * p
    return mean

def pmf_binomial_dist(n: int, p: float, k: int) -> float:
    """Performs a probability mass function on a binomial distribution. P(X=k)
    
    :param n: The sample size.
    :type n: int
    :param p: The p-value.
    :type p: float
    :param k: The number of successes.
    :type k: int
    :returns: The probability of P(X=k)
    :rtype: float
    :raises ValueError: if n is not a non-zero positive integer.
    :raises ValueError: if p is not within 0 <= p <= 1.
    :raises ValueError: if k is not 0 or a positive integer.
    """

    if not isinstance(n, int):
        raise ValueError("n must be a non-zero positive integer.")
    
    if n <= 0:
        raise ValueError("n must be a non-zero positive integer.")
    
    if not (0 <= p <= 1):
        raise ValueError("p must agree with 0 <= p <= 1.")
    
    if k < 0:
        raise ValueError("k must be 0 or a positive integer.")
    
    if not isinstance(k, int):
        raise ValueError("k must be 0 or a positive integer.")
    
    if k > n:
        raise ValueError("k cannot be larger than n.")
    
    nCk = math.factorial(n) / (math.factorial(k) * math.factorial(n-k))
    pmf = nCk * pow(p, k) * pow(1-p, n-k)
    return pmf