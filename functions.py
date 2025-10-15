def mean_geometric_dist(p: float, support=0) -> float:
    """Gets the mean of a geometric distribution.

    Arguments:
        p (float): The p-value.
        support (int): Specifies the parameter of the two forms of geometric distribution. 0, the default,
            represents the total number of independent Bernoulli trials required to achieve the first success. k = {1, 2, 3, ...}.
            1 represents the number of failures that occur before the first success is achieved. k = {0, 1 , 2, ...}.

    Returns:
        The mean of the geometric distribution.
    """

    if not (0 < p <= 1):
        raise ValueError("p must agree with 0 < p <= 1.")
    
    if support == 0:
        mean = 1/p
    elif support == 1:
        mean = (1-p)/p
    else:
        raise ValueError("Support can only be 0 or 1.")

def pmf_geometric_dist(p: float, k: int, support=0) -> float:
    """Performs a probability mass function on a geometric distribution. P(X=k)

    Arguments:
        p (float): The p-value.
        k (int): The amount of trials/failures.
        support (int): Specifies the parameter of the two forms of geometric distribution. 0, the default,
            represents the total number of independent Bernoulli trials required to achieve the first success. k = {1, 2, 3, ...}.
            1 represents the number of failures that occur before the first success is achieved. k = {0, 1 , 2, ...}.

    Returns:
        The probability of P(X=k).      
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

    Arguments:
        p (float): The p-value.
        k (int): The number of trials/failures.
        support (int): Specifies the parameter of the two forms of geometric distribution. 0, the default,
            represents the total number of independent Bernoulli trials required to achieve the first success. k = {1, 2, 3, ...}.
            1 represents the number of failures that occur before the first success is achieved. k = {0, 1 , 2, ...}.

    Returns:
        The probability of P=(X<=k)
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
