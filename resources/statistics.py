import numpy as np
from scipy import stats
def mean_diff_test(a, b, check_col, sample_size=2000):
    ## Define 2 random distributions
    #Sample Size
    a = a[[check_col]]
    b = b[[check_col]]

    ## Calculate the Standard Deviation
    # Calculate the variance to get the standard deviation
    a = np.array(a)
    b = np.array(b)

    a = np.array(a).astype(np.float)
    b = np.array(b).astype(np.float)

    # For unbiased max likelihood estimate we have to divide the var by N-1, and therefore the parameter ddof = 1
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)

    # std deviation
    s = np.sqrt((var_a + var_b)/2)

    ## Calculate the t-statistics
    t = (a.mean() - b.mean())/(s*np.sqrt(2/sample_size))



    # Compare with the critical t-value
    #Degrees of freedom
    df = 2 * sample_size - 2

    # p-value after comparison with the t
    p = 1 - stats.t.cdf(t, df=df)


    print("t = " + str(t))
    print("p = " + str(2*p))
    # Note that we multiply the p value by 2 because its a twp tail t-test
    # You can see that after comparing the t statistic with the critical t value (computed internally)
    # we get a good p value of 0.0005 and thus we reject the null hypothesis and thus it proves that the mean of the
    # two distributions are different and statistically significant.


    # Cross Checking with the internal scipy function
    t2, p2 = stats.ttest_ind(a,b)
    print("t = " + str(t2))
    print("p = " + str(2*p2))

