import pandas as pd
import numpy as np
import pandas_flavor as pf
from scipy.stats import pearsonr

##################################################################################

@pf.register_dataframe_method
def numeric_features_matrix(x):
    """
    Takes a pandas dataframe that includes all feature types and returns 
    a numpy array of only continuous numeric features, where all inf, -inf,
    and nan values are replaced with 0.

    Args:
        x ([pandas.DataFrame]):
            A dataframe that includes all features for consideration for 
            model development.

    Returns:
        [numpy.array]: 
            A square matrix of all numeric feature values.
    """

    # Numeric Features
    x = x.select_dtypes(include=np.number)

    x.replace([np.inf, -np.inf], np.nan, inplace=True)

    x = np.where(np.isnan(x), 0, x)

    return x

###############################################################################

@pf.register_dataframe_method
def numeric_features_dataframe(x):
    """
    Takes a pandas dataframe that includes all feature types and returns 
    a pandas dataframe that includes only numeric features.

    Args:
        x ([pandas.DataFrame]):
            A dataframe that includes all features for consideration for 
            model development.

    Returns:
        [pandas.DataFrame]: 
            A dataframe that includes only numeric features being considered
            for model development.
    """

    # Numeric Features
    all_numeric_features = x.select_dtypes(include=np.number)

    return all_numeric_features

###############################################################################

@pf.register_dataframe_method
def condition_number(x):
    """
    Calculates the condition number from a correlation matrix 
    leveraging eigenvalues as per Callaghan, Karen J. and Chen, 
    Jie (2008) "Revisiting the Collinear Data Problem: An 
    Assessment of Estimator 'Ill-Conditioning' in Linear Regression,
    "Practical Assessment, Research, and Evaluation": Vol. 13, Article 5

    Args:
        x ([numpy.array]): 
            A correlation matrix of all training features for consideration 
            in model development.

    Returns:
        [dtype('float64')]:
            Returns the condition number, which is the
            ratio of the largest to the smallest eigenvalue (denoted by λ), 
            Condition Number = (λ max/λ min). 
    """
    eigenValues,eigenVectors = np.linalg.eig(x) 

    return abs(max(eigenValues)/min(eigenValues))

################################################################################

@pf.register_dataframe_method
def remove_multicollinearity_by_cn_threshold(x, CN_threshold = 1000):
    """
    Function that leverages the condition number (CN) of a correlation matrix
    of features, the correlation matrix of the features, and removes all
    those that are above a user-defined threshold (CN). If (CN < 100) the degree
    of collinearity is generally considered weak. If (100 < CN < 1000) collinearity
    is considered moderate to strong, and CN > 1000 is considered severe. For 
    reference these values are based on Gujarati, D. N. (2002). Basic econometrics. 
    New York: McGraw-Hill. 

    Args:
        x ([numpy.array]): 
            An array of all X features for consideration in model development.
        CN_threshold (int, optional): 
            Defaults to 100.

    Returns:
        [numpy.array]: 
            An array of all X features that did not exhibit collinearity/multicollinearity
            is returned, with all those features that exhibited collinearity removed.
    """
    correlation_matrix = np.corrcoef(x.T)
    
    if (condition_number(correlation_matrix) < CN_threshold or x.shape[1]<=1): 
        return x 

    features_to_remove = np.argmax([max(correlation_matrix[i+1:,i]) for i in range(x.shape[1]-1)] )
    
    features_to_keep = [True]*x.shape[1]
    
    features_to_keep[features_to_remove] = False
    
    return remove_multicollinearity_by_cn_threshold(x[:,features_to_keep])
