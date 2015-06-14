'''
Dimensionality reduction
'''

from sklearn.decomposition import TruncatedSVD

class DR(object):
    '''
    The base class for dimensionality reduction
    '''
    
    def __init__(self, num_dimensions=None):
        '''
        num_dimensions -- Number of dimensions to keep. If None, keep all useful dimensions where usefullness is measured by the underlying DR method.
        '''
        self.num_dimensions=num_dimensions
    
    def fit(self, X, y=None):
        '''Fit the necessary params needed'''
        raise NotImplementedError('Needs to be overriden by a subclass')
    
    def transform(self, X):
        '''Transform based on the params fitted'''
        raise NotImplementedError('Needs to be overriden by a subclass')

class NoDR(DR):
    '''
    Performs no fitting or transformation
    '''    
    def fit(self, X, y=None):
        pass
    
    def transform(self, X):
        return X

class TruncatedSVDDR(DR):
    '''
    Performs TruncatedSVD. For text, this is equivalent to LSA.
    See http://scikit-learn.org/stable/modules/decomposition.html
    '''
    def __init__(self, num_dimensions=None):
        super(TruncatedSVDDR, self).__init__(num_dimensions)
        self.truncated_svd = None
        self._fitted = False
        # These vars are used for speed reasons
        self.previously_transformed = None
        self.previous_transformation = None
    
    def fit(self, X, y=None):
        if not self._fitted:
            nd = X.shape[1]
            if self.num_dimensions:
                nd = self.num_dimensions
            self.truncated_svd = TruncatedSVD(n_components=nd)
            self.truncated_svd.fit(X)
            #for val in self.truncated_svd.explained_variance_ratio_:
            #    print val
            self._fitted = True
    
    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("First, the fit function need to be called")
        else:
            
            #If we applied a transformation on this same X before, don't do a new transformation; simply return the previous transformation
            if X is self.previously_transformed:
                return self.previous_transformation
            else:
                self.previous_transformation = self.truncated_svd.transform(X)
                self.previously_transformed = X
                return self.previous_transformation