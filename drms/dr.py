'''
Dimensionality reduction
'''

class DR(object):
    '''
    The base class for dimensionality reduction
    '''
    
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

class PCADR(DR):
    '''
    Performs PCA
    '''
    def __init__(self):
        self._fitted = False
    
    def fit(self, X, y=None):
        if not self._fitted:
            raise NotImplementedError('Not implemented yet :(')
    
    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("First, the fit function need to be called")
        else:
            raise NotImplementedError('Not implemented yet :(')