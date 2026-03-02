import numpy as np

class LDA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        
        self.class_labels = None
        self.global_mean = None
        self.class_means = None
        self.transformation_matrix = None
        self.explained_variance_ratio = None
        
    def fit(self, X, y):
        # Validate Model Parameters
        self._validate_input(X, y)
        
        # Compute means
        self.global_mean = np.mean(X, axis=1)
        self.class_means = self._compute_class_means(X, y)

        # Compute Scatter Matrices
        S_w = self._compute_within_scatter(X, y)
        class_counts = self._compute_class_counts(X, y)
        S_b = self._compute_between_scatter(X.shape[0], class_counts)
        
        # Compute the Whitening matrix
        P = self._compute_whitening(S_w)
        # Whiten S_b
        S_b_white = P @ S_b @ P.T
        # Compute directions of maximal between-class variance.
        V_k = self._eigendecompose_S_b_white(S_b_white)
        # Compute transformation matrix
        self.transformation_matrix = P.T @ V_k
        
        
    def transform(self, X_new):
        # Validate that LDA has been fit
        if self.global_mean is None or self.transformation_matrix is None:
            raise Exception("Please fit LDA before transforming data")
        
        # Centre new data using training global mean
        D_new = X_new - np.outer(self.global_mean, np.ones(shape=X_new.shape[1]))
        
        # Project into LDA space
        Z = self.transformation_matrix.T @ D_new
        return Z
    
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
        
    
    
    def _validate_input(self, X, y):
        if np.ndim(X) != 2:
            raise ValueError(f"LDA expects X to be of 2 dimensions, got {np.ndim(X)}")
        if np.ndim(y) != 1:
            raise ValueError(f"LDA expects y to be of 1 dimension, got {np.ndim(y)}")
        
        self.class_labels = np.unique(y)
        
        max_components = min(X.shape[0], len(self.class_labels)-1)
        if self.n_components is None:
             self.n_components = max_components
        elif self.n_components > max_components:
            raise ValueError(f"LDA can produce at most {max_components}, got {self.n_components}")


    def _compute_class_means(self, X, y):
        class_means = []
        for c in self.class_labels:
            mask = (y == c)
            X_c = X[:, mask]
            class_means.append(np.mean(X_c, axis=1))
            
        return np.column_stack(class_means)
    
    
    def _compute_within_scatter(self, X, y):
        S_w = np.zeros((X.shape[0], X.shape[0]))
        for i, c in enumerate(self.class_labels):
            mask = (y == c)
            X_c = X[:, mask]
            D_c = X_c - np.outer(self.class_means[:, i], np.ones(X_c.shape[1]).T)
            S_w += D_c @ D_c.T
        
        return S_w
            
            
    def _compute_class_counts(self, X, y):
        class_counts = []
        for c in self.class_labels:
            mask = (y == c)
            X_c = X[:, mask]
            class_counts.append(X_c.shape[1])
            
        return class_counts
    
    
    def _compute_between_scatter(self, d, class_counts):
        S_b = np.zeros((d,d))
        for i in range(len(self.class_labels)):
            m_c = self.class_means[:, i] - self.global_mean
            S_b += class_counts[i] * np.outer(m_c, m_c.T)
    
        return S_b
    
    
    def _compute_whitening(self, S_w, tol=1e-12):
        # eigendecompose S_w
        eigenvalues, eigenvectors = np.linalg.eigh(S_w)
        
        # remove near 0 eigenvalues
        tol_lvl = tol * max(eigenvalues)
        keep = eigenvalues > tol_lvl
        eigenvalues = eigenvalues[keep]
        eigenvectors = eigenvectors[:, keep]
        
        # build whitening matrix
        A = np.diag(np.ones(shape=len(eigenvalues)) / np.sqrt(eigenvalues))
        P = A @ eigenvectors.T
        
        return P
    
    
    def _eigendecompose_S_b_white(self, S_b_white):
        # eigendecompose S_w
        eigenvalues, eigenvectors = np.linalg.eigh(S_b_white)
        
        # compute total explained variance
        total_variance = sum(eigenvalues)
        
        # sort eigenvalues and vectors in descending order
        eigenvalues = np.flip(eigenvalues)
        eigenvectors = np.fliplr(eigenvectors)
        
        # keep n_component first eigenvalues
        eigenvalues_clipped = eigenvalues[:self.n_components]
        # keep n_component first eigenvectors
        V_k = eigenvectors[:, :self.n_components]
        
        # compute explained variance ratios
        self.explained_variance_ratio = np.zeros(self.n_components)
        for i, eig in enumerate(eigenvalues_clipped):
            self.explained_variance_ratio[i] = eig / total_variance
        
        return V_k
    
    
    
    
    
    