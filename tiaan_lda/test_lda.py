import pandas as pd
import numpy as np

from lda import LDA


def main():
    df = pd.read_csv("../data/iris.csv")
    X = df.drop("variety", axis=1).to_numpy()
    X_t = X.T
    y = df["variety"].to_numpy()
    
    print(f"X_t shape: {X_t.shape}")
    print(f"y shape: {y.shape}")
    
    test_lda = LDA()
    Z = test_lda.fit_transform(X_t, y)
    print(test_lda.explained_variance_ratio)
    
    
    
if __name__ == "__main__":
    main()