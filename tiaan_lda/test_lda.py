"""Quick manual test for the LDA dimensionality-reduction implementation.

This script follows the CS315 data convention:
- rows represent features
- columns represent observations
"""

import pandas as pd

from lda import LDA


def main():
    """Load Iris data, reshape to module convention, and run LDA.

    Parameters:
    -------------------
    None

    Returns:
    --------------
    None
    """
    df = pd.read_csv("../data/iris.csv")
    X = df.drop("variety", axis=1).to_numpy()
    X_t = X.T
    y = df["variety"].to_numpy()

    print(f"X_t shape: {X_t.shape}")
    print(f"y shape: {y.shape}")

    test_lda = LDA()
    Z = test_lda.fit_transform(X_t, y)
    print(test_lda.explained_variance_ratio)
    print(f"Projected shape: {Z.shape}")


if __name__ == "__main__":
    main()
