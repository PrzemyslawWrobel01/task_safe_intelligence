import numpy as np
import pandas as pd

from .constants import DATA_DIR


def generate_linearly_separable_3d(
    filepath, n_samples=1000, noise=0.1, random_state=None
):
    """Generate linearly separable data in 3D space with binary labels and save to CSV.

    Parameters:
    -----------
    n_samples : int, default=1000
        Number of samples to generate
    noise : float, default=0.1
        Standard deviation of Gaussian noise to add
    random_state : int or None, default=None
        Random state for reproducibility
    filepath : str, default='linearly_separable_3d.csv'
        Path where the CSV file will be saved

    Returns:
    --------
    X : ndarray of shape (n_samples, 3)
        The 3D feature matrix
    y : ndarray of shape (n_samples,)
        Binary labels (0 or 1)
    plane : tuple
        (a, b, c, d) coefficients of the separating plane ax + by + cz + d = 0
    """

    if random_state is not None:
        np.random.seed(random_state)

    # Generate random plane coefficients
    a, b, c = np.random.randn(3)
    d = np.random.randn()

    # Generate random points in 3D space
    X = np.random.randn(n_samples, 3)

    # Calculate distance from points to plane
    distances = (a * X[:, 0] + b * X[:, 1] + c * X[:, 2] + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)

    # Add some noise to make it more realistic
    distances += np.random.normal(0, noise, n_samples)

    # Assign labels based on which side of the plane points lie
    y = (distances >= 0).astype(int)

    # Add more separation between classes by moving points further from the plane
    X += 2 * np.outer(distances, [a, b, c]) / (a ** 2 + b ** 2 + c ** 2)

    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        'x1': X[:, 0],
        'x2': X[:, 1],
        'x3': X[:, 2],
        'label': y
    })

    # Save to CSV
    df.to_csv(filepath, index=False)

    # Save plane coefficients to a separate CSV
    plane_df = pd.DataFrame({
        'coefficient': ['a', 'b', 'c', 'd'],
        'value': [a, b, c, d]
    })
    plane_filepath = str(filepath).replace('.csv', '_plane.csv')
    plane_df.to_csv(plane_filepath, index=False)

    print(f"Data saved to: {filepath}")
    print(f"Plane coefficients saved to: {plane_filepath}")

    return X, y, (a, b, c, d)


# Example usage:
if __name__ == "__main__":
    # Generate data and save to CSV
    X, y, plane = generate_linearly_separable_3d(
        filepath=DATA_DIR / '3d_data.csv',
        n_samples=10000,
        noise=0.001,
        random_state=42,
    )

    # Print some basic statistics
    print(f"\nNumber of class 0 samples: {np.sum(y == 0)}")
    print(f"Number of class 1 samples: {np.sum(y == 1)}")
    print(f"Separating plane coefficients (a, b, c, d): {plane}")

    # Verify the saved data
    loaded_data = pd.read_csv(DATA_DIR / '3d_data.csv')
    print(f"\nShape of saved data: {loaded_data.shape}")
    print("\nFirst few rows of the saved data:")
    print(loaded_data.head())
