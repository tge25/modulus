
import numpy as np

def compute_ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    """
    Compute the SSIM (Structural Similarity Index) between two images using NumPy.

    Args:
        img1 (np.ndarray): First image with shape (..., H, W).
        img2 (np.ndarray): Second image with shape (..., H, W).
        C1 (float): Stabilization constant for mean.
        C2 (float): Stabilization constant for variance.

    Returns:
        np.ndarray: SSIM index for each image pair.
    """
    # Ensure that the images are the same size
    assert img1.shape == img2.shape, "Input images must have the same dimensions."

    # Compute mean of each image
    mu1 = np.mean(img1, axis=(-2, -1), keepdims=True)
    mu2 = np.mean(img2, axis=(-2, -1), keepdims=True)

    # Compute variance and covariance
    sigma1_sq = np.var(img1, axis=(-2, -1), keepdims=True)
    sigma2_sq = np.var(img2, axis=(-2, -1), keepdims=True)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2), axis=(-2, -1), keepdims=True)

    # Compute SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim = numerator / denominator

    # Average SSIM over the last two dimensions (H, W)
    ssim_mean = np.mean(ssim, axis=(-2, -1))

    return ssim_mean

# Example usage:
# Assuming `image1` and `image2` are NumPy arrays with shape (..., H, W)
# ssim_value = compute_ssim(image1, image2)

