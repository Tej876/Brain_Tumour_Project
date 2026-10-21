"""Reusable DIP preprocessing functions for brain MRI images."""
import cv2
import numpy as np

DEFAULT_THRESHOLDS = {'contrast_threshold': 0.13575222715735435, 'noise_threshold': 0.0018169281538575888, 'sharpness_threshold': 0.0005936163215665147}

def validate_image(image):
    image = np.asarray(image, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError("Expected a grayscale image.")

    if not np.isfinite(image).all():
        raise ValueError("Image contains invalid values.")

    if image.max() > 1:
        image = image / 255.0

    return np.clip(image, 0, 1)


def apply_clahe(image, clip_limit=2.0):
    image = validate_image(image)
    image_u8 = np.round(image * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8)
    )

    return clahe.apply(image_u8).astype(np.float32) / 255.0


def apply_gaussian(image, kernel_size=5, sigma=0):
    image = validate_image(image)

    result = cv2.GaussianBlur(
        image, (kernel_size, kernel_size), sigmaX=sigma
    )

    return np.clip(result, 0, 1)


def apply_median(image, kernel_size=3):
    image = validate_image(image)
    image_u8 = np.round(image * 255).astype(np.uint8)

    result = cv2.medianBlur(image_u8, kernel_size)

    return result.astype(np.float32) / 255.0


def apply_unsharp_mask(image, amount=0.5, sigma=1.0):
    image = validate_image(image)

    blurred = cv2.GaussianBlur(
        image, (0, 0), sigmaX=sigma
    )

    sharpened = image + amount * (image - blurred)

    return np.clip(sharpened, 0, 1)


def measure_quality(image):
    image = validate_image(image)

    # Global contrast
    contrast = float(np.std(image))

    # Robust estimate of high-frequency noise
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    residual = image - blurred

    noise = float(
        1.4826 * np.median(
            np.abs(residual - np.median(residual))
        )
    )

    # Sharpness proxy
    laplacian = cv2.Laplacian(image, cv2.CV_32F)
    sharpness = float(np.var(laplacian))

    return {
        "contrast": contrast,
        "noise": noise,
        "sharpness": sharpness,
    }


def fixed_preprocess(image):
    result = apply_clahe(image)
    result = apply_median(result, kernel_size=3)
    result = apply_unsharp_mask(result, amount=0.5)

    return result


def adaptive_preprocess(
    image,
    contrast_threshold,
    noise_threshold,
    sharpness_threshold,
):
    result = validate_image(image)
    operations = []

    quality = measure_quality(result)

    if quality["contrast"] < contrast_threshold:
        result = apply_clahe(result)
        operations.append("CLAHE")

    quality = measure_quality(result)

    if quality["noise"] > noise_threshold:
        result = apply_median(result, kernel_size=3)
        operations.append("Median filtering")

    quality = measure_quality(result)

    if quality["sharpness"] < sharpness_threshold:
        result = apply_unsharp_mask(result, amount=0.5)
        operations.append("Unsharp masking")

    if not operations:
        operations.append("No change")

    return result, operations

