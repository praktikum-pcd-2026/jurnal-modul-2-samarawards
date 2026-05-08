# pcdlib.py
# Library Pengolahan Citra Digital
# Author: Ara ✨

import numpy as np



# =========================================================
# UTILITAS DASAR
# =========================================================

def _clip(val):
    return np.clip(val, 0, 255).astype(np.uint8)


def _is_grayscale(img):
    return len(img.shape) == 2


def _is_rgb(img):
    return len(img.shape) == 3 and img.shape[2] == 3


def rgb_to_grayscale(img):
    """
    Konversi RGB ke grayscale
    """
    if not _is_rgb(img):
        raise ValueError("Input harus citra RGB")

    gray = (
        0.299 * img[:, :, 0] +
        0.587 * img[:, :, 1] +
        0.114 * img[:, :, 2]
    )

    return _clip(gray)


def grayscale_to_rgb(img):
    """
    Konversi grayscale ke RGB
    """
    if not _is_grayscale(img):
        raise ValueError("Input harus grayscale")

    return np.stack([img, img, img], axis=2)


# =========================================================
# HISTOGRAM
# =========================================================

def histogram_grayscale(img):
    hist = np.zeros(256, dtype=int)

    for pixel in img.flatten():
        hist[pixel] += 1

    return hist


def cumulative_histogram(hist):
    cdf = np.cumsum(hist)
    return cdf


# =========================================================
# 1. EKUALISASI GRAYSCALE
# =========================================================

def equalization_grayscale(img):
    """
    Ekualisasi histogram grayscale
    """

    if not _is_grayscale(img):
        raise ValueError("Input harus grayscale")

    hist = histogram_grayscale(img)

    cdf = cumulative_histogram(hist)

    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

    cdf_normalized = cdf_normalized.astype(np.uint8)

    result = cdf_normalized[img]

    return result


# =========================================================
# 2. EKUALISASI RGB
# =========================================================

def equalization_rgb(img):
    """
    Ekualisasi histogram RGB per channel
    """

    if not _is_rgb(img):
        raise ValueError("Input harus RGB")

    result = np.zeros_like(img)

    for c in range(3):
        result[:, :, c] = equalization_grayscale(img[:, :, c])

    return result


# =========================================================
# SPESIFIKASI HISTOGRAM
# =========================================================

def _create_mapping(source_hist, target_hist):

    source_cdf = cumulative_histogram(source_hist)
    target_cdf = cumulative_histogram(target_hist)

    source_cdf = source_cdf / source_cdf[-1]
    target_cdf = target_cdf / target_cdf[-1]

    mapping = np.zeros(256, dtype=np.uint8)

    for i in range(256):

        diff = np.abs(target_cdf - source_cdf[i])

        mapping[i] = np.argmin(diff)

    return mapping


# =========================================================
# 3. SPESIFIKASI GRAYSCALE TO GRAYSCALE
# =========================================================

def specification_gray_to_gray(source, target):

    if not _is_grayscale(source):
        raise ValueError("Source harus grayscale")

    if not _is_grayscale(target):
        raise ValueError("Target harus grayscale")

    source_hist = histogram_grayscale(source)
    target_hist = histogram_grayscale(target)

    mapping = _create_mapping(source_hist, target_hist)

    result = mapping[source]

    return result


# =========================================================
# 4. SPESIFIKASI GRAYSCALE TO RGB
# HASIL RGB
# =========================================================

def specification_gray_to_rgb(source, target):

    if not _is_grayscale(source):
        raise ValueError("Source harus grayscale")

    if not _is_rgb(target):
        raise ValueError("Target harus RGB")

    source_rgb = grayscale_to_rgb(source)

    result = np.zeros_like(source_rgb)

    for c in range(3):

        source_hist = histogram_grayscale(source_rgb[:, :, c])
        target_hist = histogram_grayscale(target[:, :, c])

        mapping = _create_mapping(source_hist, target_hist)

        result[:, :, c] = mapping[source_rgb[:, :, c]]

    return result


# =========================================================
# 5. SPESIFIKASI RGB TO GRAYSCALE
# HASIL GRAYSCALE
# =========================================================

def specification_rgb_to_gray(source, target):

    if not _is_rgb(source):
        raise ValueError("Source harus RGB")

    if not _is_grayscale(target):
        raise ValueError("Target harus grayscale")

    source_gray = rgb_to_grayscale(source)

    source_hist = histogram_grayscale(source_gray)
    target_hist = histogram_grayscale(target)

    mapping = _create_mapping(source_hist, target_hist)

    result = mapping[source_gray]

    return result


# =========================================================
# 6. SPESIFIKASI RGB TO RGB
# =========================================================

def specification_rgb_to_rgb(source, target):

    if not _is_rgb(source):
        raise ValueError("Source harus RGB")

    if not _is_rgb(target):
        raise ValueError("Target harus RGB")

    result = np.zeros_like(source)

    for c in range(3):

        source_hist = histogram_grayscale(source[:, :, c])
        target_hist = histogram_grayscale(target[:, :, c])

        mapping = _create_mapping(source_hist, target_hist)

        result[:, :, c] = mapping[source[:, :, c]]

    return result


# =========================================================
# 7. MASKING GRAYSCALE TO GRAYSCALE
# =========================================================

def masking_gray_to_gray(image, mask):
    """
    image  : grayscale
    mask   : grayscale
    hasil  : grayscale
    """

    if not _is_grayscale(image):
        raise ValueError("Image harus grayscale")

    if not _is_grayscale(mask):
        raise ValueError("Mask harus grayscale")

    if image.shape != mask.shape:
        raise ValueError("Ukuran image dan mask harus sama")

    normalized_mask = mask / 255.0

    result = image * normalized_mask

    return _clip(result)


# =========================================================
# 8. MASKING GRAY TO RGB
# HASIL GRAYSCALE
# =========================================================

def masking_gray_to_rgb_gray_result(image, mask):
    """
    image  : grayscale
    mask   : RGB
    hasil  : grayscale
    """

    if not _is_grayscale(image):
        raise ValueError("Image harus grayscale")

    if not _is_rgb(mask):
        raise ValueError("Mask harus RGB")

    mask_gray = rgb_to_grayscale(mask)

    if image.shape != mask_gray.shape:
        raise ValueError("Ukuran image dan mask harus sama")

    normalized_mask = mask_gray / 255.0

    result = image * normalized_mask

    return _clip(result)


# =========================================================
# 9. MASKING GRAY TO RGB
# HASIL RGB
# =========================================================

def masking_gray_to_rgb_rgb_result(image, mask):
    """
    image  : grayscale
    mask   : RGB
    hasil  : RGB
    """

    if not _is_grayscale(image):
        raise ValueError("Image harus grayscale")

    if not _is_rgb(mask):
        raise ValueError("Mask harus RGB")

    image_rgb = grayscale_to_rgb(image)

    if image_rgb.shape != mask.shape:
        raise ValueError("Ukuran image dan mask harus sama")

    normalized_mask = mask / 255.0

    result = image_rgb * normalized_mask

    return _clip(result)