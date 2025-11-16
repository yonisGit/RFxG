import numpy as np
from sklearn.metrics import auc


def mask_top_pixels(image, saliency_map, alpha, masking_method='black'):
    """
    Masks the top-alpha most salient pixels in the image.
    """
    if len(image.shape) == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    elif image.shape[0] == 3 and len(image.shape) == 3:
        image = np.transpose(image, (1, 2, 0))

    h, w, c = image.shape

    if saliency_map.max() > 1.0 or saliency_map.min() < 0:
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

    num_pixels = h * w
    num_mask = int(alpha * num_pixels)

    flat_saliency = saliency_map.flatten()
    top_indices = np.argsort(flat_saliency)[::-1][:num_mask]

    mask = np.ones(num_pixels, dtype=bool)
    mask[top_indices] = False
    mask = mask.reshape(h, w)

    masked_image = image.copy()

    if masking_method == 'black':
        masked_image[~mask] = 0
    elif masking_method == 'blur':
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(image, sigma=5, axes=(0, 1))
        masked_image[~mask] = blurred[~mask]
    elif masking_method == 'uniform':
        noise = np.random.uniform(0, 1, size=image.shape)
        masked_image[~mask] = noise[~mask]

    return masked_image


def compute_ccs(model, image, saliency_map, class_a, class_b,
                alphas=None, masking_method='black', return_curve=False):
    """
    Contrastive Contrastivity Score (CCS).

    """
    if alphas is None:
        alphas = np.linspace(0.1, 0.9, 9)  # 10% to 90% in 10% steps

    scores = []
    for alpha in alphas:
        masked_image = mask_top_pixels(image, saliency_map, alpha, masking_method)
        probs = model(masked_image)[0]
        score = probs[class_a] - probs[class_b]
        scores.append(score)

    ccs = auc(alphas, scores)

    if return_curve:
        return alphas, scores
    return ccs


def compute_cgc(model, image, saliency_map, class_a, group_a,
                alphas=None, masking_method='black', return_curve=False):
    """
    Class Group Contrastivity (CGC).

    """
    if alphas is None:
        alphas = np.linspace(0.1, 0.9, 9)

    orig_probs = model.predict_proba([image])[0]

    scores = []
    for alpha in alphas:
        masked_image = mask_top_pixels(image, saliency_map, alpha, masking_method)
        probs = model(masked_image)[0]
        group_term = np.mean([probs[k] - orig_probs[k] for k in group_a])
        class_term = probs[class_a] - orig_probs[class_a]
        score = 0.5 * (group_term + class_term)
        scores.append(score)

    cgc = auc(alphas, scores)

    if return_curve:
        return alphas, scores
    return cgc


def compute_pgs(model, image, saliency_map, group_a,
                alphas=None, masking_method='black', return_curve=False):
    """
    Pointwise Group Score (PGS).

    """
    if alphas is None:
        alphas = np.linspace(0.1, 0.9, 9)
    orig_probs = model(image)[0]

    scores = []
    for alpha in alphas:
        masked_image = mask_top_pixels(image, saliency_map, alpha, masking_method)
        probs = model(masked_image)[0]
        score = np.mean([orig_probs[k] - probs[k] for k in group_a])
        scores.append(score)

    pgs = auc(alphas, scores)

    if return_curve:
        return alphas, scores
    return pgs


def compute_cgs(model, image, saliency_map, group_a, group_b,
                alphas=None, masking_method='black', return_curve=False):
    """
    Contrastive Group Score (CGS).
    """
    if alphas is None:
        alphas = np.linspace(0.1, 0.9, 9)

    orig_probs = model(image)[0]

    scores = []
    for alpha in alphas:
        masked_image = mask_top_pixels(image, saliency_map, alpha, masking_method)
        probs = model(masked_image)[0]
        group_a_term = np.mean([probs[k] - orig_probs[k] for k in group_a])
        group_b_term = np.mean([probs[j] - orig_probs[j] for j in group_b])
        score = 0.5 * (group_a_term + group_b_term)
        scores.append(score)

    cgs = auc(alphas, scores)

    if return_curve:
        return alphas, scores
    return cgs



