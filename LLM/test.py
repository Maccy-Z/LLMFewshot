import numpy as np


def sample_balanced_batches(X, y, bs, num_batches, seed=0):
    """
    Generator function that yields balanced batches from dataset X with labels y.
    Each batch will have total_batch_size samples, distributed as evenly as possible among classes.

    Parameters:
    - X: Features in the dataset.
    - y: Labels corresponding to X.
    - total_batch_size: Total number of samples required in the batch.
    - rng: A numpy random number generator instance for controlled sampling.

    Yields:
    - X_batch: Features of the sampled batch.
    - y_batch: Labels corresponding to X_batch.
    """

    RNG = np.random.default_rng(seed)

    unique_labels = np.unique(y)
    n_classes = len(unique_labels)

    # Calculate samples per class and determine the "extra" samples
    samples_per_class = bs // n_classes
    extra_samples = bs % n_classes

    batches = []
    for _ in range(num_batches):
        batch_indices = []
        for idx, label in enumerate(unique_labels):
            label_indices = np.where(y == label)[0]

            # Adjust samples for this class if there are extra samples
            current_samples = samples_per_class + 1 if idx < extra_samples else samples_per_class

            if len(label_indices) < current_samples:
                raise ValueError(f"Label {label} has fewer samples than the requested samples for this class.")

            sampled_indices = RNG.choice(label_indices, current_samples, replace=False)
            batch_indices.extend(sampled_indices)

        remainder_indices = np.setdiff1d(np.arange(len(X)), batch_indices)

        X_batch, y_batch = X[batch_indices], y[batch_indices]
        X_remainder, y_remainder = X[remainder_indices], y[remainder_indices]

        batches.append((X_batch, y_batch, X_remainder, y_remainder))
    return batches


X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

X_batch = sample_balanced_batches(X, y, bs=9, num_batches=5, seed=2)

print("Sampled X batch:", X_batch[0])
