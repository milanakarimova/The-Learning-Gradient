import numpy as np


def information_content(p, base=2):
    if p <= 0 or p > 1:
        raise ValueError("Probability must be in the interval (0, 1].")
    return -np.log(p) / np.log(base)


def entropy(probs, base=2):
    probs = np.array(probs, dtype=float)
    probs = probs[probs > 0]
    logs = np.log(probs) / np.log(base)
    return -np.sum(probs * logs)


def cross_entropy(p_true, q_model, base=2, eps=1e-15):
    p_true = np.array(p_true, dtype=float)
    q_model = np.array(q_model, dtype=float)

    q_model = np.clip(q_model, eps, 1)

    logs = np.log(q_model) / np.log(base)
    return -np.sum(p_true * logs)


def kl_divergence(p_true, q_model, base=2, eps=1e-15):
    p_true = np.array(p_true, dtype=float)
    q_model = np.array(q_model, dtype=float)

    mask = p_true > 0
    p = p_true[mask]
    q = np.clip(q_model[mask], eps, 1)

    logs = np.log(p / q) / np.log(base)
    return np.sum(p * logs)


def mutual_information(joint, base=2, eps=1e-15):
    joint = np.array(joint, dtype=float)

    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)

    expected = px @ py

    mask = joint > 0
    ratio = joint[mask] / np.clip(expected[mask], eps, None)

    logs = np.log(ratio) / np.log(base)
    return np.sum(joint[mask] * logs)


def information_gain(parent_labels, left_labels, right_labels):
    def label_entropy(labels):
        values, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        return entropy(probs)

    parent_entropy = label_entropy(parent_labels)

    n = len(parent_labels)
    weighted_child_entropy = (
        len(left_labels) / n * label_entropy(left_labels)
        + len(right_labels) / n * label_entropy(right_labels)
    )

    return parent_entropy - weighted_child_entropy


def perplexity_from_bits(cross_entropy_bits):
    return 2 ** cross_entropy_bits


print("Information content")
for p in [1.0, 0.5, 0.25, 0.125]:
    print(f"p={p}, surprise={information_content(p):.3f} bits")

certain = np.array([1.0, 0.0, 0.0, 0.0])
skewed = np.array([0.70, 0.15, 0.10, 0.05])
uniform = np.array([0.25, 0.25, 0.25, 0.25])

print()
print("Entropy comparison")
print("certain:", entropy(certain))
print("skewed:", entropy(skewed))
print("uniform:", entropy(uniform))

P = np.array([0.55, 0.25, 0.15, 0.05])
Q = np.array([0.50, 0.28, 0.15, 0.07])

H_P = entropy(P)
H_PQ = cross_entropy(P, Q)
KL_PQ = kl_divergence(P, Q)

print()
print("Entropy, cross-entropy, KL")
print("H(P):", H_P)
print("H(P,Q):", H_PQ)
print("D_KL(P||Q):", KL_PQ)
print("H(P) + KL:", H_P + KL_PQ)

joint_independent = np.array([
    [0.25, 0.25],
    [0.25, 0.25],
])

joint_dependent = np.array([
    [0.45, 0.05],
    [0.05, 0.45],
])

print()
print("Mutual information")
print("independent:", mutual_information(joint_independent))
print("dependent:", mutual_information(joint_dependent))

parent = np.array([0, 0, 0, 0, 1, 1, 1, 1])
left = np.array([0, 0, 0, 1])
right = np.array([0, 1, 1, 1])

print()
print("Information gain:")
print(information_gain(parent, left, right))

cross_entropy_bits = 2.5

print()
print("Perplexity from cross-entropy bits:")
print(perplexity_from_bits(cross_entropy_bits))

predicted_probs = np.array([0.70, 0.12, 0.08, 0.06, 0.04])

print()
print("Prediction entropy:")
print(entropy(predicted_probs))
