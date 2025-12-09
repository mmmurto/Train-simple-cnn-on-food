import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

from food11_autoencoder import AE, same_seeds


def build_transform(img_size: int = 64):
    """Image transform consistent with autoencoder training."""

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_index(
    model: AE,
    device: torch.device,
    split: str = "training",
    img_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    index_dir: str = "indices",
) -> None:
    """Extract latent features for a split and save them for retrieval.

    Saves three files under `index_dir`:
      - {split}_latents.npy: 2D array of latent vectors
      - {split}_paths.npy:   image file paths
      - {split}_labels.npy:  integer class labels
      - {split}_classes.txt: mapping from label id to class name
    """

    root = os.path.join("food-11", split)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Split directory not found: {root}")

    transform = build_transform(img_size)
    dataset = datasets.ImageFolder(root, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model.eval()
    all_latents = []

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            z, _ = model(x)
            z = z.view(z.size(0), -1)
            all_latents.append(z.cpu().numpy())

    latents = np.concatenate(all_latents, axis=0)
    paths = np.array([p for p, _ in dataset.samples])
    labels = np.array([lbl for _, lbl in dataset.samples], dtype=np.int64)

    assert latents.shape[0] == len(paths) == len(labels)

    os.makedirs(index_dir, exist_ok=True)
    np.save(os.path.join(index_dir, f"{split}_latents.npy"), latents)
    np.save(os.path.join(index_dir, f"{split}_paths.npy"), paths)
    np.save(os.path.join(index_dir, f"{split}_labels.npy"), labels)

    classes_path = os.path.join(index_dir, f"{split}_classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f:
        for idx, name in enumerate(dataset.classes):
            f.write(f"{idx}\t{name}\n")

    print(
        f"Index built for split='{split}': {latents.shape[0]} images, "
        f"feature dim={latents.shape[1]}"
    )
    print(f"Saved index files to '{index_dir}'")


def load_index(index_dir: str, split: str = "training"):
    """Load latent index (features, paths, labels) for retrieval."""

    latents_path = os.path.join(index_dir, f"{split}_latents.npy")
    paths_path = os.path.join(index_dir, f"{split}_paths.npy")
    labels_path = os.path.join(index_dir, f"{split}_labels.npy")

    if not (os.path.exists(latents_path) and os.path.exists(paths_path)):
        raise FileNotFoundError(
            f"Index files not found for split='{split}' under '{index_dir}'. "
            "Run 'build-index' mode first."
        )

    latents = np.load(latents_path)
    paths = np.load(paths_path, allow_pickle=True)
    labels = np.load(labels_path)

    return latents, paths, labels


def load_class_mapping(index_dir: str, split: str = "training"):
    """Load mapping from label id to class name if available."""

    mapping_path = os.path.join(index_dir, f"{split}_classes.txt")
    if not os.path.exists(mapping_path):
        return None

    mapping = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, name = line.split("\t", maxsplit=1)
            mapping[int(idx_str)] = name
    return mapping


def preprocess_image(image_path: str, img_size: int, device: torch.device):
    """Load and preprocess a single image for querying."""

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Query image not found: {image_path}")

    transform = build_transform(img_size)
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)  # (1, C, H, W)
    return x.to(device)


def retrieve_similar(
    model: AE,
    device: torch.device,
    image_path: str,
    index_dir: str = "indices",
    split: str = "training",
    img_size: int = 64,
    top_k: int = 5,
    metric: str = "cosine",
    save_figure: str | None = None,
):
    """Given a query image, retrieve top-k most similar images from an index.

    metric: 'cosine' (default) or 'l2'.
    """

    latents, paths, labels = load_index(index_dir, split)
    class_mapping = load_class_mapping(index_dir, split)

    model.eval()
    with torch.no_grad():
        x = preprocess_image(image_path, img_size, device)
        z, _ = model(x)
        q = z.view(1, -1).cpu().numpy()  # (1, D)

    if metric == "cosine":
        # Normalize to unit length and use cosine similarity
        eps = 1e-8
        feats = latents.astype(np.float32)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True) + eps
        qn = q.astype(np.float32)
        qn /= np.linalg.norm(qn, axis=1, keepdims=True) + eps

        sims = feats @ qn.T  # (N, 1)
        sims = sims.squeeze(1)
        order = np.argsort(-sims)  # descending
        scores = sims
    else:
        # Euclidean distance
        diffs = latents - q
        dists = np.linalg.norm(diffs, axis=1)
        order = np.argsort(dists)  # ascending
        scores = dists

    top_k = min(top_k, len(order))
    top_idx = order[:top_k]

    print(f"Query image: {image_path}")
    print(f"Search split: {split}, index size: {latents.shape[0]} images")
    print(f"Metric: {metric}, top-k: {top_k}")
    print("\nTop results:")

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        path = str(paths[idx])
        label_id = int(labels[idx])
        label_name = class_mapping.get(label_id, str(label_id)) if class_mapping else str(label_id)
        score = float(scores[idx])

        if metric == "cosine":
            score_str = f"similarity={score:.4f}"
        else:
            score_str = f"distance={score:.4f}"

        print(f"#{rank}: {path}  [label={label_id} ({label_name})]  {score_str}")
        results.append((path, label_id, label_name, score))

    # Optional visualization
    if save_figure is not None:
        visualize_query_and_results(
            query_image_path=image_path,
            results=results,
            save_path=save_figure,
            metric=metric,
        )

    return results


def visualize_query_and_results(
    query_image_path: str,
    results: list[tuple[str, int, str, float]],
    save_path: str,
    metric: str = "cosine",
) -> None:
    """Save a matplotlib grid of the query image and its top-k results.

    results: list of (path, label_id, label_name, score)
    """

    image_paths = [query_image_path] + [r[0] for r in results]
    titles = ["Query"]
    for rank, (_, _, label_name, score) in enumerate(results, start=1):
        if metric == "cosine":
            score_str = f"sim={score:.3f}"
        else:
            score_str = f"dist={score:.3f}"
        titles.append(f"#{rank}: {label_name}\n{score_str}")

    n = len(image_paths)
    cols = min(n, 6)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1:
        axes = np.array(axes).reshape(1, -1)
    axes = axes.flatten()

    for ax, path, title in zip(axes, image_paths, titles):
        try:
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
            ax.set_title(title, fontsize=8)
        except Exception:
            ax.text(
                0.5,
                0.5,
                os.path.basename(path),
                ha="center",
                va="center",
                fontsize=8,
            )
        ax.axis("off")

    # Hide any unused axes
    for ax in axes[len(image_paths) :]:
        ax.axis("off")

    out_dir = os.path.dirname(save_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved retrieval visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Image retrieval on Food-11 using the trained autoencoder",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Build index mode
    p_build = subparsers.add_parser(
        "build-index", help="Precompute latent index for a split"
    )
    p_build.add_argument(
        "--split", type=str, default="training", choices=["training", "validation"]
    )
    p_build.add_argument("--img-size", type=int, default=64)
    p_build.add_argument("--batch-size", type=int, default=64)
    p_build.add_argument("--index-dir", type=str, default="indices")
    p_build.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("checkpoints", "ae_last.pth"),
        help="Path to the trained autoencoder checkpoint.",
    )
    p_build.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (e.g. 'cuda' or 'cpu').",
    )

    # Query mode
    p_query = subparsers.add_parser("query", help="Query similar images by example")
    p_query.add_argument("image", type=str, help="Path to the query image")
    p_query.add_argument(
        "--split", type=str, default="training", choices=["training", "validation"]
    )
    p_query.add_argument("--img-size", type=int, default=64)
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("--index-dir", type=str, default="indices")
    p_query.add_argument(
        "--metric", type=str, default="cosine", choices=["cosine", "l2"]
    )
    p_query.add_argument(
        "--save-figure",
        type=str,
        default=None,
        help="If set, save a matplotlib grid of the query and top-k results to this path.",
    )
    p_query.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("checkpoints", "ae_last.pth"),
        help="Path to the trained autoencoder checkpoint.",
    )
    p_query.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (e.g. 'cuda' or 'cpu').",
    )

    args = parser.parse_args()

    same_seeds(0)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load autoencoder
    model = AE().to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded autoencoder from {args.checkpoint}")
    else:
        raise FileNotFoundError(
            f"Checkpoint not found at {args.checkpoint}. Train the model first using food11_autoencoder.py."
        )

    if args.mode == "build-index":
        build_index(
            model=model,
            device=device,
            split=args.split,
            img_size=args.img_size,
            batch_size=args.batch_size,
            index_dir=args.index_dir,
        )
    elif args.mode == "query":
        retrieve_similar(
            model=model,
            device=device,
            image_path=args.image,
            index_dir=args.index_dir,
            split=args.split,
            img_size=args.img_size,
            top_k=args.top_k,
            metric=args.metric,
            save_figure=args.save_figure,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
