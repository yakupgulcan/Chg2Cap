
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import cg
from tqdm import tqdm

ALPHA = 1.0

DATASET_ROOT = Path(r"E:\Bitirme\Chg2Cap\datasets\LEVIR_CC")
OUTPUT_ROOT = DATASET_ROOT.parent / f"LEVIR_CC_structural_alpha_{ALPHA}"

GRAPH_SIZE = (256, 256)
SPLITS = ["train", "val", "test"]


def create_shared_laplacian(A, B):
    h, w, c = A.shape
    n = h * w

    X1 = A.reshape(n, c)
    X2 = B.reshape(n, c)

    # Ortak graf özellikleri
    Z = (X1 + X2) / 2.0
    Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-6)

    # 8-komşuluk
    idx = np.arange(n).reshape(h, w)

    pairs = np.concatenate([
        np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1),
        np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1),
        np.stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel()], axis=1),
        np.stack([idx[1:, :-1].ravel(), idx[:-1, 1:].ravel()], axis=1)
    ])

    diff = Z[pairs[:, 0]] - Z[pairs[:, 1]]
    delta2 = np.sum(diff ** 2, axis=1)

    sigma2 = max(float(np.median(delta2)), 1e-6)
    weights = np.exp(-delta2 / (2.0 * sigma2))

    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    vals = np.concatenate([weights, weights])

    W = coo_matrix(
        (vals, (rows, cols)), shape=(n, n)
    ).tocsr()

    d = np.asarray(W.sum(axis=1)).ravel()
    Dinv = diags(1.0 / np.sqrt(np.maximum(d, 1e-8)))

    L = eye(n, format="csr") - Dinv @ W @ Dinv

    return L.tocsr()


def structural_pair(image_a, image_b):
    A = np.asarray(Image.open(image_a).convert("RGB"), dtype=np.float64) / 255.0
    B = np.asarray(Image.open(image_b).convert("RGB"), dtype=np.float64) / 255.0

    # Orijinal çözünürlükte çalış
    if A.shape[:2] != GRAPH_SIZE or B.shape[:2] != GRAPH_SIZE:
        raise ValueError(f"Beklenmeyen görüntü boyutu: {image_a.name}")

    h, w, c = A.shape
    n = h * w

    X1 = A.reshape(n, c)
    X2 = B.reshape(n, c)

    L = create_shared_laplacian(A, B)
    M = (eye(n, format="csr") + ALPHA * L).tocsr()

    # CG iteratif çözüm; yoğun matris terslemesi yok
    precond = diags(1.0 / M.diagonal())

    def solve(X):
        S = np.empty_like(X)

        for channel in range(c):
            result, info = cg(
                M,
                X[:, channel],
                x0=X[:, channel],
                M=precond,
                rtol=1e-5,
                atol=0.0,
                maxiter=1000
            )

            if info != 0:
                raise RuntimeError(f"CG yakınsamadı: info={info}")

            S[:, channel] = result

        return S.reshape(h, w, c)

    S1 = solve(X1)
    S2 = solve(X2)

    return S1, S2


def save_image(image, path):
    # Görüntü başına rescale yapmıyoruz!
    image = np.clip(image, 0.0, 1.0)
    image = np.round(image * 255).astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def copy_metadata():
    # images dışındaki annotation/caption dosyalarını kopyala
    for path in DATASET_ROOT.rglob("*"):
        if not path.is_file():
            continue

        relative = path.relative_to(DATASET_ROOT)

        if relative.parts[0] == "images":
            continue

        destination = OUTPUT_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def process_dataset():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    copy_metadata()

    for split in SPLITS:
        folder_a = DATASET_ROOT / "images" / split / "A"
        folder_b = DATASET_ROOT / "images" / split / "B"

        if not folder_a.is_dir() or not folder_b.is_dir():
            raise FileNotFoundError(f"Eksik split klasörü: {split}")

        files_a = sorted(folder_a.glob("*.png"))
        files_b = sorted(folder_b.glob("*.png"))

        names_a = {p.name for p in files_a}
        names_b = {p.name for p in files_b}

        if names_a != names_b:
            raise ValueError(f"{split}: A ve B görüntü isimleri eşleşmiyor")

        print(f"\nProcessing {split}: {len(files_a)} pairs")

        for image_a in tqdm(files_a, desc=split):
            image_b = folder_b / image_a.name

            out_a = OUTPUT_ROOT / "images" / split / "A" / image_a.name
            out_b = OUTPUT_ROOT / "images" / split / "B" / image_a.name

            # İşlem yarıda kesilirse tamamlanan çiftleri atla
            if out_a.exists() and out_b.exists():
                continue

            S1, S2 = structural_pair(image_a, image_b)

            save_image(S1, out_a)
            save_image(S2, out_b)

    print("\nCompleted!")
    print(f"Dataset saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    process_dataset()