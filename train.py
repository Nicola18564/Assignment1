"""Train, evaluate, and process bill, receipt, and invoice documents."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
import zipfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import LinearSVC

# Project paths, dataset labels, and training settings.
PROJECT_DIR = Path(__file__).resolve().parent
ARCHIVE_PATH = PROJECT_DIR / "archive (5).zip"
MODEL_PATH = PROJECT_DIR / "document_classifier.joblib"
REPORT_PATH = PROJECT_DIR / "training_report.json"
RECEIPT_TRAIN_FILE = "train-00000-of-00001.parquet"
RECEIPT_TEST_FILE = "test-00000-of-00001.parquet"
INVOICE_FILES = tuple(f"train-{index:05d}-of-00005 (1).parquet" for index in range(5))
PARQUET_FILES = (RECEIPT_TRAIN_FILE, RECEIPT_TEST_FILE, *INVOICE_FILES)
INVOICE_LABEL_ID = 6
CURRENCY_CLASS_FOLDERS = {
    "$1 bill": "1 Dollar",
    "$2 bill": "2 Doolar",
    "$5 bill": "5 Dollar",
    "$10 bill": "10 Dollar",
    "$50 bill": "50 Dollar",
    "$100 bill": "100 Dollar",
}
LABELS = (*CURRENCY_CLASS_FOLDERS.keys(), "receipt", "invoice")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
RANDOM_SEED = 42

HOG_WINDOW = (64, 96)  # width, height
HOG_CELL = (8, 8)
HOG_BINS = 9
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

ProgressCallback = Callable[[str], None]


# Dataset extraction.
def find_7zip() -> Path:
    """Find 7-Zip, which is needed for this archive's Deflate64 entries."""
    executable = shutil.which("7z.exe") or shutil.which("7z")
    if executable:
        return Path(executable)

    common_path = Path(r"C:\Program Files\7-Zip\7z.exe")
    if common_path.is_file():
        return common_path

    raise FileNotFoundError(
        "7-Zip is required to read the Parquet files in this archive. "
        "Install 7-Zip and run the training again."
    )


def extract_parquet_files(destination: Path) -> dict[str, Path]:
    """Extract the receipt and invoice Parquet files from the dataset ZIP."""
    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {ARCHIVE_PATH}")

    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        str(find_7zip()),
        "x",
        str(ARCHIVE_PATH),
        *PARQUET_FILES,
        f"-o{destination}",
        "-y",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Could not extract the training data: {details[-1200:]}")

    files = {name: destination / name for name in PARQUET_FILES}
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Parquet files in ZIP: " + ", ".join(missing))
    return files


# OCR and PDF reading.
def _configure_tesseract() -> None:
    import pytesseract

    executable = shutil.which("tesseract.exe") or shutil.which("tesseract")
    if executable is None:
        candidate = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if candidate.is_file():
            executable = str(candidate)
    if executable is None:
        raise FileNotFoundError(
            "Tesseract OCR was not found. Install Tesseract OCR to extract text from images."
        )
    pytesseract.pytesseract.tesseract_cmd = str(executable)


def extract_image_text(image_bytes: bytes) -> str:
    """Extract text from an uploaded image with Tesseract OCR."""
    import pytesseract

    _configure_tesseract()
    with Image.open(BytesIO(image_bytes)) as image:
        return pytesseract.image_to_string(image, config="--oem 3 --psm 6").strip()


def pdf_pages(pdf_bytes: bytes) -> list[dict[str, str | bytes]]:
    """Render each PDF page and extract its native text or OCR text."""
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "PDF support needs PyMuPDF. Install it with `python -m pip install pymupdf`."
        ) from exc

    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages: list[dict[str, str | bytes]] = []
    try:
        for page_number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            image_buffer = BytesIO()
            image.save(image_buffer, format="PNG")
            image_bytes = image_buffer.getvalue()

            text = page.get_text("text").strip()
            source = "PDF text"
            if len("".join(text.split())) < 30:
                text = extract_image_text(image_bytes)
                source = "OCR"

            pages.append(
                {
                    "page": page_number,
                    "image": image_bytes,
                    "text": text,
                    "text_source": source,
                }
            )
    finally:
        document.close()
    return pages


@contextmanager
def _workspace_temp_directory():
    """Create a workspace directory that the 7-Zip child process can write to."""
    directory = PROJECT_DIR / f".document_training_{uuid.uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _image_features(image_bytes: bytes) -> np.ndarray | None:
    """Resize a document and calculate a compact HOG descriptor."""
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            gray = image.convert("L")
            width, height = gray.size
            scale = min(HOG_WINDOW[0] / width, HOG_WINDOW[1] / height)
            resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            resized = gray.resize(resized_size, Image.Resampling.LANCZOS)
            canvas = Image.new("L", HOG_WINDOW, color=255)
            canvas.paste(
                resized,
                (
                    (HOG_WINDOW[0] - resized_size[0]) // 2,
                    (HOG_WINDOW[1] - resized_size[1]) // 2,
                ),
            )
            pixels = np.asarray(canvas, dtype=np.float32)
    except (OSError, ValueError, ZeroDivisionError):
        return None

    gradient_y, gradient_x = np.gradient(pixels)
    magnitude = np.hypot(gradient_x, gradient_y)
    angle = (np.arctan2(gradient_y, gradient_x) * (180.0 / np.pi)) % 180.0
    orientation_bin = np.minimum((angle * (HOG_BINS / 180.0)).astype(np.int32), HOG_BINS - 1)

    cell_y = np.arange(HOG_WINDOW[1])[:, None] // HOG_CELL[1]
    cell_x = np.arange(HOG_WINDOW[0])[None, :] // HOG_CELL[0]
    histogram_index = (
        (cell_y * (HOG_WINDOW[0] // HOG_CELL[0]) + cell_x) * HOG_BINS
        + orientation_bin
    )
    cells = np.bincount(
        histogram_index.ravel(),
        weights=magnitude.ravel(),
        minlength=(HOG_WINDOW[1] // HOG_CELL[1])
        * (HOG_WINDOW[0] // HOG_CELL[0])
        * HOG_BINS,
    ).reshape(
        HOG_WINDOW[1] // HOG_CELL[1],
        HOG_WINDOW[0] // HOG_CELL[0],
        HOG_BINS,
    )

    blocks = np.lib.stride_tricks.sliding_window_view(cells, (2, 2), axis=(0, 1))
    blocks = blocks.transpose(0, 1, 3, 4, 2).reshape(-1, 4 * HOG_BINS)
    blocks /= np.maximum(np.linalg.norm(blocks, axis=1, keepdims=True), 1e-6)
    np.clip(blocks, 0.0, 0.2, out=blocks)
    blocks /= np.maximum(np.linalg.norm(blocks, axis=1, keepdims=True), 1e-6)
    return blocks.reshape(-1).astype(np.float32)


def _read_bill_images(
    progress: ProgressCallback,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], dict[str, int]]:
    train_features = {label: [] for label in CURRENCY_CLASS_FOLDERS}
    test_features = {label: [] for label in CURRENCY_CLASS_FOLDERS}
    skipped = {label: 0 for label in CURRENCY_CLASS_FOLDERS}
    rng = np.random.default_rng(RANDOM_SEED)

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        members = archive.infolist()
        for label, folder in CURRENCY_CLASS_FOLDERS.items():
            prefix = f"USA currency/{folder}/"
            images = sorted(
                (
                    item for item in members
                    if item.filename.startswith(prefix)
                    and Path(item.filename).suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=lambda item: item.filename,
            )
            if len(images) < 2:
                raise ValueError(f"Not enough bill images found for {label}.")

            order = rng.permutation(len(images))
            test_count = max(1, round(len(images) * 0.2))
            test_indices = set(order[:test_count].tolist())
            for index, member in enumerate(images):
                feature = _image_features(archive.read(member))
                destination = test_features if index in test_indices else train_features
                if feature is None:
                    skipped[label] += 1
                else:
                    destination[label].append(feature)

        # Keep the archive's six explicitly labeled examples as additional holdout data.
        for member in members:
            parts = member.filename.split("/")
            if (
                len(parts) == 3
                and parts[:2] == ["USA currency", "Test Set"]
                and Path(member.filename).suffix.lower() in IMAGE_EXTENSIONS
            ):
                amount = Path(member.filename).stem.split("_", 1)[0]
                label = f"${amount} bill"
                if label in test_features:
                    feature = _image_features(archive.read(member))
                    if feature is None:
                        skipped[label] += 1
                    else:
                        test_features[label].append(feature)

    progress("Read the six bill denominations")
    return train_features, test_features, skipped


def _read_receipt_features(path: Path) -> tuple[list[np.ndarray], int]:
    features: list[np.ndarray] = []
    skipped = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=16, columns=["image"]):
        for image in batch.column("image").to_pylist():
            image_bytes = image.get("bytes") if image else None
            feature = _image_features(image_bytes) if image_bytes else None
            if feature is None:
                skipped += 1
            else:
                features.append(feature)
    return features, skipped


def _read_invoice_features(
    paths: list[Path],
    progress: ProgressCallback,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    parquet_files = [pq.ParquetFile(path) for path in paths]
    total_rows = sum(parquet.metadata.num_rows for parquet in parquet_files)
    test_size = max(1, round(total_rows * 0.2))
    rng = np.random.default_rng(RANDOM_SEED)
    test_indices = set(rng.choice(total_rows, size=test_size, replace=False).tolist())

    train_features: list[np.ndarray] = []
    test_features: list[np.ndarray] = []
    skipped = 0
    row_index = 0

    for file_number, parquet in enumerate(parquet_files, start=1):
        for batch in parquet.iter_batches(batch_size=16, columns=["image", "label"]):
            images = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            for image, label in zip(images, labels):
                if label != INVOICE_LABEL_ID:
                    raise ValueError(
                        f"Expected invoice label {INVOICE_LABEL_ID}, but found {label}. "
                        "The archive's invoice label mapping needs to be checked."
                    )
                image_bytes = image.get("bytes") if image else None
                feature = _image_features(image_bytes) if image_bytes else None
                if feature is None:
                    skipped += 1
                elif row_index in test_indices:
                    test_features.append(feature)
                else:
                    train_features.append(feature)
                row_index += 1

        progress(f"Read invoice shard {file_number}/{len(parquet_files)}")

    if row_index != total_rows:
        raise RuntimeError(f"Read {row_index} invoice rows, expected {total_rows}.")
    return train_features, test_features, skipped


def _make_dataset(
    samples_by_class: dict[str, list[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for class_name in LABELS:
        class_features = samples_by_class[class_name]
        features.extend(class_features)
        labels.extend([LABEL_TO_ID[class_name]] * len(class_features))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def train_model(progress_callback: ProgressCallback | None = None) -> dict:
    """Train one eight-class model and save its model and evaluation report."""
    progress = progress_callback or (lambda _message: None)
    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {ARCHIVE_PATH}")

    train_by_class = {label: [] for label in LABELS}
    test_by_class = {label: [] for label in LABELS}
    skipped_images = {label: 0 for label in LABELS}

    progress("Reading bill images from the ZIP...")
    bill_train, bill_test, bill_skipped = _read_bill_images(progress)
    train_by_class.update(bill_train)
    test_by_class.update(bill_test)
    skipped_images.update(bill_skipped)

    with _workspace_temp_directory() as temp_dir:
        progress("Extracting receipt and invoice Parquet files...")
        files = extract_parquet_files(temp_dir)

        progress("Reading receipt training images...")
        train_by_class["receipt"], skipped_train_receipts = _read_receipt_features(
            files[RECEIPT_TRAIN_FILE]
        )
        progress("Reading held-out receipt images...")
        test_by_class["receipt"], skipped_test_receipts = _read_receipt_features(
            files[RECEIPT_TEST_FILE]
        )
        skipped_images["receipt"] = skipped_train_receipts + skipped_test_receipts

        progress("Reading invoice images...")
        invoice_train, invoice_test, skipped_images["invoice"] = _read_invoice_features(
            [files[name] for name in INVOICE_FILES], progress
        )
        train_by_class["invoice"] = invoice_train
        test_by_class["invoice"] = invoice_test

    train_support = {label: len(train_by_class[label]) for label in LABELS}
    test_support = {label: len(test_by_class[label]) for label in LABELS}
    if any(count == 0 for count in train_support.values()) or any(
        count == 0 for count in test_support.values()
    ):
        raise ValueError(f"Every class needs train and test images. Got {train_support=} {test_support=}")

    x_train, y_train = _make_dataset(train_by_class)
    x_test, y_test = _make_dataset(test_by_class)
    progress(f"Fitting one Linear SVM on {len(y_train):,} images across 8 classes...")

    model = LinearSVC(
        class_weight="balanced",
        dual="auto",
        max_iter=10_000,
        random_state=RANDOM_SEED,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    class_ids = list(range(len(LABELS)))

    report = {
        "model": "Linear SVM with HOG image features",
        "classes": list(LABELS),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, labels=class_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, labels=class_ids, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=class_ids,
            target_names=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "classification_report_text": classification_report(
            y_test,
            predictions,
            labels=class_ids,
            target_names=list(LABELS),
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=class_ids).tolist(),
        "train_support": train_support,
        "test_support": test_support,
        "skipped_images": skipped_images,
        "invoice_label_id": INVOICE_LABEL_ID,
        "hog_image_size": [HOG_WINDOW[0], HOG_WINDOW[1]],
        "random_seed": RANDOM_SEED,
    }

    joblib.dump(
        {"model": model, "classes": list(LABELS), "hog_image_size": list(HOG_WINDOW)},
        MODEL_PATH,
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    progress("Training complete. Model and metrics have been saved.")
    return report


def load_report() -> dict | None:
    if not REPORT_PATH.is_file():
        return None
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def load_classifier() -> dict:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Model file not found. Run `python train.py` first.")
    return joblib.load(MODEL_PATH)


def predict_image_bytes(image_bytes: bytes, classifier: dict | None = None) -> str:
    """Predict one uploaded image using the same HOG features as training."""
    features = _image_features(image_bytes)
    if features is None:
        raise ValueError("The uploaded page could not be decoded as an image.")

    classifier = classifier or load_classifier()
    class_id = int(classifier["model"].predict(features.reshape(1, -1))[0])
    return classifier["classes"][class_id]


def print_report(report: dict) -> None:
    print(f"Model: {report['model']}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Balanced accuracy: {report['balanced_accuracy']:.4f}")
    print(f"Macro F1: {report['macro_f1']:.4f}")
    print("\nClassification report:")
    print(report["classification_report_text"])
    print("Confusion matrix (rows/columns follow the class order below):")
    print(", ".join(report["classes"]))
    print(np.asarray(report["confusion_matrix"]))
    print(f"\nSaved model: {MODEL_PATH.name}")
    print(f"Saved report: {REPORT_PATH.name}")


if __name__ == "__main__":
    print_report(train_model(print))
