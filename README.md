# Bills, Receipts, and Invoices Classifier

A Python project that trains one image classifier for six U.S. banknote denominations, receipts, and invoices. Training, dataset handling, PDF processing, and OCR are kept together in `train.py`; `app.py` provides the Streamlit interface.

## What it does

- Trains a single eight-class Linear Support Vector Machine (Linear SVM) using Histogram of Oriented Gradients (HOG) image features.
- Evaluates the model and saves a classification report, confusion matrix, and summary metrics.
- Classifies uploaded images and PDF pages.
- Displays the uploaded image or rendered PDF pages.
- Extracts image text with Tesseract OCR. For PDFs, it uses embedded PDF text when available and OCR for scanned pages with little embedded text.
- Displays the overall model metrics in Streamlit. When a PDF is uploaded, it also displays validation accuracy for the three document categories: bill, receipt, and invoice.

> **Bill label meaning:** In this project, “bill” means a U.S. currency note. The six bill classes are $1, $2, $5, $10, $50, and $100. Utility bills and other payment statements are not included in the current training classes.

## Project files

| File | Purpose |
| --- | --- |
| `train.py` | Defines the dataset and labels, extracts training files, trains and evaluates the classifier, and handles image OCR and PDF text extraction/rendering. |
| `app.py` | Streamlit interface for uploads, predictions, extracted text, and summary metrics. |
| `archive (5).zip` | Local dataset archive expected by training; intentionally excluded from the GitHub push because of its size. |
| `document_classifier.joblib` | Saved classifier created by training. |
| `training_report.json` | Saved metrics and evaluation details created by training. |

The model and report files may already be present. Running training again replaces them. The dataset ZIP is not included in the GitHub repository; after cloning, obtain the archive separately and place it beside `train.py` before training.

## Requirements

- Python 3.10 or newer.
- 7-Zip, required to extract the archive's Deflate64-compressed Parquet files. The code searches for `7z.exe` on `PATH` and at `C:\Program Files\7-Zip\7z.exe`.
- Tesseract OCR with English language data, required to extract text from images and scanned PDF pages. The code searches for Tesseract on `PATH` and at `C:\Program Files\Tesseract-OCR\tesseract.exe`.
- The Python packages below.

Install the Python packages in PowerShell from the project directory:

```powershell
python -m pip install --upgrade pip
python -m pip install streamlit numpy pillow scikit-learn joblib pyarrow pytesseract pymupdf
```

`pytesseract` is a Python wrapper; the Tesseract OCR application must also be installed separately. PyMuPDF provides PDF page rendering and is imported in Python as `pymupdf`.

## Dataset layout

For training, obtain `archive (5).zip` and keep it in the same directory as `train.py`. The archive is intentionally not committed because it is about 3 GB. The training code expects these items in the archive:

- Bill images under `USA currency/<denomination folder>/` for all six denominations, with a `USA currency/Test Set/` folder for any additional labeled holdout images.
- Receipt Parquet files named `train-00000-of-00001.parquet` and `test-00000-of-00001.parquet`.
- Five invoice Parquet shards named `train-00000-of-00005 (1).parquet` through `train-00004-of-00005 (1).parquet`.

The receipt and invoice Parquet files are extracted from the ZIP into a temporary project folder during training. 7-Zip must be available before training starts. If your archive has a different name or layout, update the paths and filenames near the top of `train.py`.

## Train and evaluate

From the project directory, run:

```powershell
python train.py
```

This command trains the model from the archive, prints accuracy, balanced accuracy, macro F1, the classification report, and the confusion matrix, and saves:

- `document_classifier.joblib` — model used by the app.
- `training_report.json` — metrics and evaluation data used by the app.

Training uses a fixed random seed of 42. Currency images and invoice images are split into training and holdout sets; receipt training and test files are supplied separately in the archive. The saved results can vary if the dataset changes.

## Run the Streamlit app

Start the app with:

```powershell
python -m streamlit run app.py
```

Using `python -m streamlit` works even when the `streamlit` command is not on your Windows `PATH`. Streamlit prints a local address in the terminal and usually opens it in your browser.

The app needs `document_classifier.joblib` to classify uploads. If it is missing, train the model first with `python train.py`. The **Train all classes** button in the app also starts a new training run.

### Upload a document

Use **Upload an image or PDF** to select a `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.tif`, `.tiff`, or `.pdf` file.

- **Image:** The app previews the image, predicts one of the eight trained classes, and displays OCR text.
- **PDF:** The app renders and classifies each page, displays each page and extracted text, and reports an overall prediction based on the most common page prediction. PDFs with embedded text use that text; scanned pages use OCR when embedded text is missing or very short.
- **Validation accuracy:** The PDF upload view shows overall held-out accuracy and category accuracy for bill, receipt, and invoice. Category accuracy is measured on the labeled holdout set; it is a model evaluation result, not a certainty score for the uploaded document. Bill category accuracy treats any prediction among the six bill denominations as a correct bill-category prediction.

## Current saved evaluation

The `training_report.json` currently in this project reports the following results on 5,137 held-out examples:

| Metric | Result |
| --- | ---: |
| Accuracy | 99.42% |
| Balanced accuracy | 97.94% |
| Macro F1 | 0.979 |
| Weighted F1 | 0.994 |

These values describe the saved training run. Retraining with a different dataset or split can change the results. Because invoice examples make up most of the holdout set, accuracy should be considered together with balanced accuracy and macro F1.

## Troubleshooting

### `streamlit` is not recognized

Run Streamlit through Python:

```powershell
python -m streamlit run app.py
```

### The app says the model is missing

Train and save the model first:

```powershell
python train.py
```

### 7-Zip cannot be found or the archive cannot be extracted

Install 7-Zip and make `7z.exe` available on `PATH`, or install it in `C:\Program Files\7-Zip\`. Confirm that `archive (5).zip` is in the project directory and contains the expected Parquet files.

### Tesseract cannot be found

Install the Tesseract OCR application and its English language data. Add the folder containing `tesseract.exe` to `PATH`, or use the default location `C:\Program Files\Tesseract-OCR\tesseract.exe`.

### PDF support is unavailable

Install PyMuPDF in the same Python environment used to start Streamlit:

```powershell
python -m pip install pymupdf
```

### A document is classified incorrectly or no text is extracted

Predictions are based on the image appearance of the eight classes represented in the training data. The model is not trained for other document types, and OCR quality depends on image clarity, orientation, and text quality. Try a clear, upright scan or image. For new document categories, add labeled examples and update the training pipeline before retraining.
