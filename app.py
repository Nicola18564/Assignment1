"""Streamlit interface for document classification and text extraction."""

from collections import Counter

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from train import (
    MODEL_PATH,
    extract_image_text,
    load_classifier,
    load_report,
    pdf_pages,
    predict_image_bytes,
    print_report,
    train_model,
)


def _show_report(report: dict) -> None:
    columns = st.columns(4)
    columns[0].metric("Accuracy", f"{report['accuracy']:.2%}")
    columns[1].metric("Balanced accuracy", f"{report['balanced_accuracy']:.2%}")
    columns[2].metric("Macro F1", f"{report['macro_f1']:.3f}")
    columns[3].metric("Weighted F1", f"{report['weighted_f1']:.3f}")

    st.caption(f"Model: {report['model']} · Saved as {MODEL_PATH.name}")


@st.cache_resource
def _get_classifier() -> dict:
    return load_classifier()


def _display_class(label: str) -> str:
    if label.endswith(" bill"):
        return f"Bill — {label.removesuffix(' bill')} denomination"
    return label.title()


def _document_category(label: str) -> str:
    if label.lower().endswith(" bill"):
        return "Bill"
    return label.title()


def _category_metrics(report: dict) -> dict[str, float]:
    """Aggregate held-out class results into bill, receipt, and invoice categories."""
    classes = report["classes"]
    matrix = report["confusion_matrix"]
    indices_by_category = {
        category: [
            index for index, label in enumerate(classes)
            if _document_category(label) == category
        ]
        for category in ("Receipt", "Invoice", "Bill")
    }
    metrics = {}
    for category, category_indices in indices_by_category.items():
        if not category_indices:
            continue
        support = sum(sum(matrix[index]) for index in category_indices)
        true_positive = sum(
            matrix[actual][predicted]
            for actual in category_indices
            for predicted in category_indices
        )
        accuracy = true_positive / support if support else 0.0
        metrics[category] = accuracy
    return metrics


def _show_pdf_validation_metrics(report: dict | None) -> None:
    if report is None:
        st.info("Training validation metrics are unavailable. Run `python train.py` to create them.")
        return

    st.subheader("Model validation accuracy")
    st.metric("Overall held-out accuracy", f"{report['accuracy']:.2%}")
    category_metrics = _category_metrics(report)
    rows = [
        {
            "Document type": category,
            "Accuracy": f"{accuracy:.2%}",
        }
        for category, accuracy in category_metrics.items()
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption(
        "Category accuracy is measured on held-out labeled examples; it describes model performance, "
        "not certainty for this individual upload."
    )


def _show_image_upload(uploaded_file, classifier: dict) -> None:
    image_bytes = uploaded_file.getvalue()
    left, right = st.columns([1, 1])
    with left:
        st.image(image_bytes, caption=uploaded_file.name, use_container_width=True)
    with right:
        try:
            prediction = predict_image_bytes(image_bytes, classifier)
            st.success(f"Predicted document: **{_display_class(prediction)}**")
            extracted_text = extract_image_text(image_bytes)
            st.subheader("Extracted text")
            if extracted_text:
                st.text_area("OCR text", extracted_text, height=280, label_visibility="collapsed")
            else:
                st.info("No readable text was detected in this image.")
        except Exception as exc:
            st.error(f"Could not process this image: {exc}")


def _show_pdf_upload(uploaded_file, classifier: dict, report: dict | None) -> None:
    _show_pdf_validation_metrics(report)
    try:
        pages = pdf_pages(uploaded_file.getvalue())
    except Exception as exc:
        st.error(f"Could not open this PDF: {exc}")
        return

    if not pages:
        st.warning("This PDF contains no pages.")
        return

    predictions = []
    results = []
    for page in pages:
        try:
            prediction = predict_image_bytes(page["image"], classifier)
        except Exception as exc:
            st.error(f"Could not classify page {page['page']}: {exc}")
            continue
        predictions.append(prediction)
        results.append((page, prediction))

    if predictions:
        overall_prediction = Counter(predictions).most_common(1)[0][0]
        st.success(
            f"Predicted document: **{_display_class(overall_prediction)}** "
            f"({len(predictions)} page(s) classified)"
        )

    for page, prediction in results:
        with st.expander(f"Page {page['page']} — {_display_class(prediction)}", expanded=True):
            left, right = st.columns([1, 1])
            with left:
                st.image(page["image"], caption=f"Page {page['page']}", use_container_width=True)
            with right:
                st.caption(f"Text source: {page['text_source']}")
                if page["text"]:
                    st.text_area(
                        f"Extracted text from page {page['page']}",
                        page["text"],
                        height=260,
                        label_visibility="collapsed",
                        key=f"pdf_text_{uploaded_file.file_id}_{page['page']}",
                    )
                else:
                    st.info("No readable text was detected on this page.")


def main() -> None:
    # Running with Python prints the report; Streamlit provides the interactive page.
    if get_script_run_ctx(suppress_warning=True) is None:
        report = load_report()
        if report is None:
            report = train_model(print)
        print_report(report)
        return

    st.set_page_config(page_title="Document Classifier Training", page_icon="📄")
    st.title("Bills, Receipts, and Invoices")
    st.write("Upload a document to identify its class and extract its text.")

    uploaded_file = st.file_uploader(
        "Upload an image or PDF",
        type=["png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff", "pdf"],
        help="Image uploads are read with OCR. PDFs are classified page by page and their text is extracted.",
    )

    if uploaded_file is not None:
        if not MODEL_PATH.is_file():
            st.warning("The classifier has not been trained yet. Run `python train.py` first.")
        else:
            try:
                classifier = _get_classifier()
                if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                    report = st.session_state.get("training_report") or load_report()
                    _show_pdf_upload(uploaded_file, classifier, report)
                else:
                    _show_image_upload(uploaded_file, classifier)
            except Exception as exc:
                st.error(f"Could not load the classifier: {exc}")

    if st.button("Train all classes", type="primary"):
        status = st.status("Starting training...", expanded=True)
        try:
            report = train_model(lambda message: status.update(label=message, state="running"))
            _get_classifier.clear()
            st.session_state["training_report"] = report
            status.update(label="Training complete", state="complete")
        except Exception as exc:
            status.update(label="Training failed", state="error")
            st.exception(exc)

    report = st.session_state.get("training_report") or load_report()
    if report is None:
        st.info("Click **Train all classes** to start the single eight-class training run.")
    else:
        _show_report(report)


if __name__ == "__main__":
    main()
