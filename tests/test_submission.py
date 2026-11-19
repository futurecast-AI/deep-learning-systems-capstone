import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from docx import Document
import nbformat
import numpy as np
import pandas as pd
import pymupdf as fitz
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SUMMARY = json.loads((OUT / "experiment_summary.json").read_text())
NOTEBOOK = nbformat.read(ROOT / "deep_learning.ipynb", as_version=4)
PREDICTIONS = pd.read_csv(OUT / "test_predictions.csv")
torch.set_num_threads(2)


def raw_labels(kind):
    return np.frombuffer(gzip.decompress(
        (ROOT / "data" / f"{kind}-labels-idx1-ubyte.gz").read_bytes()),
        dtype=np.uint8, offset=8)


def model_definition():
    source = next(c.source for c in NOTEBOOK.cells
                  if c.cell_type == "code" and "class ClothingCNN" in c.source)
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    scope = {"nn": nn}
    exec(compile(tree, "<notebook architecture>", "exec"), scope)
    return scope["ClothingCNN"]


def test_original_files_match_source_fingerprints_and_dimensions():
    manifest = json.loads((OUT / "dataset_manifest.json").read_text())
    assert len(manifest) == 4
    for name, entry in manifest.items():
        raw = (ROOT / "data" / name).read_bytes()
        assert len(raw) == entry["bytes"]
        assert hashlib.md5(raw).hexdigest() == entry["md5"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    for kind, n in [("train", 60000), ("t10k", 10000)]:
        image_bytes = gzip.decompress((ROOT / "data" / f"{kind}-images-idx3-ubyte.gz").read_bytes())
        assert len(image_bytes) == 16 + n * 28 * 28
        assert np.array_equal(np.bincount(raw_labels(kind)), np.full(10, n // 10))
    assert "Zalando" in (ROOT / "data/LICENSE.fashion-mnist").read_text()


def test_stratified_splits_are_disjoint_and_cover_all_images():
    split = pd.read_csv(OUT / "split_assignments.csv")
    assert len(split) == 70000
    assert not split.duplicated(["source", "source_index"]).any()
    development = split[split.source == "official_training"]
    assert np.array_equal(development.source_index, np.arange(60000))
    assert np.array_equal(development.class_id, raw_labels("train"))
    assert (split[split.source == "official_test"].split == "test").all()
    assert set(development.split) == {"train", "validation"}
    counts = split.groupby(["split", "class_id"]).size()
    assert (counts["train"] == 5000).all()
    assert (counts["validation"] == 1000).all()
    assert (counts["test"] == 1000).all()
    for name, key in [("train", "train_indices_sha256"),
                      ("validation", "validation_indices_sha256")]:
        indices = development.loc[development.split == name, "source_index"].to_numpy(dtype="<i8")
        assert hashlib.sha256(indices.tobytes()).hexdigest() == SUMMARY["data_audit"][key]


def test_pixel_overlap_audit_independently():
    development = np.frombuffer(gzip.decompress(
        (ROOT / "data/train-images-idx3-ubyte.gz").read_bytes()),
        dtype=np.uint8, offset=16).reshape(60000, 784)
    test = np.frombuffer(gzip.decompress(
        (ROOT / "data/t10k-images-idx3-ubyte.gz").read_bytes()),
        dtype=np.uint8, offset=16).reshape(10000, 784)
    dev_hashes = {hashlib.sha256(row.tobytes()).digest() for row in development}
    test_hashes = {hashlib.sha256(row.tobytes()).digest() for row in test}
    assert len(dev_hashes) == 60000 and len(test_hashes) == 10000
    assert not dev_hashes.intersection(test_hashes)
    assert SUMMARY["data_audit"]["test_images_matching_official_training_pixels"] == 0


def test_exactly_one_configuration_variable_changes():
    configs = json.loads((OUT / "experiment_configs.json").read_text())
    a, b = configs["baseline"], configs["experimental"]
    assert [key for key in a if a[key] != b[key]] == ["learning_rate"]
    assert a["learning_rate"] == 0.001 and b["learning_rate"] == 0.0003
    assert a["epochs"] == 5 and a["batch_size"] == 128
    assert a["checkpoint"] == "final_epoch"
    assert SUMMARY["changed_keys"] == ["learning_rate"]


def test_initialization_and_batch_orders_identical_and_reproducible():
    audit = json.loads((OUT / "control_audit.json").read_text())
    a, b = audit["baseline"], audit["experimental"]
    assert a["initial_state_sha256"] == b["initial_state_sha256"]
    assert a["epoch_order_sha256"] == b["epoch_order_sha256"]
    assert a["final_state_sha256"] != b["final_state_sha256"]
    torch.manual_seed(SUMMARY["seed"])
    model = model_definition()()
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.numpy().tobytes())
    assert digest.hexdigest() == a["initial_state_sha256"]
    for epoch, actual in enumerate(a["epoch_order_sha256"], 1):
        order = torch.randperm(50000, generator=torch.Generator().manual_seed(SUMMARY["seed"] + epoch))
        assert hashlib.sha256(order.numpy().astype("<i8").tobytes()).hexdigest() == actual
    assert sum(p.numel() for p in model.parameters()) == SUMMARY["parameter_count"] == 52138


def test_training_history_has_all_epochs_and_finite_values():
    history = pd.read_csv(OUT / "training_history.csv")
    assert len(history) == 10
    for name, rows in history.groupby("model"):
        assert rows.epoch.tolist() == [1, 2, 3, 4, 5]
        assert np.isfinite(rows.select_dtypes("number").to_numpy()).all()
        assert rows.train_loss.gt(0).all() and rows.validation_loss.gt(0).all()
        assert rows.train_accuracy.between(0, 1).all()
        assert rows.validation_accuracy.between(0, 1).all()
        assert np.isclose(rows.iloc[-1].validation_loss,
                          SUMMARY["models"][name]["validation_loss"])


def test_saved_predictions_cover_official_test_labels_and_valid_scores():
    assert len(PREDICTIONS) == 20000
    for name, rows in PREDICTIONS.groupby("model", sort=False):
        assert np.array_equal(rows.test_index, np.arange(10000))
        assert np.array_equal(rows.true_class, raw_labels("t10k"))
        probability = rows[[f"probability_{i}" for i in range(10)]].to_numpy()
        assert np.isfinite(probability).all() and (probability >= 0).all()
        np.testing.assert_allclose(probability.sum(1), 1, atol=3e-7)
        assert np.array_equal(probability.argmax(1), rows.predicted_class)
        assert np.array_equal(rows.correct, rows.true_class == rows.predicted_class)
        loss = -np.log(probability[np.arange(10000), rows.true_class])
        np.testing.assert_allclose(loss, rows.example_loss, atol=5e-6, rtol=2e-6)
        assert np.isclose(loss.mean(), SUMMARY["models"][name]["test_loss"], rtol=1e-6)


def test_metrics_and_confusion_matrices_independently():
    for name, rows in PREDICTIONS.groupby("model"):
        cm = np.zeros((10, 10), dtype=int)
        np.add.at(cm, (rows.true_class.to_numpy(), rows.predicted_class.to_numpy()), 1)
        saved = pd.read_csv(OUT / f"{name}_confusion_matrix.csv", index_col=0)
        assert np.array_equal(cm, saved.to_numpy())
        tp = np.diag(cm).astype(float)
        f1 = 2 * tp / (cm.sum(0) + cm.sum(1))
        result = SUMMARY["models"][name]
        assert np.isclose(tp.sum() / cm.sum(), result["test_accuracy"])
        assert np.isclose(f1.mean(), result["test_macro_f1"])
        assert result["test_correct"] == tp.sum()
        assert result["test_errors"] == cm.sum() - tp.sum()
        per_class = pd.read_csv(OUT / "per_class_metrics.csv")
        saved_classes = per_class[per_class.model == name]
        np.testing.assert_allclose(saved_classes.f1, f1)
        np.testing.assert_allclose(saved_classes.recall, tp / cm.sum(1))


def test_saved_model_states_reproduce_every_test_prediction():
    raw = gzip.decompress((ROOT / "data/t10k-images-idx3-ubyte.gz").read_bytes())
    x = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=16).copy()
                         .reshape(10000, 1, 28, 28)).float() / 255
    for name in ["baseline", "experimental"]:
        model = model_definition()()
        model.load_state_dict(torch.load(OUT / f"{name}_state.pt", weights_only=True))
        model.eval()
        with torch.inference_mode():
            actual = torch.cat([model(batch).argmax(1) for batch in x.split(512)]).numpy()
        expected = PREDICTIONS[PREDICTIONS.model == name].predicted_class.to_numpy()
        assert np.array_equal(actual, expected)


def test_error_examples_are_first_eight_baseline_failures():
    rows = PREDICTIONS[PREDICTIONS.model == "baseline"]
    indices = rows.loc[~rows.correct, "test_index"].head(8).tolist()
    errors = pd.read_csv(OUT / "error_examples.csv")
    assert errors.test_index.tolist() == indices == SUMMARY["shown_error_indices"]
    assert errors.iloc[0].to_dict() == SUMMARY["first_error"]
    for _, row in errors.iterrows():
        for name in ["baseline", "experimental"]:
            prediction = PREDICTIONS[(PREDICTIONS.model == name) &
                                     (PREDICTIONS.test_index == row.test_index)].iloc[0]
            assert prediction.true_label == row.true_label
            assert prediction.predicted_label == row[f"{name}_label"]


def test_notebook_executed_with_four_figures_and_full_training_code():
    code = [c for c in NOTEBOOK.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type == "error" for c in code for o in c.outputs)
    assert sum("image/png" in o.get("data", {}) for c in code for o in c.outputs) == 4
    source = "\n".join(c.source for c in code)
    for term in ["class ClothingCNN", "loss.backward()", "optimizer.step()",
                 "torch.optim.Adam", "torch.inference_mode()", "learning_rate"]:
        assert term in source


def test_final_notebook_summary_has_five_actual_result_sentences():
    last = NOTEBOOK.cells[-1]
    assert last.cell_type == "markdown" and "## Notebook summary" in last.source
    assert len(last.source.split("\n\n", 1)[1].split(".\n")) == 5
    assert "Pending" not in last.source
    for result in SUMMARY["models"].values():
        assert f"{result['test_accuracy']:.2%}" in last.source
    assert str(SUMMARY["first_error"]["test_index"]) in last.source
    assert "exact pixel matches across partitions" not in last.source


def test_report_required_sections_real_values_and_citations():
    markdown = (ROOT / "module_summary.md").read_text()
    for heading in ["Report Overview", "Dataset and Task Description",
                    "Model Architecture and Design Decisions", "Experimental Comparison",
                    "Results and Interpretation", "Limitations and Risks",
                    "Ethical and Responsible Use", "Future Improvements", "References"]:
        assert f"## {heading}" in markdown
    for result in SUMMARY["models"].values():
        for key, spec in [("test_accuracy", ".2%"), ("test_macro_f1", ".4f"),
                          ("test_loss", ".4f")]:
            assert format(result[key], spec) in markdown
    for citation in ["LeCun et al., 2015", "Xiao et al., 2017", "Mitchell et al.'s (2019)",
                     "PyTorch contributors, n.d.-a", "PyTorch contributors, n.d.-b",
                     "PyTorch contributors, n.d.-c"]:
        assert citation in markdown
    assert f"image {SUMMARY['first_error']['test_index']}" in markdown
    assert "Instructor Name" not in markdown and "Due Date" not in markdown
    assert len(Document(ROOT / "module_summary.docx").inline_shapes) == 4


def test_plain_language_section_in_all_three_report_formats():
    with fitz.open(ROOT / "module_summary.pdf") as pdf:
        pdf_text = "\n".join(p.get_text() for p in pdf)
        assert sum(len(p.get_images()) for p in pdf) >= 4
        assert "10.1038/nature14539" in pdf_text
        assert "References" in pdf_text
    docx_text = "\n".join(p.text for p in Document(ROOT / "module_summary.docx").paragraphs)
    for text in [pdf_text, docx_text, (ROOT / "module_summary.md").read_text()]:
        plain = text.split("Interpretation for a Non-Technical Audience", 1)[1]
        plain = " ".join(plain.split("Limitations and Risks", 1)[0].split())
        for term in ["macro", "cross-entropy", "hyperparameter", "p-value",
                     "logits", "CNN", "PyTorch", "et al.", "(2017)"]:
            assert term not in plain
        assert "8,729" in plain and "1,271" in plain and "266" in plain
        assert "person would need to check" in plain
        assert "sneaker" in plain and "bag" in plain


def test_template_formatting_and_frozen_environment():
    audit = json.loads((OUT / "report_template_audit.json").read_text())
    assert audit["styles_preserved"] and audit["section_preserved"] and audit["headers_preserved"]
    doc = Document(ROOT / "module_summary.docx")
    for margin in [doc.sections[0].top_margin, doc.sections[0].bottom_margin,
                   doc.sections[0].left_margin, doc.sections[0].right_margin]:
        assert abs(float(margin) / 914400 - 1) < 0.001
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    assert frozen == (ROOT / "requirements.txt").read_text()
    assert "torch==2.15.0+cpu" in frozen
    for path in ["deep_learning.ipynb", "module_summary.pdf", "requirements.txt",
                 "data_provenance.md", "README.md"]:
        assert (ROOT / path).is_file()
