# Verification record

The complete notebook ran top to bottom without execution errors on Python
3.11.18 / PyTorch 2.15.0+cpu, with two CPU threads. Both CNN configurations
trained for five epochs. Four figures, all epoch logs and both final
evaluations are embedded in the submitted notebook.

## Local tests

All **15 tests passed**. They verify source fingerprints and dimensions,
balanced/disjoint split assignments, an independent exact-pixel overlap
check, the single changed configuration key, reproducible initialization
and batch-order hashes, complete finite epoch metrics, official test-label
alignment and valid scores, independently recomputed metrics/confusion
matrices, every prediction from both saved model states, deterministic error
selection, notebook execution/summary, report sections/citations/actual
values, jargon-free interpretation in Markdown/Word/PDF, retained template
formatting and an exact match to the installed environment's `pip freeze`.

Results are saved in `outputs/validation.txt`. These are local results, not
a claim that GitHub CI passed.

## Fresh-environment reproduction

A separate Python 3.11.18 virtual environment installed the pinned CPU torch
build from the official PyTorch index, then all frozen requirements using
the documented sequence. Only the final notebook and original data were
copied into a fresh execution directory.

Running that notebook independently reproduced these files **byte for byte**:

- `outputs/experiment_summary.json`
- `outputs/test_predictions.csv` (every prediction and probability)
- `outputs/training_history.csv` (every epoch's training/validation metrics)

Wall-clock durations vary and are intentionally stored separately. This
unchanged rerun was a reproducibility check, not a retuned third configuration.
Later citation-only Markdown edits did not change any experiment code.
No cross-platform reproducibility claim is made.

## Report inspection

All **11 PDF pages** were rendered and visually inspected, including the
four actual figures and the reference page. The plain-language section on
page 5 was inspected separately at higher resolution. The report includes
all nine required rubric sections, a real named/indexed error and an
additional jargon-free interpretation with counts a reader can picture.

The source Word template's style XML, section geometry and header/footer
parts were preserved. Word retains Arial styles; PDF conversion substitutes
metrically compatible Liberation Sans. Instructor and due date were not
supplied and were not invented. The original shared Google document was not
modified. Learner review and course-policy compliance remain necessary before
submission; no classroom upload or merge is performed automatically.
