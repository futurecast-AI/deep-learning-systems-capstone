# Deep Learning Systems Capstone

**Author:** AI Foundations (Edward Hickman)  
**Task:** supervised image classification with a PyTorch CNN on Fashion-MNIST.

Two identical 52,138-parameter CNNs train for five epochs on the same 50,000
images, with the same initial weights and batch orders. **Only Adam's learning
rate changes: 0.001 versus 0.0003.** Validation uses 10,000 images from the
official training set; final testing uses the untouched official 10,000 images.
This is a controlled educational experiment, not a production classifier.

## Required submission files

- `deep_learning.ipynb`: executed notebook, architecture, training logs, both
  evaluations, four inline figures and a final five-sentence summary.
- `module_summary.pdf`: cited Deep Learning Systems Analysis Report.
- `requirements.txt`: actual `pip freeze` from the executed environment.
- `data/`: all four original compressed dataset files and MIT license notice;
  `data_provenance.md` also documents public access and checksums.

The editable report is included as Word and Markdown. `outputs/` contains
both saved state dictionaries, all 20,000 test-prediction rows, probabilities,
class metrics, split assignments, training history, source fingerprints,
control hashes, figures and a self-contained HTML notebook export.

## Run locally

Tested with **Python 3.11.18, PyTorch 2.15.0+cpu** on Linux, using two CPU
threads. No GPU, account, credential or dataset download is needed when the
included data files are retained. Dependency installation needs internet.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m jupyter notebook deep_learning.ipynb
```

On Windows activate with `.venv\Scripts\activate` instead. The pinned CPU
PyTorch build comes from the official CPU wheel index, not the default PyPI
index; installing it first makes the subsequent frozen install resolvable.
If the exact build lacks a wheel for your platform, use a compatible Linux
Python 3.11 environment rather than silently changing versions.

Restart the kernel, then run all cells from this directory. This retrains both
models and overwrites numerical outputs and state dictionaries. The notebook
contains the complete experiment, not a call to a hidden training script.
If a data file is missing, its notebook loader retrieves the original from a
pinned author-repository revision and verifies the published MD5 checksum.
Seeds and deterministic algorithms target this environment; cross-platform
bitwise identity is not guaranteed. See `VERIFICATION.md` for actual checks.

```bash
python -m pytest tests -q
```

## Report and construction

The report includes every required rubric section, actual results and failure
examples, scholarly/official references, and a jargon-free explanation with
concrete counts. It reuses the previously supplied APA student-paper template;
instructor and due date were not supplied and are not invented.

Optional authoring tools (not required to run the submitted notebook):

```bash
python scripts/build_notebook.py
python -m nbconvert --to notebook --execute --inplace deep_learning.ipynb --ExecutePreprocessor.timeout=1200
python scripts/build_notebook.py --finalize
python -m nbconvert --to html deep_learning.ipynb --output-dir outputs --output deep_learning.html
python scripts/build_report.py /path/to/original-apa-template.docx
```

The first command replaces the notebook with unexecuted cells; only use it
when rebuilding intentionally. Finalization changes only the last Markdown
summary using actual saved results. Report rebuilding needs the original
template and system LibreOffice; editing the completed DOCX does not.
`requirements.txt` was generated with `python -m pip freeze > requirements.txt`
after installation, not hand-written. PDF conversion uses Liberation Sans in
place of retained Arial Word styles.

## Review before submission

Read `EXPERIMENT_PLAN.md`, `SUBMISSION_CHECKLIST.md` and the notebook/report.
The baseline's better result applies to this seed and fixed budget, not every
dataset or longer training. No demographic fairness or production readiness
is established. Replit AI assisted with implementation, analysis and writing;
the learner must understand and revise the work and confirm course permission.
Nothing is automatically submitted to the classroom or merged into `main`.
