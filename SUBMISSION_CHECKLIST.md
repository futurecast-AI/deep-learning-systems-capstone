# Deep learning capstone submission checklist

Author: AI Foundations (Edward Hickman)

## Required files

- `deep_learning.ipynb`: executed top to bottom; representative samples,
  shapes/dtypes, preprocessing, full PyTorch CNN, training code, logs, four
  embedded figures, controlled comparison and five-sentence final summary.
- `module_summary.pdf`: all nine required report headings, actual experimental
  numbers, concrete model errors and APA 7 citations.
- `requirements.txt`: actual environment freeze, including the CPU torch build.
- `data/`: original Fashion-MNIST data files and license; access instructions
  and fingerprints also supplied in `data_provenance.md`.

## Experiment integrity

- Only the learning rate differs: 0.001 versus 0.0003.
- Identical starting weights, batch orders, architecture, partitions, optimizer
  settings, batch size and fixed five-epoch duration.
- Test images do not enter training or validation; neither test results nor
  validation scores select a checkpoint. Both epoch-five models are reported.
- Results include both configurations and real failure cases, not just the
  better outcome. Scores are not evidence of deployment readiness.

## Before uploading

- Review and understand the notebook and report, especially the practical
  explanation and the limitations of a single-seed comparison.
- Verify the course's academic-integrity/AI-use policy; assistance is disclosed.
- Use this capstone's ZIP, not one from an earlier assignment.
- If sharing GitHub before merging, link the completed branch/PR rather than
  the initial README on `main`. A ZIP contains files, not Git history.
- The report has editable DOCX/Markdown companions. Instructor and due date
  were not supplied; the original shared Google document was not modified.
- No classroom submission or merge is performed automatically.
