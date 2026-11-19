# Controlled experiment plan

Author: AI Foundations (Edward Hickman)

This plan was recorded before downloading the images or training either model.
It is a small educational experiment, not a preregistered research study.

## Question

For a fixed small PyTorch convolutional neural network (CNN) and five-epoch
training budget, how does reducing Adam's learning rate from 0.001 to 0.0003
change clothing-image classification performance?

## Data and partitions

- Fashion-MNIST: real Zalando product images, 70,000 grayscale 28 × 28 images,
  ten classes; not used in any preceding capstone.
- Preserve the official 60,000 development / 10,000 test split.
- From each class in the official training set, select 1,000 validation images
  with a seeded NumPy generator; the remaining 5,000 form the training set.
  Thus the experiment has 50,000 training, 10,000 validation and 10,000 test images.
- Audit shapes, labels, class counts and exact pixel repetitions. Preserve the
  official benchmark rather than deleting repeated images after seeing outcomes;
  disclose any cross-partition pixel matches as a limitation.
- Convert uint8 pixels to float32 and divide by 255; add a channel dimension.
  No learned preprocessing, augmentation, additional datasets or pretrained weights.

## Fixed architecture and training

Conv2d(1,8,3,padding=1) → ReLU → MaxPool2d(2) →
Conv2d(8,16,3,padding=1) → ReLU → MaxPool2d(2) →
Flatten → Linear(784,64) → ReLU → Linear(64,10).

Use unnormalized logits with CrossEntropyLoss and Adam, betas=(0.9,0.999),
eps=1e-8, weight_decay=0, amsgrad=False, foreach=False. Batch size 128,
five epochs, float32 CPU execution with two PyTorch threads and deterministic
algorithms. No scheduler, dropout, early stopping or checkpoint selection.
Both evaluations use each model's final fifth-epoch weights.

## Only intentional experimental change

- Baseline: learning rate **0.001**.
- Experimental: learning rate **0.0003**.

All other settings, train/validation/test partitions, initial parameter tensors
and epoch-specific shuffled training orders are identical. Initialize once with
seed 20261119, copy that state into both models, and verify initialization and
batch-order hashes. Recreate Adam independently for each run. The smaller rate
tests whether smaller updates help held-out performance or instead learn more
slowly under the same update budget; neither outcome is assumed.

## Evaluation and reporting

- Record sample-weighted training and validation loss and accuracy every epoch.
- Evaluate both final models on the untouched official test set after training
  both configurations. Do not retune after inspecting test results.
- Report accuracy, macro-F1, cross-entropy, per-class precision/recall/F1 and
  confusion matrices; preserve every test prediction and probability.
- Plot representative training images, training/validation curves, both test
  confusion matrices and the first eight baseline errors in official index
  order (or all errors if fewer), showing both models' predictions.
- Discuss actual differences and at least one identifiable failure example.
  Timing is descriptive of this machine/run, not a general speed benchmark.
- One seed cannot establish robustness or a universally best learning rate.
  Small images, one retailer and absent demographic/group metadata limit claims.
- Repeat unchanged execution only to check reproducibility, not select results.
- Explain practical meaning without jargon in a dedicated report subsection.
  Cite technical concepts and responsible-use claims elsewhere in APA 7.
- Disclose AI assistance; learner review and course-policy compliance are required.
