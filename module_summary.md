# Clothing Image Classification: A Controlled Learning-Rate Experiment

AI Foundations (Edward Hickman)

## Report Overview

This project classifies real Fashion-MNIST clothing images into ten product categories using a small convolutional neural network (CNN) implemented in PyTorch (Xiao et al., 2017). Two configurations differ only in Adam's learning rate, allowing a controlled comparison under a fixed five-epoch budget. The goal is a reproducible deep learning experiment with honest error analysis, not a production-ready retail classifier.

## Dataset and Task Description

Fashion-MNIST contains 70,000 Zalando product images, each 28 × 28 grayscale pixels, with ten categories including shirts, coats, sneakers and bags (Xiao et al., 2017). The task is supervised image classification: the input is an image and the target is its supplied category. The dataset is public, non-synthetic and distinct from the three preceding capstone datasets. The author repository and access instructions are at https://github.com/zalandoresearch/fashion-mnist; the original compressed files and MIT notice are included with this submission.

The official 60,000-image training set supplies 50,000 training and 10,000 validation images through a seeded within-class split; the official 10,000-image test set is reserved for final evaluation. Each class has 5,000 training, 1,000 validation and 1,000 test images. The notebook checks file fingerprints, IDX headers, lengths, labels, shapes and pixel ranges, and displays the first training image in each class in Figure 1. It found 0 repeated pixel rows in the official training set, 0 in the test set, and 0 test images matching any official training image exactly. This is not an audit of garment identities or near duplicates. Pixels become float32 in [0,1] by division by 255, with a single channel added; labels become int64. No estimated preprocessing statistics, imputation, augmentation or additional images are used.

## Model Architecture and Design Decisions

CNNs are appropriate for image representations, rather than imposing a text-token or time-series structure on the inputs (LeCun et al., 2015). The baseline is Conv2d(1,8,3,padding=1), ReLU, MaxPool2d(2), Conv2d(8,16,3,padding=1), ReLU, MaxPool2d(2), Flatten, Linear(784,64), ReLU and Linear(64,10). Pooling produces 14 × 14 then 7 × 7 maps, leaving 16 × 7 × 7 = 784 flattened values. The network has 52,138 trainable parameters. Small channel counts and one hidden dense layer keep this five-epoch CPU experiment inspectable; they are not experimentally proven optimal.

ReLU supplies nonlinear activations, while the final ten outputs are raw class logits. CrossEntropyLoss accepts these unnormalized scores and integer labels; no softmax is applied before the training loss (PyTorch contributors, n.d.-b). Adam uses learning rate 0.001, its documented default, betas=(0.9,0.999), eps=1e-8, weight_decay=0, amsgrad=False and foreach=False (PyTorch contributors, n.d.-a). Both runs use batches of 128, five complete epochs, float32, CPU and two threads. There is no dropout, scheduler, early stopping or checkpoint selection. The default rate is a starting point, not a literature claim that it must win. The observed finite, declining losses and held-out scores show this modest architecture learned useful distinctions here; without an architecture ablation they do not justify its width or depth as better than alternatives.

## Experimental Comparison

The experimental model changes exactly one major aspect: learning rate 0.001 becomes 0.0003. The question is whether smaller updates improve held-out performance or simply learn more slowly within the same budget. No search, extra training or test-driven revision follows the comparison. The experiment plan was committed before image download and training.

Both models load copies of the same initialized parameter tensors (seed 20261119), and each epoch's seeded permutation is identical. A fresh Adam optimizer is constructed for each model. Saved configuration dictionaries differ only in learning_rate; initial-state and epoch-order hashes verify the controls. Architecture, splits, preprocessing, loss, batch size, optimizer settings and five-epoch duration stay fixed. Both final epoch-five models are evaluated, with no validation-based selection. Deterministic CPU algorithms and seeds target repeatability on this stack, not identical behavior on every platform (PyTorch contributors, n.d.-c).

## Results and Interpretation

Figure 2 records all training and validation epochs. Baseline validation loss fell from 0.4904 to 0.3305, with final validation accuracy 88.13%. The smaller-rate model's validation loss fell from 0.6303 to 0.4114, with final accuracy 85.22%. Both improved throughout these five validation measurements; neither shows a rising validation-loss trend in this window. That does not prove the absence of overfitting. Training metrics are online averages while weights change, so a validation score exceeding an epoch's training score is not itself evidence of leakage.

On all 10,000 test images, the baseline achieved accuracy 87.29%, macro-F1 0.8705 and cross-entropy 0.3566. The smaller-rate model achieved 84.63%, 0.8430 and 0.4341, respectively. Thus the baseline classified 266 more images correctly, a 2.66-percentage-point accuracy advantage, with better macro-F1 and lower loss as well. Accuracy is the share of correct top-one labels; macro-F1 averages the class-level precision/recall balance equally (scikit-learn developers, n.d.), and cross-entropy penalizes poor scores for the correct label (PyTorch contributors, n.d.-b). The balanced benchmark makes overall accuracy informative, but it does not replace class-level inspection.

Figure 3 exposes a clear weakness: the baseline correctly recognized 564 of 1,000 shirts (56.4% recall), versus 484 (48.4%) for the smaller rate. In Figure 4, the first baseline error in original test order is image 12, labeled Sneaker, which both models called Bag. This is a wrong product category, not merely an uncertain choice between similar upper garments. The eight examples are the first baseline errors, not handpicked cases; all predictions are available for inspection.

Within this paired run the smaller rate offered no measured accuracy/loss trade-off advantage; its curves are consistent with slower learning under the fixed budget. They do not establish what would happen after longer training. Training plus validation took 18.55 seconds for the baseline and 17.52 seconds for the experimental model. The order was not randomized and timing was measured once, so these durations are not evidence of an inherent speed advantage. No claim of statistical significance or multi-seed reliability is made.

## Interpretation for a Non-Technical Audience

The program tries to sort small clothing pictures into ten groups. With the original training setting, it labeled 8,729 of 10,000 test pictures correctly and got 1,271 wrong. Making each learning update smaller did not help in this trial: it produced 266 fewer correct labels after the same amount of training. For example, both versions put a sneaker picture in the bag category. A shop could use a tool like this to suggest product categories, but a person would need to check and correct mistakes before customers rely on them. These results do not tell us how well it would handle new phone photos, other shops or longer training.

## Limitations and Risks

One seed provides one paired comparison, not a distribution of outcomes across initializations. Five epochs and a single small architecture leave training-budget and capacity questions unanswered; no ablation separates these possibilities. The continuing validation improvement supports testing longer training in future, not claiming that the smaller rate will catch up. This study deliberately avoids tuning on test results.

The images come from one retailer and have already been reduced to tiny grayscale pictures (Xiao et al., 2017). Lighting, backgrounds, color, new garment styles and category definitions in real listings may differ. Zero exact pixel matches does not rule out shared products or near duplicates, since garment identifiers are unavailable in these files. The benchmark cannot establish product-independent generalization. Softmax outputs are not calibrated confidence, and no calibration, out-of-distribution, demographic or seller-group evaluation was performed. The shirt errors show that even balanced class counts do not imply equal class performance.

## Ethical and Responsible Use

A wrong retail label could hide a seller's product from relevant searches, creating avoidable visibility and correction costs. Clothing styles or categories absent from this catalog may be poorly served. The files do not supply demographic or seller-group metadata; neither class balance nor aggregate accuracy demonstrates fair outcomes for those groups. No demographic attributes are inferred from product appearance.

The intended use is education and controlled benchmarking. Any retail trial should obtain permission for new images, include diverse styles and sellers, measure relevant error differences, provide human correction and allow users to contest wrong labels. Documenting intended uses, evaluation conditions and limitations follows Mitchell et al.'s (2019) model-card guidance. The trained classifier should not be repurposed to make judgments about people, identity or suitability for employment.

## Future Improvements

Repeat the same paired comparison across multiple seeds and report variation. Separately test longer fixed training budgets without changing other settings. Seek garment identifiers and an independent, permissioned color-photo evaluation set before considering retail use. Audit difficult shirt/coat/pullover cases, label ambiguity, calibration and seller/style coverage; keep new evaluation data separate from tuning.

The executed notebook, original data, frozen dependencies, saved states and complete experimental outputs accompany this report. Replit AI assisted with planning, implementation, verification and writing. All experiment values are actual outputs. The learner must understand and revise the work and follow course academic-integrity and AI-use policies.

## References

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521, 436–444. https://doi.org/10.1038/nature14539

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. In Proceedings of the Conference on Fairness, Accountability, and Transparency (pp. 220–229). Association for Computing Machinery. https://doi.org/10.1145/3287560.3287596

PyTorch contributors. (n.d.-a). Adam. PyTorch 2.15 documentation. https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html

PyTorch contributors. (n.d.-b). CrossEntropyLoss. PyTorch 2.15 documentation. https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html

PyTorch contributors. (n.d.-c). Reproducibility. PyTorch 2.15 documentation. https://docs.pytorch.org/docs/2.15/notes/randomness.html

scikit-learn developers. (n.d.). f1_score. scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms [Preprint]. arXiv. https://doi.org/10.48550/arXiv.1708.07747

