# Dataset provenance and access

Fashion-MNIST is a public collection of real Zalando fashion-product images,
not synthetic or AI-generated imagery. It contains 60,000 official training
and 10,000 official test images, each 28 × 28 grayscale pixels with one of ten
labels. There are 6,000 / 1,000 images per class in the official splits.
It is distinct from the Titanic, Student Performance and Bike Sharing datasets
used in earlier capstones.

Source: https://github.com/zalandoresearch/fashion-mnist  
Paper: https://doi.org/10.48550/arXiv.1708.07747  
Authors: Han Xiao, Kashif Rasul and Roland Vollgraf (2017).

## Included original files

All four `.gz` files are preserved unchanged in `data/`, downloaded from the
author repository at revision `b2617bb6d3ffa2e429640350f613e3291e10b141`.
The files use IDX, with big-endian headers, not CSV or JPEG.

| File | Published MD5 |
| --- | --- |
| `train-images-idx3-ubyte.gz` | `8d4fb7e6c68d591d4c3dfef9ec88bf0d` |
| `train-labels-idx1-ubyte.gz` | `25c81989df183df01b3e8a0aad5dffbe` |
| `t10k-images-idx3-ubyte.gz` | `bef4ecab320f06d8554ea6380940ec79` |
| `t10k-labels-idx1-ubyte.gz` | `bb300cfdad3c16e7a12a480ee83cd310` |

Direct access pattern:
`https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/b2617bb6d3ffa2e429640350f613e3291e10b141/data/fashion/<filename>`

`outputs/dataset_manifest.json` records actual byte sizes, verified MD5 and
SHA-256 values, and each full download URL. The notebook can download missing
files automatically and checks the MD5 before decoding; it does not silently
switch to a different dataset. The repository's MIT notice is retained in
`data/LICENSE.fashion-mnist`.

## Preparation and quality

The official training set is split by a seeded within-class permutation into
50,000 training and 10,000 validation images; the official test set is retained
unchanged. `outputs/split_assignments.csv` records every original index, label
and assignment. Labels in order are T-shirt/top, Trouser, Pullover, Dress,
Coat, Sandal, Shirt, Sneaker, Bag and Ankle boot.

No image is discarded, augmented or relabeled. Pixel values are converted from
uint8 to float32 and divided by 255, with a single-channel dimension added.
The audit found no exact pixel duplicates within the official sets or across
the experiment partitions. This does not establish separate garment identities,
absence of near duplicates, demographic coverage or applicability to new
retailers. Header/length/range/label checks and the actual audit are saved.
