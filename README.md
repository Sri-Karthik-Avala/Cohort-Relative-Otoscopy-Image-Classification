# Cohort-Relative Otoscopy Image Classification

| | |
| --- | --- |
| Final rank | #27 |
| Domain | Computer Vision |
| Difficulty | Hard |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Solutions submitted | 4 |
| Last submission | 2026-06-23 |

## Problem statement

### Cohort-Relative Image Classification on Otoscopy Images

### Task

This is a supervised **binary image-classification** task on otoscopy (eardrum) images. The
 images are organised into small cohorts of 5–7. Each cohort mostly shares one ear condition
 (its "home" condition), but some cohorts also contain a hidden number (0, 1, or 2) of images
 whose condition differs from that majority. For each test image, predict a binary label: 1 if
 its condition differs from its cohort's majority, else 0.

Two properties make it more than a standard classifier:

- **The label is relative to each cohort.** The same condition can be the majority (label 0)
    in one cohort and the minority (label 1) in another, so there is no fixed global
    normal-vs-abnormal boundary; each image must be classified against its own cohort-mates.
- **The minority count is unknown and may be zero.** A model has to decide how many images in
    a cohort carry label 1, not just rank the most-different image — always labelling the single
    most-different image is penalised on cohorts that actually have zero or two.

---

### Data

The `public/` folder contains the images and three CSV files:

```
public/
├── images/                 # all otoscopy images, one .jpg per image_id (train + test)
├── train.csv               # labelled cohorts: image_id, group_id, is_intruder
├── test.csv                # cohorts to classify: image_id, group_id (is_intruder hidden)
└── sample_submission.csv   # a valid example submission you can overwrite
```

- `**images/**` — about **800** de-identified otoscopy images, one per row, named `<image_id>.jpg`.
    Each is an 8-bit **RGB JPEG, 96×96 pixels**, strongly obfuscated (random crop, flip, colour
    jitter, noise, down-scale and re-encode) so it cannot be traced to a source. The folder holds
    **every** image — both train and test.
- The images are organised into small **cohorts** of **5–7** images each. There are about **133**
    **cohorts** in all, split into roughly **67 labelled (train)** and **66 hidden (test)** cohorts;
    train and test cohorts are **disjoint**. Each cohort hides **0, 1, or 2** intruders, so about
    **15%** of the test images carry label 1.

`**train.csv**` — one row per training image (about **400** rows across the ~67 train cohorts).
 It has **three** columns:

- `**image_id**` — data type **string**. Opaque image id of the form `fif-<n>-<rand>`; matches the
    file `images/<image_id>.jpg`. One row per image.
- `**group_id**` — data type **string**. Opaque cohort id of the form `coh-<n>-<rand>`; every row
    that shares a `group_id` belongs to the same cohort.
- `**is_intruder**` — data type **integer, 0 or 1** (this is the training label). `1` if the
    image's condition differs from its cohort's majority, else `0`. No condition names are given —
    only these binary labels.

`**test.csv**` — one row per test image (about **400** rows across the ~66 test cohorts).
 It has **two** columns (the `is_intruder` label is withheld):

- `**image_id**` — data type **string**. Opaque image id; matches `images/<image_id>.jpg`.
- `**group_id**` — data type **string**. Opaque cohort id; tells you which test images share a
    cohort. The `is_intruder` label is **not** included — predicting it for every row is the task.

Each image belongs to **exactly one** cohort, and `group_id` is given for both train and test, so
 you can classify each image **relative to its own cohort-mates**.

---

### Submission format

A CSV with a header and exactly one row per test `image_id`:

```
image_id,is_intruder
```

The submission has exactly **two** columns:

- `**image_id**` — data type **string**. The image identifier, copied verbatim from `test.csv`.
- `**is_intruder**` — data type **integer, 0 or 1**. The binary class label: `1` if the image's
    condition differs from its cohort majority, else `0`.
- Provide a label for every `image_id` in `test.csv`. Duplicate, unknown, or missing ids are
    rejected. You do not submit `group_id`; the grader scores the labels globally.
- Labels are coerced to {0, 1} (a numeric value ≥ 0.5, or `true`/`yes`, counts as 1), so a
    probability is thresholded at 0.5; the score depends on the 0/1 decision.

Example:

```
image_id,is_intruder
fif-552-t14kc,0
fif-633-6owo5,1
fif-602-f4wsa,0
```

Both `image_id` and `group_id` are random opaque tokens (`fif-<n>-<rand>` and `coh-<n>-<rand>`) and
 the images are obfuscated, so they cannot be matched to any external source. Recovering the
 conditions by lookup, source-matching, or cached labels is prohibited (`WHAT_NOT_TO_USE.md`); labels
 must come from a model trained only on the provided images.

---

### Scoring

The grader is **strict and cohort-aware**: it is not enough to beat chance on average — you must
 reconstruct whole cohorts cleanly. Your labels are scored on **three facets** that are then passed
 through a steep transcendental response (so the bar near the top is high):

```
m = clip( MCC(your labels, true), 0, 1 )            # chance-corrected balance (gate)
j = |flagged ∩ true| / |flagged ∪ true|             # intruder-set overlap (Jaccard)
c = mean over cohorts of exp( -1.1 * #errors in the cohort )   # per-cohort cleanliness

base  = ( m * j * c ) ** (1/3)                       # every facet must hold up (geometric mean)
score = clip( R(base), 0, 1 ) ** 1.4                 # R = mean of exp / tan / cos / log curves (convex)
```

MCC is chance-corrected, so a constant or random prediction scores **0** (and because the facets
 are multiplied, no balance means no score). The Jaccard and the **exponential per-cohort cleanliness**
 term make every stray flag expensive: one wrong label inside a cohort cuts that cohort's credit by
 about two-thirds. The convex response `R` plus the `** 1.4` exponent compress the middle, so only
 near-perfect cohort reconstruction approaches 1.

A constant or global rule cannot score, because the same condition is the majority in some cohorts
 and the minority in others. The strictness means even a model that is right most of the time scores
 modestly — reaching the top demands clean, per-cohort reconstruction.

### What earns a high score

- A feature space in which "matches this cohort" is separable — self-supervised, metric, or
    contrastive features — plus a cohort-relative comparison (distance to the cohort's core),
    learned from the labelled train cohorts.
- A calibrated decision of how many images in each cohort carry label 1 (including zero), e.g.
    a learned threshold or count model on the within-cohort distribution, not a fixed
    "label the top-1".
- One label per test image, every image assigned, robust to the obfuscation.

### What you may and may not use

This is a context-relative binary image-classification task: learn from the labelled cohorts
 to predict, for each image, whether its condition matches its cohort's majority, then apply
 that to the unlabelled cohorts. The rules keep it about modelling, not reverse-engineering or
 label lookup.

### Prohibited — source lookup and label recovery (automatic disqualification)

The images are **strongly obfuscated** (random crop, flip, photometric jitter, noise,
 down-scaling, lossy re-encoding) and carry **fresh random ids** (image and cohort),
 specifically so they cannot be traced back to any original source. You must **not** attempt
 to undo or circumvent this. In particular, the following are forbidden:

- **Matching / re-identifying** the images against ANY external dataset or web source — by
    pixel/byte comparison, perceptual or cryptographic hashing, nearest-neighbour search,
    reverse-image-search, EXIF/metadata, or learned-feature similarity — to recover the
    underlying conditions and then mark as intruders those that differ from a cohort's
    majority condition.
- **Using condition labels obtained from an external source**: any public dataset, atlas,
    search engine, API, or model that was trained on, or returns, the original labels for
    these images. Your intruder flags must be **inferred by a model trained only on the**
    **provided images** (the labelled `train.csv` cohorts + the unlabelled images).
- **Cached / hard-coded / memorised labels or answers** of any kind, including a stored map
    from image to condition used to derive intruder status.
- **Parsing the `image_id` or `group_id**` or any filename/order pattern to infer condition,
    cohort composition, or intruder status (ids are random and meaningless).
- **Grader gaming**: exploiting the MCC computation, the flag-coercion, the strictness
    exponent, validation, or numeric edge cases instead of modelling.
- Manual labelling/auditing of the test cohorts by a human or an external annotation
    service/API.

Submissions found to rely on any of the above are out of scope and may be rejected.

### Allowed

- Any ML approach trained on the provided images: CNNs, vision transformers, transfer
    learning / fine-tuning from generic (non-otoscopy-label) pretrained weights, autoencoders,
    feature extractors + a learned head, etc.
- **Self-supervised / metric / contrastive representation learning** (SimCLR / DINO / MoCo /
    SupCon, triplet, etc.) to build a feature space in which "belongs to this cohort vs not"
    is separable — the intended core.
- **Cohort-relative reasoning**: distance/deviation of each image from its cohort's core
    (centroid / medoid / density), robust statistics within a cohort, set / attention models
    that take the whole cohort as input, learned within-cohort classifiers trained on the
    labelled train cohorts.
- **Count / threshold modelling**: estimating how many intruders a cohort holds (including
    zero) from the within-cohort deviation distribution; a learned global or per-cohort
    threshold; calibration on the train cohorts.
- Standard augmentation, cross-validation, ensembling; `numpy`, `pandas`, `scikit-learn`,
    `PyTorch`/`TensorFlow`, `Pillow`, `scikit-image`, `opencv` and similar general ML/vision
    libraries (generic pretrained backbones are fine; a model that outputs these images'
    original condition labels is not).

### The spirit of the task

The goal is a learned sense of which images fit their cohort and which do not, and how many
 do not. The score must come from a model trained on the provided images, not from recovering
 the conditions by lookup or matching. Off-the-shelf features get part-way (about 0.33); a
 strong cohort-relative model closes the gap. Label lookup is out of scope and, in any case,
 cannot work in general because the same condition is sometimes the intruder and sometimes the
 majority.
