# Independent hierarchical confirmation plan

This protocol is the prospective confirmation step for the learned hierarchy
reported in [`hierarchical-aggregation.md`](hierarchical-aggregation.md). It
must be frozen before either detector is run on any candidate image.

The confirmation is intentionally evaluation-only. It may load policy
`c1214e33d698daa7cda082cd2c20ef0e4553d8d63b46ac77494e9685a50a1187`,
but it must never refit coefficients, change feature definitions, select new
thresholds, or use confirmation labels to alter the classifier.

## Research question

Does the frozen hierarchical policy retain useful three-class performance on
sources, generators, and editing methods that were not used in calibration or
in the retrospective Boolean comparison?

## Prespecified sample design

The primary confirmation corpus contains 384 images, balanced with 128 images
per class.

| Class | Required diversity | Maximum contribution |
| --- | --- | ---: |
| Real | at least 2 source datasets | 64 images per dataset |
| Synthetic | at least 2 datasets and 4 generation methods | 64 per dataset and 32 per method |
| Edited | at least 2 datasets and 2 editing methods | 64 per dataset and 64 per method |

All class labels and provenance fields must come from source metadata or a
documented generation/editing procedure. Detector output must not be inspected
while selecting samples.

The exact source datasets, revisions, partitions, generation methods, editing
methods, and image identities are frozen in the candidate catalog and then in
the versioned manifest. A source is eligible only if its license permits this
research use and its revision or retrieval date can be recorded.

## Candidate source allocation

The following allocation is the acquisition target. It is not a frozen sample
list: each archive, license, metadata file, and origin pair must first be
verified locally. If one source fails that review, it must be replaced before
detector execution and the reason must be recorded.

| Class | Source | Count | Prespecified strata |
| --- | --- | ---: | --- |
| Real | Open Images V6 validation | 64 | deterministic hash-ranked sample |
| Real | RAISE-1k distributed with Synthbuster | 64 | deterministic hash-ranked sample |
| Synthetic | GenImage validation | 64 | 32 Wukong, 32 VQDM |
| Synthetic | AI Detector Arena v0.1 | 64 | 32 GPT Image 1.5, 32 Gemini 3 Pro |
| Edited | MagicBrush public dev | 64 | first-turn DALL-E 2 edited targets |
| Edited | AbstractEdit explicit outputs | 64 | Flux-Kontext; Open Images V7 origins |

Synthbuster was initially considered for the second synthetic source, but its
official Zenodo file endpoint returned `404 Not Found` during acquisition on
2026-10-08. It was replaced before manifest freezing and before any detector
execution. The independently distributed RAISE-1k archive remained available,
matched its published MD5, and is retained only as a real-image source.

The edited allocation was also finalized before manifest freezing and detector
execution. MagicBrush uses the public `dev` split because its test split is
withheld. AbstractEdit revision
`bc04f61586e5d5004a273a092131d01fd332e3e3` supplies released paired
Flux-Kontext outputs and embedded Open Images V7 originals with auditable
per-image provenance; it replaces the preliminary AURORA-Bench target.

Primary source documentation:

- [Open Images V6 downloads](https://storage.googleapis.com/openimages/web/download.html)
- [RAISE pairing and official checksum](https://github.com/grip-unina/ClipBased-SyntheticImageDetection/tree/main/data)
- [AI Detector Arena v0.1 repository](https://github.com/AI-Detect-Arena/benchmark-dataset)
- [GenImage repository](https://github.com/GenImage-Dataset/GenImage)
- [MagicBrush repository](https://github.com/OSU-NLP-Group/MagicBrush)
- [AbstractEdit dataset](https://huggingface.co/datasets/DucktorV/AbstractEdit)

Selection within every stratum is deterministic. Its exact seed, ordering
rule, source-file hashes, and revision are recorded in the source metadata,
and the first eligible identities are retained. No detector score,
preview-based quality judgment, or replacement after inference is allowed.

## Frozen confirmation corpus

Acquisition and provenance validation completed on 2026-10-09, before any
external detector execution. The catalog was frozen with commit
`61dc49274b0fe2dadbbb60125ea0ba97ddcf5849` of the validation tooling.

- candidate catalog SHA-256:
  `569c68cde4b6b980f910bd74b3e33957c953277416395f0590fcb5aabb05af3e`;
- external manifest SHA-256:
  `d901cd8e4ee5211399b50be579dda23c3814601da6763e2e46547f753c6a77a4`.

The deterministic per-source selections are identified by these SHA-256
digests:

| Source | Selection SHA-256 |
| --- | --- |
| Open Images V6 | `0d7a109b8bfb58d43f3ba6c82e98937b1fc46e29375d4242b337521c66d5168d` |
| RAISE-1k | `a12d198e420cd47b0e657c555462d05c21aba3e5427d99576082bd1097b7151a` |
| AI Detector Arena v0.1 | `5f561a7f465d16c5907fbdcec1de25b347062911d927c7db0ac3d639f803268b` |
| GenImage | `f1bf900bb7dd94f7dc7cd3812b724d5725c46be88bc2ee4f7cbded85073cccbc` |
| MagicBrush | `9e5ac5094733c5a449085073ef3729f1faf36bb72fbbd59cdaff7ff058e5f3aa` |
| AbstractEdit | `352b6efd59f91cf3730347c4bb7ba0b64981205b0e67345e039497e140ea6977` |

## Independence and exclusions

The manifest builder requires both earlier manifests as exclusions:

- `three-class-validation-manifest.csv`;
- `hierarchical-calibration-manifest.csv`.

It rejects overlap by sample identifier, `(source_dataset, source_id)`, or
image SHA-256. Every edited row additionally records the original dataset,
identifier, local path, and image SHA-256. Both the edited image and its
recorded original are verified from disk. Canonical COCO identities are
checked across aliases, so a derivative cannot evade exclusion merely by being
repackaged in another editing dataset. It also requires unique confirmation
sample identifiers, image hashes, origin hashes, and canonical source
identities. An edited origin may not match any real, synthetic, or edited input
in the confirmation corpus. Related originals and derivatives share a
`group_id` so uncertainty estimates do not treat them as independent
observations.

CocoGlide and DiffusionDB are not eligible external sources because they were
already used in the earlier experiments.

## Frozen inputs and features

The following components remain unchanged:

- DistilDIRE revision, checkpoints, preprocessing, and score definition;
- DinoLizer revision, checkpoint, preprocessing, p99 score, and marked area;
- the three aggregation features;
- both serialized logistic regressions;
- level-1 threshold `0.73` and level-2 threshold `0.50`;
- the definitions of abstention, coverage, and end-to-end metrics.

The raw-evidence pass records detector scores only. The policy is applied later
by an evaluation command that verifies its exact SHA-256. The external tooling
contains no fit command.

## Primary metrics and confirmation gates

Abstentions remain in the denominator and count as incorrect for end-to-end
accuracy, recall, balanced accuracy, and macro F1. The primary metrics are:

- end-to-end balanced accuracy;
- end-to-end macro F1;
- coverage;
- per-class precision, recall, and F1;
- level-1 and level-2 metrics;
- conclusive confusion matrix.

The hierarchy passes the prespecified confirmation gates only if:

- aggregation errors are zero;
- coverage is at least 80%;
- balanced accuracy is at least 70%;
- macro F1 is at least 65%;
- recall is at least 50% for every class.

Two thousand source-group bootstrap resamples use seed `20261008` to report
95% intervals for accuracy, balanced accuracy, macro F1, and coverage. The
intervals are descriptive; the fixed gates above determine the primary pass or
fail result.

## Required artifacts

The experiment must preserve:

- the candidate catalog and its source documentation;
- the frozen 384-row manifest and SHA-256;
- detector revisions and checkpoint hashes;
- raw evidence and execution metadata;
- the frozen policy and its verified SHA-256;
- per-image predictions, metrics, confusion table and figure;
- a checksum list for all derived artifacts.

## Execution order

1. choose and document eligible external sources;
2. acquire the images without running either detector;
3. complete the candidate catalog from
   [`hierarchical-external-candidate-template.csv`](results/hierarchical-external-candidate-template.csv);
4. generate, review, and commit the frozen manifest;
5. run a three-image smoke test in a clean GPU runtime;
6. run the complete 384-image raw-evidence pass;
7. verify the raw-result and metadata hashes;
8. apply the frozen policy without fitting;
9. report all outcomes, including failures and abstentions.

No confirmation result may be used to revise this policy. Any later revision
requires a new policy version and another untouched confirmation corpus.
