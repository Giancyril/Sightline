# ADR-002: 20-Class Open Images Subset Selection

## Status
Accepted

## Context
The full Open Images Challenge dataset covers 500 classes across ~1.74M training images (~535 GiB), which is impractical for local development and free-tier GPU sessions. SSD models pre-trained on Common Objects in Context (COCO) detect 80 everyday categories. 

To demonstrate tangible fine-tuning value as specified in the project goals:
1. The fine-tuned model must improve detection accuracy on classes **not** in COCO (target: ≥ +10 mAP points).
2. The fine-tuned model must retain detection capability on classes **shared** with COCO (regression: ≤ 2 mAP points).

## Decision
We select a balanced 20-class subset divided evenly into 10 COCO-overlapping categories and 10 categories new to COCO:

### 1. COCO-Overlapping Categories (10 classes)
Used to verify retention of pre-trained feature representations:
| ID | Class Name | Open Images MID | COCO Category |
|:---|:---|:---|:---|
| 1 | `Person` | `/m/01g317` | person |
| 2 | `Laptop` | `/m/01c648` | laptop |
| 3 | `Chair` | `/m/01mzpv` | chair |
| 4 | `Book` | `/m/0bt9lr` | book |
| 5 | `Backpack` | `/m/0130jx` | backpack |
| 6 | `Cell phone` | `/m/050k8` | cell phone |
| 7 | `Bottle` | `/m/04dr76x` | bottle |
| 8 | `Clock` | `/m/01x3z` | clock |
| 9 | `Car` | `/m/0k4j` | car |
| 10 | `Dog` | `/m/0bt9lr_dog` | dog |

### 2. New / Non-COCO Categories (10 classes)
Used to measure fine-tuning gain on novel target concepts:
| ID | Class Name | Open Images MID | Domain / Context |
|:---|:---|:---|:---|
| 11 | `Guitar` | `/m/0342h` | Musical instrument |
| 12 | `Microphone` | `/m/0hg7b` | Audio gear |
| 13 | `Helmet` | `/m/0zvk5` | Safety / Wearable |
| 14 | `Headphones` | `/m/01b7fy` | Consumer electronics |
| 15 | `Wheelchair` | `/m/012w5l` | Assistive device |
| 16 | `Whiteboard` | `/m/02688pb` | Office / Education |
| 17 | `Flashlight` | `/m/01b_21` | Tool / Hardware |
| 18 | `Calculator` | `/m/01kggc` | Office / School |
| 19 | `Stapler` | `/m/033cnk` | Office supply |
| 20 | `Wrench` | `/m/02jnhm` | Mechanical tool |

## Licensing & Attribution
- Open Images dataset annotations are licensed under CC BY 4.0.
- Individual images have separate CC BY 2.0 / CC BY-SA 2.0 licenses, tracked via Open Images image IDs and URLs.
- Dataset attribution and licensing records will be maintained in `docs/model_card.md`.

## Group-Of Box Handling
Boxes flagged as `IsGroupOf=1` in Open Images encompass clusters of items (e.g. a crowd of people or a rack of bottles) rather than discrete single object instances. These boxes distort IoU computations and regression losses during single-instance SSD fine-tuning, so they are excluded from fine-tuning targets and flagged during EDA.

