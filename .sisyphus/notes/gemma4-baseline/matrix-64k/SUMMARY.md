# 64k drafter A/B with TQ3 + pFlash (dense 31B) — 2026-05-09T23:05:51+02:00
Prompt: long_50k.txt (~50k tokens), ctx=65536, n_predict=256

=== T1_none ===
T1_none rc=0
=== T2_mtp ===
T2_mtp rc=0
=== T3_dflash ===
T3_dflash rc=143

## Per-cell stats

### T1_none
```
[cache] narrow asymmetric: forced Q8_0 on 2 captured full-attn layer(s) (remaining 8 full-attn keep TQ3)
[cache] kv types: SWA=tq3_0, full=tq3_0
[prefill] 49904 tokens in 87859.2 ms (568.0 tok/s) [chunked+pflash, chunk_size=1024]  (last sampled token: 100)
[stats] generated=256  decode_ms=37952.4  tok/s=6.75  first_tok_ms=150.63
[stats] prefill=49904 tokens  context_used=50160/65536
[mem]  VRAM used=21.40 GB  total=24.00 GB
```

### T2_mtp
```
[cache] narrow asymmetric: forced Q8_0 on 2 captured full-attn layer(s) (remaining 8 full-attn keep TQ3)
[cache] kv types: SWA=tq3_0, full=tq3_0
[prefill] 49904 tokens in 87919.8 ms (567.6 tok/s) [chunked+pflash, chunk_size=1024]  (last sampled token: 100)
[mtp-step 8] accept_rate=0.00
532 81179 108 818 3847 1816 10594 529 [mtp-step 16] accept_rate=0.00
3131 89144 18583 568 19609 699 496 22907 [mtp-step 24] accept_rate=0.00
653 12269 3567 529 36951 508 236772 3061 [mtp-step 32] accept_rate=0.00
10772 236764 2440 4820 529 808 236780 6886 [mtp-step 40] accept_rate=0.03
40707 605 236829 532 808 40421 8488 236829 [mtp-step 48] accept_rate=0.02
769 669 22323 21132 580 506 18074 529 [mtp-step 56] accept_rate=0.02
7820 27877 236764 5255 32202 236764 532 506 [mtp-step 64] accept_rate=0.05
43866 529 1237 4664 236761 108 2595 18787 [mtp-step 72] accept_rate=0.04
137944 108 236829 139 1018 203460 532 19839 [mtp-step 80] accept_rate=0.04
4499 53121 669 6082 12160 84022 2101 506 [mtp-step 88] accept_rate=0.03
16625 1534 33641 532 125860 236761 102301 605 [mtp-step 96] accept_rate=0.03
2481 81341 568 236780 6886 40707 605 236768 [mtp-step 104] accept_rate=0.03
563 496 24240 1933 31451 236764 840 914 [mtp-step 112] accept_rate=0.04
125688 573 506 3364 1331 532 914 45208 [mtp-step 120] accept_rate=0.03
531 623 1674 2737 236775 1091 2080 531 [mtp-step 128] accept_rate=0.03
914 124466 236761 4923 21077 3590 1515 496 [mtp-step 136] accept_rate=0.03
623 45513 236775 528 506 6114 529 914 [mtp-step 144] accept_rate=0.03
22816 532 496 179267 531 506 11838 236761 [mtp-step 152] accept_rate=0.03
107 236829 139 1018 818 6285 26633 529 [mtp-step 160] accept_rate=0.03
506 623 13666 4637 1083 1018 669 1816 [mtp-step 168] accept_rate=0.02
46235 506 214696 4135 529 506 3364 1331 [mtp-step 176] accept_rate=0.02
```

### T3_dflash
```
[cache] narrow asymmetric: forced Q8_0 on 2 captured full-attn layer(s) (remaining 8 full-attn keep TQ3)
[cache] kv types: SWA=tq3_0, full=tq3_0
```

## First 80 generated tokens (decoded)

### T1_none
raw extracted (first 80): [49904, 87859, 2, 568, 0, 100, 0, 3, 13, 134, 2, 895, 5, 308, 13, 206, 376, 45518, 100, 45518, 107, 101, 10354, 25252, 529, 137944, 532, 81179, 108, 818, 3847, 1816, 10594, 529, 3131, 89144, 18583, 568, 19609, 699, 496, 22907, 653, 12269, 3567, 529, 36951, 508, 236772, 3061, 10772, 236764, 2440, 4820, 529, 808, 236780, 6886, 40707, 605, 236829, 532, 808, 40421, 8488, 236829, 769, 669, 22323, 21132, 580, 506, 18074, 529, 7820, 27877, 236764, 5255, 32202, 236764]
decoded (first 80): 'sweולם<bos> (<pad><unused94><pad><unk><unused7>\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n<bos> Y[multimodal]F<unused7><code>�thought<unused94>thought\n<unused95>### Summary of Themes and Characters\n\nThe provided text consists of several fragmented scenes (likely from a composite or modified version of Shakespearean-style plays, including elements of *Coriolanus* and *Richard III*). The narrative focuses on the intersection of military glory, political instability,'

### T2_mtp
raw extracted (first 80): [49904, 87919, 8, 567, 6, 100, 100, 45518, 107, 101, 10354, 25252, 529, 137944, 532, 81179, 108, 818, 3847, 1816, 10594, 529, 3131, 89144, 18583, 568, 19609, 699, 496, 22907, 653, 12269, 3567, 529, 36951, 508, 236772, 3061, 10772, 236764, 2440, 4820, 529, 808, 236780, 6886, 40707, 605, 236829, 532, 808, 40421, 8488, 236829, 769, 669, 22323, 21132, 580, 506, 18074, 529, 7820, 27877, 236764, 5255, 32202, 236764, 532, 506, 43866, 529, 1237, 4664, 236761, 108, 2595, 18787, 137944, 108]
decoded (first 80): 'swe Tahun<unused2>ation<unused0><unused94><unused94>thought\n<unused95>### Summary of Themes and Characters\n\nThe provided text consists of several fragmented scenes (likely from a composite or modified version of Shakespearean-style plays, including elements of *Coriolanus* and *Richard III*). The narrative focuses on the intersection of military glory, political instability, and the volatility of public favor.\n\n#### Major Themes\n\n'

### T3_dflash: no [prefill] marker

DONE

