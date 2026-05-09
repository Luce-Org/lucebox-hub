# Matrix v2 at 64k — all fixes in. 2026-05-09T23:48:10+02:00

=== V1_none starting at 23:48:10 ===
V1_none rc=0
=== V2_mtp starting at 23:50:30 ===
V2_mtp rc=0
=== V3_dflash_dm8 starting at 23:52:54 ===
V3_dflash_dm8 rc=0

## Per-cell stats

### V1_none
```
[cache] narrow asymmetric: forced Q8_0 on 2 captured full-attn layer(s) (remaining 8 full-attn keep TQ3)
[cache] kv types: SWA=tq3_0, full=tq3_0
[prefill] 49904 tokens in 85278.6 ms (585.2 tok/s) [chunked+pflash, chunk_size=1024]  (last sampled token: 100)
[stats] generated=256  decode_ms=37108.4  tok/s=6.90  first_tok_ms=145.88
[stats] prefill=49904 tokens  context_used=50160/65536
[mem]  VRAM used=21.25 GB  total=24.00 GB
```

### V2_mtp
```
[cache] narrow asymmetric: forced Q8_0 on 2 captured full-attn layer(s) (remaining 8 full-attn keep TQ3)
[cache] kv types: SWA=tq3_0, full=tq3_0
[prefill] 49904 tokens in 85189.9 ms (585.8 tok/s) [chunked+pflash, chunk_size=1024]  (last sampled token: 100)
[mtp] steps=256 accepted=5 accept_rate=0.02
[stats] generated=256  decode_ms=40432.7  tok/s=6.33  first_tok_ms=164.94
[stats] prefill=49904 tokens  context_used=50160/65536
[mem]  VRAM used=21.70 GB  total=24.00 GB
```

### V3_dflash_dm8
```
[cache] narrow asymmetric: forced Q8_0 on 2 captured full-attn layer(s) (remaining 8 full-attn keep TQ3)
[cache] kv types: SWA=tq3_0, full=tq3_0
[draft] KV cache allocated: 2096 slots
[prefill] 49904 tokens in 85184.4 ms (585.8 tok/s) [chunked+pflash, chunk_size=1024]  (last sampled token: 100)
[draft] KV prefill done: 2096 positions materialized (skipped 47808 early tokens, cap=2096)
[stats] generated=256  decode_ms=27753.8  tok/s=9.22  first_tok_ms=257.06
[stats] prefill=49904 tokens  context_used=50160/65536
[spec] draft_steps=112 total_accepted=256 avg_accept=2.29
[mem]  VRAM used=23.59 GB  total=24.00 GB
```

## Decoded text comparison (first 80 generated tokens)

### V1_none
first_80_decoded: 'swe relentless<unused0>os<bos><unused94><pad><unk><unused6>ock<bos><blockquote><unused0>8<unused6>ublic<unused63>thought<unused94>thought\n<unused95>### Summary of Themes and Characters\n\nThis text consists of several fragmented scenes (likely from a play or a series of dramatic sketches) focusing on the political instability of Rome and the personal conflicts of its leaders.\n\n#### **Major Themes**\n\n*   **Pride vs. Humility:** The'

### V2_mtp
first_80_decoded: 'swe absorber<unused3>os<unused2><unused94><unused94>thought\n<unused95>### Summary of Themes and Characters\n\nThe provided text consists of several fragmented scenes (likely from a composite or modified version of Shakespearean-style plays, including elements of *Coriolanus* and *Richard III*). The narrative focuses on the intersection of military glory, political instability, and the volatility of public favor.\n\n#### Major Themes\n\n'

### V3_dflash_dm8
first_80_decoded: 'swe Bras<mask>os<unused2><unused94><unused94>thought\n<unused95>### Summary of Themes and Characters\n\nThe provided text is a fragmented collection of scenes (likely from a composite or modified version of Shakespearean-style plays, blending elements of *Coriolanus* and *Richard III*). It depicts a world of political instability, violent ambition, and the volatile relationship between the ruling elite and the common people.'

DONE
