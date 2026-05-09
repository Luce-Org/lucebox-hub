# Scaling matrix — 2026-05-10T00:00:56+02:00
=== D1_dense_q8_q8_64k starting at 00:00:56 ===
D1_dense_q8_q8_64k rc=0
=== M_moe_dflash_q8q8_16384 starting at 00:02:19 ===
M_moe_dflash_q8q8_16384 rc=0
=== M_moe_dflash_q8q8_32768 starting at 00:02:43 ===
M_moe_dflash_q8q8_32768 rc=0
=== M_moe_dflash_q8q8_65536 starting at 00:03:03 ===
M_moe_dflash_q8q8_65536 rc=0
=== M_moe_dflash_q8q8_131072 starting at 00:03:29 ===
M_moe_dflash_q8q8_131072 rc=0
=== M_moe_dflash_q8q8_262144 starting at 00:03:51 ===
M_moe_dflash_q8q8_262144 rc=0

## Per-cell stats

### D1_dense_q8_q8_64k
```
[cache] kv types: SWA=q8_0, full=q8_0
[prefill] 49904 tokens in 35599.5 ms (1401.8 tok/s) [chunked+pflash, chunk_size=1024]  (last sampled token: 236798)
[stats] generated=256  decode_ms=32143.1  tok/s=7.96  first_tok_ms=157.93
[stats] prefill=49904 tokens  context_used=50160/65536
[mem]  VRAM used=22.60 GB  total=24.00 GB
```

### M_moe_dflash_q8q8_131072
```
[cache] kv types: SWA=q8_0, full=q8_0
[draft] KV cache allocated: 2096 slots
[prefill] 49904 tokens in 10209.1 ms (4888.2 tok/s) [chunked+pflash, chunk_size=1024]  (last sampled token: 100)
[draft] KV prefill done: 2096 positions materialized (skipped 47808 early tokens, cap=2096)
[stats] generated=256  decode_ms=8560.5  tok/s=29.90  first_tok_ms=62.33
[stats] prefill=49904 tokens  context_used=50160/131072
[spec] draft_steps=177 total_accepted=256 avg_accept=1.45
[mem]  VRAM used=20.42 GB  total=24.00 GB
```

### M_moe_dflash_q8q8_16384
```
[cache] kv types: SWA=q8_0, full=q8_0
[draft] KV cache allocated: 2096 slots
[prefill] 2612 tokens in 703.8 ms (3711.5 tok/s) [chunked+pflash, chunk_size=1024]  (last sampled token: 7243)
[draft] KV prefill done: 2096 positions materialized (skipped 516 early tokens, cap=2096)
[stats] generated=256  decode_ms=3540.0  tok/s=72.32  first_tok_ms=41.71
[stats] prefill=2612 tokens  context_used=2868/16384
[spec] draft_steps=154 total_accepted=256 avg_accept=1.66
[mem]  VRAM used=19.27 GB  total=24.00 GB
```

### M_moe_dflash_q8q8_262144
```
[cache] kv types: SWA=q8_0, full=q8_0
[draft] KV cache allocated: 2096 slots
[prefill] 49904 tokens in 10197.4 ms (4893.8 tok/s) [chunked+pflash, chunk_size=1024]  (last sampled token: 100)
[draft] KV prefill done: 2096 positions materialized (skipped 47808 early tokens, cap=2096)
[stats] generated=256  decode_ms=8707.2  tok/s=29.40  first_tok_ms=62.09
[stats] prefill=49904 tokens  context_used=50160/262144
[spec] draft_steps=177 total_accepted=256 avg_accept=1.45
[mem]  VRAM used=21.74 GB  total=24.00 GB
```

### M_moe_dflash_q8q8_32768
```
[cache] kv types: SWA=q8_0, full=q8_0
[draft] KV cache allocated: 2096 slots
[prefill] 2612 tokens in 681.4 ms (3833.4 tok/s) [chunked+pflash, chunk_size=1024]  (last sampled token: 7243)
[draft] KV prefill done: 2096 positions materialized (skipped 516 early tokens, cap=2096)
[stats] generated=256  decode_ms=3629.2  tok/s=70.54  first_tok_ms=38.51
[stats] prefill=2612 tokens  context_used=2868/32768
[spec] draft_steps=154 total_accepted=256 avg_accept=1.66
[mem]  VRAM used=19.45 GB  total=24.00 GB
```

### M_moe_dflash_q8q8_65536
```
[cache] kv types: SWA=q8_0, full=q8_0
[draft] KV cache allocated: 2096 slots
[prefill] 49904 tokens in 10229.4 ms (4878.5 tok/s) [chunked+pflash, chunk_size=1024]  (last sampled token: 100)
[draft] KV prefill done: 2096 positions materialized (skipped 47808 early tokens, cap=2096)
[stats] generated=256  decode_ms=8851.8  tok/s=28.92  first_tok_ms=65.18
[stats] prefill=49904 tokens  context_used=50160/65536
[spec] draft_steps=177 total_accepted=256 avg_accept=1.45
[mem]  VRAM used=19.74 GB  total=24.00 GB
```

### orchestrator
```
