"""Phase A: single-point prefix cache.

Auto-detects the system-prompt boundary in token id streams via Qwen chat
template markers, hashes prefixes, and maintains an LRU map of hash → daemon
slot id. Daemon owns slot buffers; Python is the index.

Usage:
    bus = DaemonStdoutBus(daemon_proc.stdout)
    bus.start(loop)

    pc = PrefixCache(
        daemon_stdin=daemon_proc.stdin,
        await_reply=bus.await_reply,
        daemon_lock=lock,
        tokenizer=tokenizer,
        cap=4,
    )
    await pc.startup_sync()  # free orphaned slots from a previous daemon run

    # Per request (caller holds daemon_lock):
    hit = pc.lookup(prompt_ids, kv_k_type, fa_window)   # (slot_id, prefix_len) or None
    if hit:
        slot, prefix_len = hit
        # send "RESTORE <slot> <prompt_bin> <n_gen>" instead of bare line
        ...
    else:
        # send bare "<prompt_bin> <n_gen>"
        ...
        # after daemon finishes, snapshot for future cache hits:
        await pc.maybe_snapshot(prompt_ids, kv_k_type, fa_window)
"""
import asyncio
import hashlib
import struct
from collections import OrderedDict


# ---------------------------------------------------------------------------
# DaemonStdoutBus
# ---------------------------------------------------------------------------

class DaemonStdoutBus:
    """Owns the read loop on daemon stdout.

    Lines that start with a registered prefix are routed to the waiting
    coroutine; everything else is printed as a log (with noise filtering).
    """

    # Prefixes that are too spammy to print in normal operation.
    _SUPPRESS_PREFIXES = (
        "[step ", "[timing]", "[dflash]", "[prompt]",
        "[prefill]", "[migrate]", "[dbg ", "  ",
    )

    def __init__(self, stdout):
        self.stdout = stdout
        self._waiters: list[tuple[str, asyncio.Future]] = []
        self._task: asyncio.Task | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._task = loop.create_task(self._run())

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, self.stdout.readline)
            if not line:
                # Daemon exited — wake all waiters with an error.
                for _, fut in self._waiters:
                    if not fut.done():
                        fut.set_exception(EOFError("daemon stdout closed"))
                self._waiters.clear()
                return
            decoded = line.decode("utf-8", errors="replace").rstrip()

            # Try to satisfy a waiter first.
            matched = False
            for i, (prefix, fut) in enumerate(self._waiters):
                if decoded.startswith(prefix) and not fut.done():
                    fut.set_result(decoded)
                    self._waiters.pop(i)
                    matched = True
                    break

            if not matched:
                # Log line — suppress very noisy prefixes.
                if decoded and not any(decoded.startswith(p) for p in self._SUPPRESS_PREFIXES):
                    print(f"  [daemon] {decoded}", flush=True)

    async def await_reply(self, prefix: str, timeout: float = 10.0) -> str:
        """Block until daemon emits a line starting with *prefix*."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        self._waiters.append((prefix, fut))
        return await asyncio.wait_for(fut, timeout=timeout)


# ---------------------------------------------------------------------------
# Qwen chat template helpers
# ---------------------------------------------------------------------------

def _qwen_marker_ids(tokenizer):
    """Resolve <|im_end|>, <|im_start|>, and 'system' token ids."""
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    system_t = tokenizer.encode("system", add_special_tokens=False)
    if len(im_end) != 1 or len(im_start) != 1:
        raise ValueError(
            f"Expected single-token chat markers; got "
            f"im_end={im_end} im_start={im_start}"
        )
    return im_end[0], im_start[0], system_t[0] if len(system_t) == 1 else None


def find_prefix_boundary(ids, im_end_id, im_start_id, system_token_id):
    """Return the index AFTER the FIRST end-of-system-message marker, or -1.

    Qwen's chat template renders to:

        <|im_start|>system\\nCONTENT<|im_end|>\\n<|im_start|>user\\n...

    so a `\\n` token sits BETWEEN ``<|im_end|>`` and the next ``<|im_start|>``.
    We allow up to 2 intervening tokens (covers `\\n` and similar separators).

    The cacheable prefix is the SYSTEM message: from index 0 through and
    including the ``<|im_start|>`` that begins the next role. Subsequent turns
    sharing this system message hash to the same key.

    Returns the index right after that ``<|im_start|>``, so ``ids[:boundary]``
    is the cached state and ``ids[boundary:]`` is the per-request suffix.
    Returns -1 if there is no recognizable system message.
    """
    # Find the first <|im_start|>system sequence.
    sys_idx = -1
    for i in range(len(ids) - 1):
        if ids[i] == im_start_id:
            if system_token_id is None or ids[i + 1] == system_token_id:
                sys_idx = i
                break
    if sys_idx < 0:
        return -1

    # Find the FIRST <|im_end|> after sys_idx, then locate the next <|im_start|>
    # within a small lookahead (handles a single-token newline separator).
    for i in range(sys_idx + 1, len(ids)):
        if ids[i] == im_end_id:
            for j in range(i + 1, min(i + 3, len(ids))):
                if ids[j] == im_start_id:
                    return j + 1   # boundary is one past <|im_start|>
            return -1   # malformed — im_end without subsequent im_start
    return -1


def find_all_boundaries(ids, im_end_id, im_start_id, system_token_id):
    """Return ascending list of candidate cut points for multi-slot caching.

    Each cut point is the index AFTER an ``<|im_start|>`` that begins a new
    role's content. The first cut is the system-prompt boundary (same as
    ``find_prefix_boundary``); subsequent cuts are at every following
    ``<|im_end|>`` + ``<|im_start|>`` pair.

    Returns an empty list if no recognizable system message is found.
    """
    boundaries = []

    # Locate the opening <|im_start|>system token.
    sys_idx = -1
    for i in range(len(ids) - 1):
        if ids[i] == im_start_id:
            if system_token_id is None or ids[i + 1] == system_token_id:
                sys_idx = i
                break
    if sys_idx < 0:
        return boundaries

    # Walk forward from sys_idx: every time we see <|im_end|> followed
    # (within 2 tokens) by <|im_start|>, record the position just after
    # that <|im_start|> as a cache cut-point.
    i = sys_idx + 1
    while i < len(ids):
        if ids[i] == im_end_id:
            found_start = False
            for j in range(i + 1, min(i + 3, len(ids))):
                if ids[j] == im_start_id:
                    boundaries.append(j + 1)
                    i = j + 1
                    found_start = True
                    break
            if not found_start:
                break
        else:
            i += 1
    return boundaries


def hash_prefix(prefix_ids, kv_k_type, fa_window):
    """Stable SHA-1 (truncated 16 B) of (token ids, kv type, fa window)."""
    h = hashlib.sha1()
    h.update(struct.pack("<I", len(prefix_ids)))
    h.update(struct.pack(f"<{len(prefix_ids)}i", *prefix_ids))
    h.update(str(kv_k_type).encode())
    h.update(b"\x00")
    h.update(struct.pack("<I", fa_window or 0))
    return h.digest()[:16]


# ---------------------------------------------------------------------------
# PrefixCache
# ---------------------------------------------------------------------------

class PrefixCache:
    """LRU prefix cache.  Daemon owns the GPU slots; Python tracks hash→slot.

    Parameters
    ----------
    daemon_stdin:
        The ``stdin`` pipe of the daemon subprocess (``subprocess.Popen.stdin``).
    await_reply:
        Async callable ``(prefix: str, timeout: float) -> str`` — provided by
        ``DaemonStdoutBus.await_reply``.
    daemon_lock:
        ``asyncio.Lock`` that serialises all stdin writes + stdout reads.
        Callers must acquire it before calling ``lookup`` and hold it through
        any subsequent ``RESTORE`` / ``SNAPSHOT`` IPC.
    tokenizer:
        HuggingFace tokenizer (used only to resolve Qwen chat marker ids).
    cap:
        Maximum number of snapshot slots.  0 disables the cache entirely.
    log_prefix:
        String prepended to cache-hit/miss log lines.
    """

    # Daemon-side hard cap (PREFIX_CACHE_SLOTS in test_dflash.cpp). Any
    # configured cap > this is silently clamped down — exceeding it would
    # cause silent SNAPSHOT failures on slots ≥ 8.
    DAEMON_MAX_SLOTS = 8

    def __init__(self, *, daemon_stdin, await_reply, daemon_lock,
                 tokenizer, kv_k_type: str, fa_window: int,
                 cap: int = 4, log_prefix: str = "[pc]"):
        self.stdin = daemon_stdin
        self._await_reply = await_reply
        self.lock = daemon_lock
        self.log_prefix = log_prefix
        # Cache key fields — fixed at daemon spawn (env vars passed through).
        # Mismatched values across turns are not possible within one server
        # process, but they're still part of the hash so a daemon restart
        # with different flags doesn't return stale state.
        self.kv_k_type = kv_k_type
        self.fa_window = fa_window

        if cap > self.DAEMON_MAX_SLOTS:
            print(f"{log_prefix} cap={cap} exceeds daemon limit "
                  f"({self.DAEMON_MAX_SLOTS}); clamping", flush=True)
            cap = self.DAEMON_MAX_SLOTS
        self.cap = cap

        if cap <= 0:
            self.disabled = True
            return
        self.disabled = False

        self.entries: OrderedDict[bytes, int] = OrderedDict()  # hash → slot_id
        self.next_slot = 0
        self.im_end, self.im_start, self.system_t = _qwen_marker_ids(tokenizer)
        # Pending eviction: set by prepare_inline_snap when at cap; the old
        # entry is NOT removed until confirm_inline_snap succeeds.  This ensures
        # that if the request aborts before confirm runs, the old entry survives
        # and the daemon slot count stays consistent.
        self._pending_evict_key: bytes | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def boundary(self, ids: list[int]) -> int:
        """Return first boundary (system-prompt end), or -1. Legacy helper."""
        if self.disabled:
            return -1
        return find_prefix_boundary(ids, self.im_end, self.im_start, self.system_t)

    def _all_boundaries(self, ids: list[int]) -> list[int]:
        """Return all candidate cache cut-points in ascending order."""
        return find_all_boundaries(ids, self.im_end, self.im_start, self.system_t)

    def lookup(self, prompt_ids: list[int]) -> tuple[int, int] | None:
        """Return ``(slot_id, prefix_len)`` for the LONGEST cached prefix, or ``None``.

        Iterates all block-aligned turn boundaries in ``prompt_ids``, checks
        each against the LRU index, and returns the deepest (longest) match.

        The caller must already hold ``daemon_lock`` before inspecting the
        returned slot, since the slot id may be evicted by a concurrent
        request otherwise.
        """
        if self.disabled:
            return None
        candidates = self._all_boundaries(prompt_ids)
        best: tuple[int, int] | None = None   # (slot_id, prefix_len)
        for cut in candidates:
            key = hash_prefix(prompt_ids[:cut], self.kv_k_type, self.fa_window)
            if key in self.entries:
                if best is None or cut > best[1]:
                    best = (self.entries[key], cut)
                self.entries.move_to_end(key)   # mark fresh
        if best is not None:
            print(f"{self.log_prefix} lookup hit slot={best[0]} prefix_len={best[1]} "
                  f"(of {len(prompt_ids)} total)", flush=True)
        return best

    def prepare_inline_snap(self, prompt_ids: list[int]) -> tuple[int, int] | None:
        """Pick a target boundary + slot for inline snapshot during the next
        request. Returns ``(slot_id, target_cut)`` or ``None`` if no
        snapshot is needed (e.g. boundary already cached).

        Caller must:
          1. Append ``snap=<target_cut>:<slot_id>`` to the daemon command
             that runs the actual response (bare prompt OR ``RESTORE``).
          2. After the daemon emits ``[snap] inline slot=N cur_pos=M``
             during prefill, call ``confirm_inline_snap(slot_id, target_cut,
             prompt_ids)`` to register the entry in the LRU.

        For an agent loop that monotonically grows conversation history, the
        most valuable cache point is "end of the most recent completed
        assistant message" — i.e., the second-to-last `<|im_start|>`
        boundary. The LAST boundary is the current turn's opening, whose
        content hasn't been generated yet.
        """
        if self.disabled:
            return None
        candidates = self._all_boundaries(prompt_ids)
        if not candidates:
            return None
        target_cut = candidates[-2] if len(candidates) >= 2 else candidates[-1]

        target_key = hash_prefix(prompt_ids[:target_cut],
                                  self.kv_k_type, self.fa_window)
        if target_key in self.entries:
            self.entries.move_to_end(target_key)
            return None   # already cached

        # Pick slot: when at cap, reserve the LRU slot WITHOUT evicting yet.
        # The actual eviction is deferred to confirm_inline_snap so that if the
        # request aborts before confirm runs, the old entry survives and the
        # daemon slot count stays consistent.
        if len(self.entries) >= self.cap:
            # Peek at LRU without removing.
            old_key = next(iter(self.entries))
            slot = self.entries[old_key]
            self._pending_evict_key = old_key
        else:
            slot = self.next_slot
            self.next_slot = (self.next_slot + 1) % self.cap
            self._pending_evict_key = None

        return (slot, target_cut)

    def confirm_inline_snap(self, slot: int, target_cut: int,
                             prompt_ids: list[int]) -> None:
        """Register an inline snapshot in the LRU after the daemon has
        successfully fired ``[snap] inline``. Called from the caller after
        the actual response stream completes.

        If prepare_inline_snap reserved a slot by displacing an LRU entry,
        the eviction happens HERE (atomically with the insert), so an aborted
        request that never reaches confirm leaves the old entry intact.
        """
        if self.disabled:
            return
        # Atomically evict the reserved old entry (if any) and insert the new one.
        if self._pending_evict_key is not None:
            self.entries.pop(self._pending_evict_key, None)
            self._pending_evict_key = None
        key = hash_prefix(prompt_ids[:target_cut],
                          self.kv_k_type, self.fa_window)
        self.entries[key] = slot
        print(f"{self.log_prefix} inline-snap committed slot={slot} "
              f"prefix_len={target_cut}", flush=True)

    def abort_inline_snap(self, slot: int) -> None:
        """Release the reservation made by prepare_inline_snap without
        evicting the old entry or registering a new one.  Call this from
        exception paths where the request failed before the daemon snapshot
        could complete, so the pending eviction is cancelled.
        """
        if self.disabled:
            return
        self._pending_evict_key = None

    # Legacy out-of-band snapshot (kept for backward-compatibility tests
    # that call it directly; new code uses prepare_inline_snap +
    # confirm_inline_snap so the snapshot rides on the actual response).
    async def maybe_snapshot(self, prompt_ids: list[int],
                              token_stream_consumer=None) -> None:
        if self.disabled:
            return
        prep = self.prepare_inline_snap(prompt_ids)
        if prep is None:
            return
        slot, cut = prep

        import os, struct, tempfile
        fd, tmp_path = tempfile.mkstemp(suffix="_prefix.bin")
        with os.fdopen(fd, "wb") as f:
            for t in prompt_ids[:cut]:
                f.write(struct.pack("<i", int(t)))
        try:
            self._send(f"{tmp_path} 0\n")
            if token_stream_consumer is not None:
                await token_stream_consumer()
            self._send(f"SNAPSHOT {slot}\n")
            await self._await_reply("[snap] slot=")
        finally:
            try: os.unlink(tmp_path)
            except OSError: pass
        self.confirm_inline_snap(slot, cut, prompt_ids)

    async def startup_sync(self) -> None:
        """Query the daemon for existing slots and free them all.

        Called once at server startup to ensure Python's hash table is
        consistent with the daemon's slot state (both empty after this).
        """
        if self.disabled:
            return
        async with self.lock:
            self._send("LIST_SLOTS\n")
            reply = await self._await_reply("[snap] slots=")
            slots_str = reply.split("[snap] slots=", 1)[1].strip()
            if not slots_str:
                return
            orphans = [s.strip() for s in slots_str.split(",") if s.strip()]
            for s in orphans:
                self._send(f"FREE_SNAPSHOT {s}\n")
                await self._await_reply("[snap] freed slot=")
            print(f"{self.log_prefix} freed {len(orphans)} orphaned daemon slots",
                  flush=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, line: str) -> None:
        self.stdin.write(line.encode("utf-8"))
        self.stdin.flush()
