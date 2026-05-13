"""
Tokenize a prompt string using a HuggingFace tokenizer (via transformers).

Two output modes:
  --out FILE   Write token IDs as a flat int32 little-endian binary file
               (consumed by the C++ library directly).
  --csv        Print comma-separated token IDs to stdout
               (for use with the --tokens flag of test_gemma4_dflash).

Usage:
    # Binary output (backward-compatible):
    python tokenize_prompt.py --out /tmp/prompt.bin --prompt "Hello, world!"

    # CSV output for --tokens flag:
    python tokenize_prompt.py --csv --prompt "Hello, world!"
    # -> 9259,236764,1902,236888

    # Explicit model:
    python tokenize_prompt.py --csv --model google/gemma-4-26b-a4b-it --prompt "..."

    # Show token count:
    python tokenize_prompt.py --csv --verbose --prompt "Hello, world!"

Notes:
    The Gemma4 tokenizer is cached locally at:
      ~/.cache/huggingface/hub/models--google--gemma-4-26b-a4b-it/
    The script tries local_files_only=True first to avoid network calls.
    Gemma4 vocab size: 262144, BOS token id: 2, EOS token id: 1.
"""

import argparse
import struct
import sys


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True, help="Text to tokenize")
    ap.add_argument("--model", default="google/gemma-4-26b-a4b-it",
                    help="HF repo id whose tokenizer to use "
                         "(default: google/gemma-4-26b-a4b-it)")
    ap.add_argument("--add-bos", action="store_true",
                    help="Prepend BOS token (add_special_tokens=True)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print token count and first/last tokens to stderr")
    # Output modes (at least one required)
    out_group = ap.add_mutually_exclusive_group(required=True)
    out_group.add_argument("--out", metavar="FILE",
                           help="Write int32 binary token ID file")
    out_group.add_argument("--csv", action="store_true",
                           help="Print comma-separated token IDs to stdout")
    return ap


def load_tokenizer(model: str):
    """Load tokenizer, preferring local cache to avoid network calls."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(
            model, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        # Fall back to network if not cached
        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def tokenize(prompt: str, model: str, add_bos: bool) -> list[int]:
    tok = load_tokenizer(model)
    return tok.encode(prompt, add_special_tokens=add_bos)


def main() -> None:
    args = build_parser().parse_args()
    ids = tokenize(args.prompt, args.model, args.add_bos)

    if args.verbose:
        preview = ids[:5] + (["..."] if len(ids) > 10 else []) + ids[-5:] if len(ids) > 10 else ids
        print(f"tokenized {len(ids)} tokens; first/last: {preview}", file=sys.stderr)

    if args.csv:
        print(",".join(str(i) for i in ids))
    else:
        with open(args.out, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        # Informational output goes to stderr so stdout stays clean
        print(f"tokenized {len(ids)} tokens, wrote {args.out} ({len(ids) * 4} bytes)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
