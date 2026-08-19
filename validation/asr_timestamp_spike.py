#!/usr/bin/env python3
"""ASR word-timestamp accuracy spike: faster-whisper vs NVIDIA Parakeet (NeMo).

Scores each engine's word timings against the hand-corrected Karpathy reference
(karpathy_10m.ref.words.json). Alignment: normalized-word SequenceMatcher; for matched
words report |start_hyp - start_ref| statistics. Rough error rate = 1 - matched/ref_words.

Usage: asr_timestamp_spike.py --engine {faster-whisper,parakeet} --audio WAV --ref JSON --out JSON
"""

from __future__ import annotations

import argparse
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


def run_faster_whisper(audio: str) -> list[dict]:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cuda", compute_type="float16"
    )
    segments, _info = model.transcribe(audio, word_timestamps=True, vad_filter=False)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)})
    return words


def run_parakeet(audio: str) -> list[dict]:
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    out = model.transcribe([audio], timestamps=True)
    hyp = out[0]
    stamps = hyp.timestamp["word"] if hasattr(hyp, "timestamp") else hyp["timestamp"]["word"]
    words = []
    for w in stamps:
        words.append(
            {
                "word": w.get("word", w.get("char", "")),
                "start": round(float(w["start"]), 3),
                "end": round(float(w["end"]), 3),
            }
        )
    return words


def score(ref_words: list[dict], hyp_words: list[dict]) -> dict:
    ref_seq = [norm(w["word"]) for w in ref_words]
    hyp_seq = [norm(w["word"]) for w in hyp_words]
    sm = SequenceMatcher(a=ref_seq, b=hyp_seq, autojunk=False)
    deltas = []
    matched = 0
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            r, h = ref_words[block.a + k], hyp_words[block.b + k]
            deltas.append(abs(h["start"] - r["start"]))
            matched += 1
    deltas.sort()
    n = len(deltas)
    return {
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "matched": matched,
        "match_rate_pct": round(100 * matched / len(ref_words), 2),
        "start_err_mean_ms": round(1000 * sum(deltas) / n, 1) if n else None,
        "start_err_median_ms": round(1000 * deltas[n // 2], 1) if n else None,
        "start_err_p95_ms": round(1000 * deltas[int(n * 0.95)], 1) if n else None,
        "start_err_max_ms": round(1000 * deltas[-1], 1) if n else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["faster-whisper", "parakeet"])
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref_words = json.load(open(args.ref))
    t0 = time.perf_counter()
    hyp_words = run_faster_whisper(args.audio) if args.engine == "faster-whisper" else run_parakeet(args.audio)
    elapsed = time.perf_counter() - t0

    result = {
        "engine": args.engine,
        "audio": Path(args.audio).name,
        "transcribe_s": round(elapsed, 1),
        **score(ref_words, hyp_words),
    }
    Path(args.out).write_text(json.dumps({"summary": result, "words": hyp_words}, indent=1))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    main()
