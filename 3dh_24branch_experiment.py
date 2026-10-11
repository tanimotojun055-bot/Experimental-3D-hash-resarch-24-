　#!/usr/bin/env python3
"""3DH-v1 Candidate A + six directional face projections and 24 branch experiments.

Research-only prototype, NOT a cryptographic hash implementation.
3D mixing is reproduced from the author's Oct. 8, 2026 note Candidate A.
The projection folding function is a NEW, provisional experimental design,
not a confirmed part of 3DH-v1. The full 3DH final digest is not defined.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import struct
import time
from typing import Iterator

import numpy as np

MASK64 = (1 << 64) - 1
FACES = (
    ("+X", 0, False), ("-X", 0, True),
    ("+Y", 1, False), ("-Y", 1, True),
    ("+Z", 2, False), ("-Z", 2, True),
)


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def rotl64(a: np.ndarray, r: int) -> np.ndarray:
    if not 1 <= r <= 63:
        raise ValueError("r must be 1..63")
    return (a << np.uint64(r)) | (a >> np.uint64(64 - r))


def rotr64(a: np.ndarray, r: int) -> np.ndarray:
    return (a >> np.uint64(r)) | (a << np.uint64(64 - r))


def schedule(t: int, n: int) -> tuple[int, int, int, int, int, int, int]:
    span = n - 1
    dx = ((2 * t + 1) % span) + 1
    dy = ((4 * t + 3) % span) + 1
    dz = ((6 * t + 5) % span) + 1
    if t & 1: dx = -dx
    if t & 2: dy = -dy
    if t & 4: dz = -dz
    return (dx, dy, dz,
            ((7 * t) % 63) + 1,
            ((13 * t + 17) % 63) + 1,
            ((29 * t + 31) % 63) + 1,
            ((43 * t + 11) % 63) + 1)


def mix_round(p: np.ndarray, t: int) -> np.ndarray:
    """Candidate A 3D-MIX64. Matches the published trial's core operations."""
    dx, dy, dz, r1, r2, r3, r4 = schedule(t, p.shape[0])
    x = np.roll(p, dx, axis=0)
    y = np.roll(p, dy, axis=1)
    z = np.roll(p, dz, axis=2)
    choose = (x & y) ^ ((~x) & z)
    majority = (x & y) ^ (x & z) ^ (y & z)
    with np.errstate(over="ignore"):
        t1 = rotl64(x, r1) + choose + np.uint64(splitmix64(t))
        t2 = rotr64(y, r2) + majority
        return t1 ^ t2 ^ rotl64(rotl64(z, r3) ^ p, r4)


def mix_state(n: int, rounds: int, bit_at: tuple[int, int, int] | None) -> np.ndarray:
    p = np.zeros((n, n, n), dtype=np.uint64)
    if bit_at is not None:
        p[bit_at] = np.uint64(1)
    for t in range(rounds):
        p = mix_round(p, t)
    return p


def face_projection(p: np.ndarray, face_number: int, axis: int, reverse: bool) -> np.ndarray:
    """PROVISIONAL depth-ordered folding of the volume onto an oriented face.

    Not merely the outermost 2D slice: all interior layers are included.
    +axis means depth 0..N-1; -axis means N-1..0.
    The exact formula is for experimenting with 24-branch data flow only.
    It is NOT a cryptographic construction with proven strength.
    """
    n = p.shape[0]
    acc = np.full((n, n), np.uint64(splitmix64(0x3D000 + face_number)), dtype=np.uint64)
    order = range(n - 1, -1, -1) if reverse else range(n)
    for depth, index in enumerate(order):
        plane = np.take(p, index, axis=axis)
        r = ((depth * 7 + face_number * 11) % 63) + 1
        with np.errstate(over="ignore"):
            acc = (rotl64(acc, r) ^ (plane + np.uint64(splitmix64(face_number * 1000 + depth))))
            acc = acc + rotl64(plane, ((3 * depth + 11) % 63) + 1)
    return acc


def branches24(p: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return 6 oriented faces, each partitioned into 4 fixed quadrants."""
    n = p.shape[0]
    h = n // 2
    out = []
    for face_number, (name, axis, reverse) in enumerate(FACES):
        face = face_projection(p, face_number, axis, reverse)
        for q, (i, j) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            out.append((f"{name}:Q{q}", face[i*h:(i+1)*h, j*h:(j+1)*h].copy()))
    assert len(out) == 24
    return out


def branch_difference(base, mutated) -> tuple[int, int, int]:
    changed_branches = changed_bits = total_bits = 0
    for (tag1, v1), (tag2, v2) in zip(base, mutated):
        assert tag1 == tag2
        n_bits = 64 * v1.size
        delta = np.bitwise_xor(v1, v2)
        bits = sum(int(x).bit_count() for x in delta.flat)
        total_bits += n_bits
        changed_bits += bits
        changed_branches += bool(bits)
        print(f"  {tag1:6s} : {bits:5d}/{n_bits:5d} bits ({100 * bits/n_bits:6.2f}%)")
    return changed_branches, changed_bits, total_bits


def experiment_diffusion(n: int, rounds: int) -> None:
    print(f"\n=== DIFFERENTIAL EXPERIMENT: N={n}, rounds={rounds} ===")
    start = time.perf_counter()
    base = branches24(mix_state(n, rounds, None))
    changed = branches24(mix_state(n, rounds, (0, 0, 0)))
    secs = time.perf_counter() - start
    print("Input difference: one bit at P[0,0,0].")
    found, changed_bits, all_bits = branch_difference(base, changed)
    print(f"Changed branches: {found}/24")
    print(f"Change in branch arrays: {100 * changed_bits / all_bits:.4f}%")
    print(f"2 state calculations + 24x2 projections: {secs:.3f} seconds")
    print("NOTE: these are intermediate 2D arrays, NOT a finalized 256-bit digest.")


def input_encoding(x: int, y: int, z: int) -> bytes:
    return b"bit1-position|" + struct.pack(">III", x, y, z)


def flat_branches(n: int, rounds: int, cell: tuple[int, int, int]) -> np.ndarray:
    b = branches24(mix_state(n, rounds, cell))
    return np.concatenate([v.ravel() for _, v in b])


def experiment_search(n: int, rounds: int, secret: tuple[int, int, int] | None) -> None:
    if n > 12:
        raise ValueError("Search demo allows only --n <=12 to limit runtime. Use --mode diffusion for N=64.")
    if secret is None:
        secret = (n-2, n//2, n//3)
    if any(not 0 <= v < n for v in secret):
        raise ValueError("secret coordinates must be inside 0..N-1")
    print(f"\n=== TOY INPUT-POSITION SEARCH: N={n}, rounds={rounds} ===")
    print("MODEL: 1 bit is set at one unknown (x,y,z), otherwise grid is zero.")
    print("This is a deliberately small toy search, NOT an attack on full 3DH.")
    print("Public intermediate observation: all 24 branches (not a public 256-bit 3DH output).")
    print(f"Number of candidate input positions: {n}^3 = {n**3} (NOT 24^k)")

    reference = flat_branches(n, rounds, secret)
    positions = itertools.product(range(n), repeat=3)
    t0 = time.perf_counter()
    guesses3d, found3d = 0, None
    for pos in positions:
        guesses3d += 1
        if np.array_equal(flat_branches(n, rounds, pos), reference):
            found3d = pos
            break
    dt3d = time.perf_counter() - t0
    print(f"3DH trial: guesses={guesses3d}, elapsed={dt3d:.6f}s, found={found3d}")

    # Independent SHA-256 benchmark over THE SAME position-encoded candidate set.
    sha_target = hashlib.sha256(input_encoding(*secret)).digest()
    t0 = time.perf_counter()
    guesses_sha, found_sha = 0, None
    for pos in itertools.product(range(n), repeat=3):
        guesses_sha += 1
        if hashlib.sha256(input_encoding(*pos)).digest() == sha_target:
            found_sha = pos
            break
    dt_sha = time.perf_counter() - t0
    print(f"SHA-256   : guesses={guesses_sha}, elapsed={dt_sha:.6f}s, found={found_sha}")
    if found3d != secret or found_sha != secret:
        raise AssertionError("Toy search failed: check collisions or implementation.")
    print("Interpretation: per-candidate costs differ, but 24 public branches did not")
    print("force 24^k attempts; stronger security remains unproven.")



def experiment_axes(n: int, rounds: int) -> None:
    """Move the activated bit by +1 cell on X, Y and Z and compare outputs.

    Moving a set bit changes two grid bits (one reset and one set), unlike
    flipping one bit at a fixed position. The distinction is important.
    """
    middle = (n // 2, n // 2, n // 2)
    print(f"\n=== X/Y/Z ONE-CELL SHIFT: N={n}, rounds={rounds} ===")
    print(f"Baseline active cell: {middle}; each +1 shift changes TWO grid bits.")
    origin = branches24(mix_state(n, rounds, middle))
    sha_base = hashlib.sha256(input_encoding(*middle)).digest()
    total = 24 * (n // 2) ** 2 * 64
    for axis, label in enumerate(("X", "Y", "Z")):
        nextpos = list(middle)
        nextpos[axis] += 1
        nextpos = tuple(nextpos)
        shifted = branches24(mix_state(n, rounds, nextpos))
        changed = sum(int(v).bit_count()
                      for (_, a), (_, b) in zip(origin, shifted)
                      for v in np.bitwise_xor(a, b).flat)
        sha_new = hashlib.sha256(input_encoding(*nextpos)).digest()
        sha_changed = sum(int(x ^ y).bit_count() for x, y in zip(sha_base, sha_new))
        print(f"  +{label}: 24 raw-branch bits changed = {changed}/{total} "
              f"({100*changed/total:.3f}%); SHA-256 = {sha_changed}/256 "
              f"({100*sha_changed/256:.3f}%)")
    print("Comparisons differ in output size and SHA-256 is an established hash;")
    print("they are diffusion observations, NOT comparable proof of security.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "both", "diffusion", "search", "axes"), default="all")
    parser.add_argument("--n", type=int, default=8, help="even grid size; 64 for previous LIGHT profile")
    parser.add_argument("--rounds", type=int, default=6, help="1..64; use 64 for previous LIGHT profile")
    parser.add_argument("--secret", nargs=3, type=int, metavar=("X", "Y", "Z"))
    args = parser.parse_args()
    if args.n < 4 or args.n % 2:
        parser.error("--n must be an even integer >= 4")
    if not 1 <= args.rounds <= 64:
        parser.error("--rounds must be 1..64")
    print("3DH-v1 Candidate A + 24-branch EXPLORATORY TEST (NOT secure)")
    print("Mix: original experimental Candidate A; face projection: new tentative prototype.")
    if args.mode in ("all", "both", "diffusion"):
        experiment_diffusion(args.n, args.rounds)
    if args.mode in ("all", "both", "search"):
        experiment_search(args.n, args.rounds, tuple(args.secret) if args.secret else None)
    if args.mode in ("all", "axes"):
        experiment_axes(args.n, args.rounds)


if __name__ == "__main__":
    main()

