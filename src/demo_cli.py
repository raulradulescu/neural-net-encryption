from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT = "outputs/anc/anc_20260417T102603Z_anc_32_ce357185"


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for the ANC demo. Install it with the README setup command.") from exc
    return torch


def _model_helpers():
    _require_torch()
    from src.models import bit_accuracy, hard_bits_from_logits

    return bit_accuracy, hard_bits_from_logits


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Presentation-friendly ANC demo with optional error correction."
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to an ANC run directory or checkpoint file.",
    )
    parser.add_argument("--text", help="Plaintext text to encrypt. If omitted, the CLI prompts for it.")
    parser.add_argument("--key", help="Passphrase used to derive the model key bits. If omitted, the CLI prompts for it.")
    parser.add_argument("--encoding", default="utf-8", help="Text encoding for plaintext and key material.")
    parser.add_argument(
        "--ecc",
        choices=("none", "hamming74", "repeat3_hamming74"),
        default="repeat3_hamming74",
        help="Outer error-correction layer applied around the ANC transport.",
    )
    parser.add_argument(
        "--primary-path",
        choices=("soft", "hard"),
        default="soft",
        help="Ciphertext representation treated as the main Bob decryption path.",
    )
    parser.add_argument(
        "--show-blocks",
        action="store_true",
        help="Print plaintext and ciphertext blocks in hex.",
    )
    return parser.parse_args(argv)


def prompt_missing_inputs(args: argparse.Namespace) -> argparse.Namespace:
    if args.text is None:
        args.text = input("Plaintext: ")
    if args.key is None:
        args.key = input("Key passphrase: ")
    return args


def bytes_to_bits(data: bytes) -> list[int]:
    bits: list[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("Bit length must be a multiple of 8")

    output = bytearray()
    for offset in range(0, len(bits), 8):
        value = 0
        for bit in bits[offset : offset + 8]:
            value = (value << 1) | int(bit)
        output.append(value)
    return bytes(output)


def pkcs7_pad(data: bytes, block_size: int) -> bytes:
    pad_len = block_size - (len(data) % block_size)
    if pad_len == 0:
        pad_len = block_size
    return data + bytes([pad_len]) * pad_len


def pkcs7_unpad(data: bytes, block_size: int) -> bytes:
    if not data or len(data) % block_size != 0:
        raise ValueError("Invalid padded payload length")

    pad_len = data[-1]
    if pad_len < 1 or pad_len > block_size:
        raise ValueError("Invalid PKCS#7 padding")
    if data[-pad_len:] != bytes([pad_len]) * pad_len:
        raise ValueError("Invalid PKCS#7 padding bytes")
    return data[:-pad_len]


def derive_key_bits(key_text: str, key_bits: int, encoding: str) -> list[int]:
    key_bytes_needed = (key_bits + 7) // 8
    seed_material = key_text.encode(encoding)
    digest = hashlib.sha256(seed_material).digest()
    derived = bytearray()

    counter = 0
    while len(derived) < key_bytes_needed:
        derived.extend(hashlib.sha256(digest + counter.to_bytes(4, "big")).digest())
        counter += 1

    return bytes_to_bits(bytes(derived[:key_bytes_needed]))[:key_bits]


def block_to_tensor(block_bits: list[int]) -> torch.Tensor:
    torch = _require_torch()
    return torch.tensor([block_bits], dtype=torch.float32)


def tensor_to_bit_list(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.squeeze(0).tolist()]


def pad_bits_to_block_size(bits: list[int], block_bits: int) -> tuple[list[int], int]:
    remainder = len(bits) % block_bits
    if remainder == 0:
        return bits[:], 0
    pad_len = block_bits - remainder
    return bits + [0] * pad_len, pad_len


def split_bits_into_blocks(bits: list[int], block_bits: int) -> list[list[int]]:
    if len(bits) % block_bits != 0:
        raise ValueError("Bit stream must align to the model block size")
    return [bits[offset : offset + block_bits] for offset in range(0, len(bits), block_bits)]


def format_blocks_as_hex(blocks: list[list[int]]) -> list[str]:
    return [bits_to_bytes(block).hex() for block in blocks]


def hamming74_encode_nibble(data_bits: list[int]) -> list[int]:
    if len(data_bits) != 4:
        raise ValueError("Hamming(7,4) encoder expects exactly 4 bits")

    d1, d2, d3, d4 = (int(bit) & 1 for bit in data_bits)
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return [p1, p2, d1, p3, d2, d3, d4]


def hamming74_decode_codeword(codeword: list[int]) -> tuple[list[int], bool]:
    if len(codeword) != 7:
        raise ValueError("Hamming(7,4) decoder expects exactly 7 bits")

    corrected = [int(bit) & 1 for bit in codeword]
    s1 = corrected[0] ^ corrected[2] ^ corrected[4] ^ corrected[6]
    s2 = corrected[1] ^ corrected[2] ^ corrected[5] ^ corrected[6]
    s3 = corrected[3] ^ corrected[4] ^ corrected[5] ^ corrected[6]
    syndrome = s1 | (s2 << 1) | (s3 << 2)

    corrected_error = False
    if syndrome:
        corrected[syndrome - 1] ^= 1
        corrected_error = True

    return [corrected[2], corrected[4], corrected[5], corrected[6]], corrected_error


def hamming74_encode_bits(bits: list[int]) -> list[int]:
    if len(bits) % 4 != 0:
        raise ValueError("Hamming(7,4) encoding expects a bit length that is a multiple of 4")

    encoded: list[int] = []
    for offset in range(0, len(bits), 4):
        encoded.extend(hamming74_encode_nibble(bits[offset : offset + 4]))
    return encoded


def hamming74_decode_bits(bits: list[int]) -> tuple[list[int], int]:
    if len(bits) % 7 != 0:
        raise ValueError("Hamming(7,4) decoding expects a bit length that is a multiple of 7")

    decoded: list[int] = []
    corrected_errors = 0
    for offset in range(0, len(bits), 7):
        data_bits, corrected_error = hamming74_decode_codeword(bits[offset : offset + 7])
        decoded.extend(data_bits)
        corrected_errors += int(corrected_error)
    return decoded, corrected_errors


def repeat_bits(bits: list[int], factor: int) -> list[int]:
    repeated: list[int] = []
    for bit in bits:
        repeated.extend([int(bit)] * factor)
    return repeated


def majority_vote(bits: list[int], factor: int) -> tuple[list[int], int]:
    if len(bits) % factor != 0:
        raise ValueError("Repeated bitstream length must be divisible by the repetition factor")

    decoded: list[int] = []
    changed_groups = 0
    for offset in range(0, len(bits), factor):
        group = bits[offset : offset + factor]
        ones = sum(int(bit) for bit in group)
        voted = 1 if ones >= ((factor // 2) + 1) else 0
        decoded.append(voted)
        if any(bit != voted for bit in group):
            changed_groups += 1
    return decoded, changed_groups


@dataclass
class EccDecodeStats:
    corrected_codewords: int = 0
    repetition_votes_changed: int = 0


def ecc_encode(bits: list[int], mode: str) -> list[int]:
    if mode == "none":
        return bits[:]
    if mode == "hamming74":
        return hamming74_encode_bits(bits)
    if mode == "repeat3_hamming74":
        return repeat_bits(hamming74_encode_bits(bits), 3)
    raise ValueError(f"Unsupported ECC mode: {mode}")


def ecc_decode(bits: list[int], mode: str) -> tuple[list[int], EccDecodeStats]:
    stats = EccDecodeStats()
    if mode == "none":
        return bits[:], stats
    if mode == "hamming74":
        decoded, corrected = hamming74_decode_bits(bits)
        stats.corrected_codewords = corrected
        return decoded, stats
    if mode == "repeat3_hamming74":
        voted, changed = majority_vote(bits, 3)
        decoded, corrected = hamming74_decode_bits(voted)
        stats.repetition_votes_changed = changed
        stats.corrected_codewords = corrected
        return decoded, stats
    raise ValueError(f"Unsupported ECC mode: {mode}")


@dataclass
class DecodedView:
    text: str
    raw_hex: str
    bit_accuracy_vs_padded: float
    exact_padded_match: bool
    padding_ok: bool
    decode_error: str | None
    ecc_stats: EccDecodeStats


def decode_transport_bits(
    transport_bits: list[int],
    *,
    ecc_mode: str,
    padded_plaintext_bits: list[int],
    padded_plaintext_bytes: bytes,
    block_bytes: int,
    encoding: str,
) -> DecodedView:
    torch = _require_torch()
    bit_accuracy, _ = _model_helpers()
    payload_bits, ecc_stats = ecc_decode(transport_bits, ecc_mode)
    payload_bytes = bits_to_bytes(payload_bits)

    try:
        text = pkcs7_unpad(payload_bytes, block_bytes).decode(encoding)
        padding_ok = True
        decode_error = None
    except (UnicodeDecodeError, ValueError) as exc:
        text = f"<decode failed: {exc}>"
        padding_ok = False
        decode_error = str(exc)

    payload_tensor = torch.tensor(payload_bits, dtype=torch.float32)
    expected_tensor = torch.tensor(padded_plaintext_bits, dtype=torch.float32)
    accuracy = bit_accuracy(payload_tensor, expected_tensor)

    return DecodedView(
        text=text,
        raw_hex=payload_bytes.hex(),
        bit_accuracy_vs_padded=accuracy,
        exact_padded_match=payload_bytes == padded_plaintext_bytes,
        padding_ok=padding_ok,
        decode_error=decode_error,
        ecc_stats=ecc_stats,
    )


def decrypt_blocks_with_bob(
    bob,
    ciphertext_blocks: list[torch.Tensor],
    key_tensor: torch.Tensor,
) -> list[list[int]]:
    decrypted_blocks: list[list[int]] = []
    torch = _require_torch()
    _, hard_bits_from_logits = _model_helpers()
    with torch.no_grad():
        for block in ciphertext_blocks:
            bob_logits = bob(block, key_tensor)
            bob_bits = hard_bits_from_logits(bob_logits)
            decrypted_blocks.append(tensor_to_bit_list(bob_bits))
    return decrypted_blocks


def guess_blocks_with_eve(eve, ciphertext_blocks: list[torch.Tensor]) -> list[list[int]]:
    guessed_blocks: list[list[int]] = []
    torch = _require_torch()
    _, hard_bits_from_logits = _model_helpers()
    with torch.no_grad():
        for block in ciphertext_blocks:
            eve_logits = eve(block)
            eve_bits = hard_bits_from_logits(eve_logits)
            guessed_blocks.append(tensor_to_bit_list(eve_bits))
    return guessed_blocks


def build_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    torch = _require_torch()
    _, hard_bits_from_logits = _model_helpers()
    from src.training.train_anc import load_checkpoint

    ckpt = load_checkpoint(Path(args.checkpoint))
    cfg = ckpt["config"]
    alice = ckpt["models"]["alice"].eval()
    bob = ckpt["models"]["bob"].eval()
    eve = ckpt["models"]["eve"].eval()

    if cfg.plaintext_len != cfg.key_len or cfg.plaintext_len != cfg.ciphertext_len:
        raise ValueError("Demo expects plaintext_len == key_len == ciphertext_len")
    if cfg.plaintext_len % 8 != 0:
        raise ValueError("Demo expects byte-aligned model block sizes")

    block_bits = int(cfg.plaintext_len)
    block_bytes = block_bits // 8

    plaintext_bytes = args.text.encode(args.encoding)
    padded_plaintext = pkcs7_pad(plaintext_bytes, block_bytes)
    padded_plaintext_bits = bytes_to_bits(padded_plaintext)

    encoded_bits = ecc_encode(padded_plaintext_bits, args.ecc)
    encoded_bits_padded, transport_pad_bits = pad_bits_to_block_size(encoded_bits, block_bits)
    plaintext_blocks = split_bits_into_blocks(encoded_bits_padded, block_bits)

    key_bits = derive_key_bits(args.key, cfg.key_len, args.encoding)
    key_tensor = block_to_tensor(key_bits)

    ciphertext_soft_blocks: list[torch.Tensor] = []
    ciphertext_hard_blocks: list[list[int]] = []
    with torch.no_grad():
        for block_bits_list in plaintext_blocks:
            plain_tensor = block_to_tensor(block_bits_list)
            ct_logits = alice(plain_tensor, key_tensor)
            ct_soft = torch.sigmoid(ct_logits)
            ct_hard = hard_bits_from_logits(ct_logits)
            ciphertext_soft_blocks.append(ct_soft)
            ciphertext_hard_blocks.append(tensor_to_bit_list(ct_hard))

    bob_from_soft_blocks = decrypt_blocks_with_bob(bob, ciphertext_soft_blocks, key_tensor)
    bob_from_hard_blocks = decrypt_blocks_with_bob(
        bob,
        [block_to_tensor(block) for block in ciphertext_hard_blocks],
        key_tensor,
    )
    eve_blocks = guess_blocks_with_eve(eve, ciphertext_soft_blocks)

    payload = {
        "cfg": cfg,
        "block_bits": block_bits,
        "block_bytes": block_bytes,
        "plaintext_bytes": plaintext_bytes,
        "padded_plaintext": padded_plaintext,
        "padded_plaintext_bits": padded_plaintext_bits,
        "encoded_bits": encoded_bits,
        "transport_pad_bits": transport_pad_bits,
        "plaintext_blocks": plaintext_blocks,
        "ciphertext_hard_blocks": ciphertext_hard_blocks,
        "bob_soft_transport_bits": [bit for block in bob_from_soft_blocks for bit in block][: len(encoded_bits)],
        "bob_hard_transport_bits": [bit for block in bob_from_hard_blocks for bit in block][: len(encoded_bits)],
        "eve_transport_bits": [bit for block in eve_blocks for bit in block][: len(encoded_bits)],
        "key_bits": key_bits,
    }
    return payload


def report_view(label: str, view: DecodedView) -> None:
    print(f"{label} raw hex        : {view.raw_hex}")
    print(f"{label} bit accuracy   : {view.bit_accuracy_vs_padded:.4f}")
    print(f"{label} exact match     : {view.exact_padded_match}")
    print(f"{label} padding ok      : {view.padding_ok}")
    if view.ecc_stats.repetition_votes_changed:
        print(f"{label} repeat-3 votes  : {view.ecc_stats.repetition_votes_changed}")
    if view.ecc_stats.corrected_codewords:
        print(f"{label} hamming fixes   : {view.ecc_stats.corrected_codewords}")
    print(f"{label} text            : {view.text}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompt_missing_inputs(args)
    payload = build_run_payload(args)

    block_bytes = int(payload["block_bytes"])
    padded_plaintext_bits = payload["padded_plaintext_bits"]
    padded_plaintext = payload["padded_plaintext"]

    bob_soft_view = decode_transport_bits(
        payload["bob_soft_transport_bits"],
        ecc_mode=args.ecc,
        padded_plaintext_bits=padded_plaintext_bits,
        padded_plaintext_bytes=padded_plaintext,
        block_bytes=block_bytes,
        encoding=args.encoding,
    )
    bob_hard_view = decode_transport_bits(
        payload["bob_hard_transport_bits"],
        ecc_mode=args.ecc,
        padded_plaintext_bits=padded_plaintext_bits,
        padded_plaintext_bytes=padded_plaintext,
        block_bytes=block_bytes,
        encoding=args.encoding,
    )
    eve_view = decode_transport_bits(
        payload["eve_transport_bits"],
        ecc_mode=args.ecc,
        padded_plaintext_bits=padded_plaintext_bits,
        padded_plaintext_bytes=padded_plaintext,
        block_bytes=block_bytes,
        encoding=args.encoding,
    )

    primary_view = bob_soft_view if args.primary_path == "soft" else bob_hard_view

    print(f"checkpoint           : {args.checkpoint}")
    print(f"block size           : {payload['block_bits']} bits ({payload['block_bytes']} bytes)")
    print(f"plaintext            : {args.text!r}")
    print(f"plaintext bytes hex  : {payload['plaintext_bytes'].hex()}")
    print(f"padded plaintext hex : {payload['padded_plaintext'].hex()}")
    print(f"ecc mode             : {args.ecc}")
    print(f"ecc payload bits     : {len(payload['encoded_bits'])}")
    print(f"transport pad bits   : {payload['transport_pad_bits']}")
    print(f"derived key hex      : {bits_to_bytes(payload['key_bits']).hex()}")
    if args.show_blocks:
        print(f"plaintext blocks hex : {format_blocks_as_hex(payload['plaintext_blocks'])}")
        print(f"ciphertext blocks hex: {format_blocks_as_hex(payload['ciphertext_hard_blocks'])}")
    print(f"primary path         : {args.primary_path}")
    print(f"demo success         : {primary_view.padding_ok and primary_view.exact_padded_match}")
    report_view("bob soft", bob_soft_view)
    report_view("bob hard", bob_hard_view)
    report_view("eve", eve_view)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
