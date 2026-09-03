from __future__ import annotations

import json
import time
import wave

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


# ============================================================
# TYPES
# ============================================================


ProgressCallback = Callable[[str, int, int], None]


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class WavInfo:
    channels: int
    sample_width: int
    frame_rate: int
    frame_count: int
    compression_type: str
    compression_name: str

    @property
    def duration_seconds(self) -> float:
        if self.frame_rate <= 0:
            return 0.0

        return self.frame_count / self.frame_rate


@dataclass
class CompressionResult:
    file: str
    k: int
    method: str

    original_size_bytes: int
    compressed_size_bytes: int

    compression_ratio: float
    percent_compression: float

    bitstream_length_bits: int
    bits_per_sample: float

    lossless_verified: bool
    differing_samples: int
    maximum_absolute_error: int

    processing_time_seconds: float

    original_wav: Path
    encoded_ex2: Path
    decoded_wav: Path


# ============================================================
# CONSTANTS
# ============================================================


PREDICTION_FIRST_ORDER = "first_order"
PREDICTION_SECOND_ORDER = "second_order"
PREDICTION_RAW = "raw"

STRATEGY_OFFICIAL = "official"
STRATEGY_SECOND_ORDER = "second_order"
STRATEGY_RAW = "raw"
STRATEGY_HYBRID_V1 = "hybrid_v1"
STRATEGY_HYBRID_V2 = "hybrid_v2"

MAX_OFFICIAL_EX2_BITS = 2**32 - 1


# ============================================================
# WAV I/O
# ============================================================


def read_wav(path: Path | str) -> tuple[np.ndarray, WavInfo]:
    """
    Read a mono, 16-bit PCM WAV file.

    Exercise 2 coursework input files are treated as 16-bit
    mono PCM WAV files. The returned samples are int16 values.
    """

    path = Path(path)

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        compression_type = wav_file.getcomptype()
        compression_name = wav_file.getcompname()
        frames = wav_file.readframes(frame_count)

    if compression_type != "NONE":
        raise ValueError(
            "Exercise 2 expects uncompressed PCM WAV audio."
        )

    if sample_width != 2:
        raise ValueError(
            "Exercise 2 expects 16-bit PCM WAV audio "
            "(sample width = 2 bytes)."
        )

    if channels != 1:
        raise ValueError(
            "This coursework implementation expects mono WAV audio. "
            f"Received {channels} channels."
        )

    samples = np.frombuffer(
        frames,
        dtype="<i2",
    ).copy()

    info = WavInfo(
        channels=channels,
        sample_width=sample_width,
        frame_rate=frame_rate,
        frame_count=frame_count,
        compression_type=compression_type,
        compression_name=compression_name,
    )

    return samples.astype(np.int16, copy=False), info


def save_wav(
    path: Path | str,
    samples: np.ndarray,
    info: WavInfo,
) -> None:
    """
    Save reconstructed 16-bit PCM WAV samples.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    samples_64 = np.asarray(
        samples,
        dtype=np.int64,
    )

    if np.any(samples_64 < -32768) or np.any(samples_64 > 32767):
        raise ValueError(
            "Decoded samples exceed the 16-bit PCM range. "
            "The .ex2 stream, K value, or prediction mode may be incorrect."
        )

    samples_16 = samples_64.astype("<i2")

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(info.channels)
        wav_file.setsampwidth(info.sample_width)
        wav_file.setframerate(info.frame_rate)
        wav_file.setcomptype("NONE", "not compressed")
        wav_file.writeframes(samples_16.tobytes())


# ============================================================
# FIRST-ORDER DELTA
# ============================================================


def delta_encode(samples: np.ndarray) -> np.ndarray:
    """
    First-order delta encoding.

    diffs[0] = samples[0]
    diffs[n] = samples[n] - samples[n - 1]
    """

    samples_32 = np.asarray(
        samples,
        dtype=np.int32,
    )

    if samples_32.size == 0:
        return np.empty(
            0,
            dtype=np.int32,
        )

    diffs = np.empty_like(
        samples_32,
        dtype=np.int32,
    )

    diffs[0] = samples_32[0]
    diffs[1:] = samples_32[1:] - samples_32[:-1]

    return diffs


def delta_decode(diffs: np.ndarray) -> np.ndarray:
    """
    Inverse first-order delta decoding.
    """

    diffs = np.asarray(
        diffs,
        dtype=np.int64,
    )

    if diffs.size == 0:
        return np.empty(
            0,
            dtype=np.int32,
        )

    samples = np.cumsum(
        diffs,
        dtype=np.int64,
    )

    return samples.astype(np.int32)


# ============================================================
# SECOND-ORDER DELTA
# ============================================================


def delta2_encode(samples: np.ndarray) -> np.ndarray:
    """
    Appendix enhancement:
    second-order delta, also called delta-of-delta coding.
    """

    first_order = delta_encode(samples)
    second_order = delta_encode(first_order)

    return second_order


def delta2_decode(second_order: np.ndarray) -> np.ndarray:
    """
    Inverse second-order delta decoding.
    """

    first_order = delta_decode(second_order)
    samples = delta_decode(first_order)

    return samples


# ============================================================
# PREDICTION STRATEGY
# ============================================================


def resolve_prediction_mode(
    strategy: str,
    filename: str,
) -> str:
    """
    Map a UI strategy to the actual prediction mode.

    Official:
        all files -> first-order delta

    Second-order:
        all files -> second-order delta

    Raw:
        all files -> raw samples

    Hybrid V1:
        Sound1 -> second-order delta
        Sound2 -> first-order delta

    Hybrid V2:
        Sound1 -> second-order delta
        Sound2 -> raw samples
    """

    name = filename.lower()

    if strategy == STRATEGY_OFFICIAL:
        return PREDICTION_FIRST_ORDER

    if strategy == STRATEGY_SECOND_ORDER:
        return PREDICTION_SECOND_ORDER

    if strategy == STRATEGY_RAW:
        return PREDICTION_RAW

    if strategy == STRATEGY_HYBRID_V1:
        if name == "sound1.wav":
            return PREDICTION_SECOND_ORDER

        return PREDICTION_FIRST_ORDER

    if strategy == STRATEGY_HYBRID_V2:
        if name == "sound1.wav":
            return PREDICTION_SECOND_ORDER

        return PREDICTION_RAW

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


def prediction_name(mode: str) -> str:
    labels = {
        PREDICTION_FIRST_ORDER: "first-order delta + Rice",
        PREDICTION_SECOND_ORDER: "second-order delta + Rice",
        PREDICTION_RAW: "raw samples + Rice",
    }

    return labels.get(
        mode,
        mode,
    )


def prediction_encode(
    samples: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == PREDICTION_FIRST_ORDER:
        return delta_encode(samples)

    if mode == PREDICTION_SECOND_ORDER:
        return delta2_encode(samples)

    if mode == PREDICTION_RAW:
        return np.asarray(
            samples,
            dtype=np.int32,
        )

    raise ValueError(
        f"Unknown prediction mode: {mode}"
    )


def prediction_decode(
    values: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == PREDICTION_FIRST_ORDER:
        return delta_decode(values)

    if mode == PREDICTION_SECOND_ORDER:
        return delta2_decode(values)

    if mode == PREDICTION_RAW:
        return np.asarray(
            values,
            dtype=np.int32,
        )

    raise ValueError(
        f"Unknown prediction mode: {mode}"
    )


# ============================================================
# SIGNED <-> UNSIGNED MAPPING
# ============================================================


def signed_to_unsigned(value: int) -> int:
    """
    Map signed residuals to non-negative integers.

    positive / zero -> magnitude << 1
    negative        -> (magnitude << 1) | 1
    """

    value = int(value)

    return (abs(value) << 1) | (1 if value < 0 else 0)


def unsigned_to_signed(value: int) -> int:
    value = int(value)

    sign = value & 1
    magnitude = value >> 1

    return -magnitude if sign else magnitude


# ============================================================
# RICE SIZE ESTIMATION
# ============================================================


def estimate_rice_bit_length(
    values: np.ndarray,
    k: int,
) -> int:
    """
    Compute the exact number of Rice bits without constructing
    the bitstream.

    Rice length per value:
        unary quotient = q ones + terminating zero
        remainder      = K bits

        length = q + 1 + K
    """

    if k < 0:
        raise ValueError(
            "Rice parameter K must be >= 0."
        )

    values_64 = np.asarray(
        values,
        dtype=np.int64,
    )

    magnitudes = np.abs(values_64)

    mapped = (
        magnitudes << 1
    ) | (
        values_64 < 0
    ).astype(np.int64)

    quotients = mapped >> k
    lengths = quotients + 1 + k

    return int(
        np.sum(
            lengths,
            dtype=np.int64,
        )
    )


def estimate_ex2_size_bytes(
    values: np.ndarray,
    k: int,
) -> tuple[int, int]:
    """
    Estimate exact official .ex2 size.

    Official .ex2 layout:
        4-byte valid-bit-count header
        packed Rice bits
    """

    bit_length = estimate_rice_bit_length(
        values,
        k,
    )

    packed_bytes = (
        bit_length + 7
    ) // 8

    total_bytes = 4 + packed_bytes

    return bit_length, total_bytes


# ============================================================
# MEMORY-SAFE BIT WRITER
# ============================================================


class BitWriter:
    """
    Stream Rice bits directly to disk.

    Official coursework-compatible .ex2 layout:

        bytes 0..3:
            number of valid Rice bits,
            unsigned big-endian integer

        bytes 4..:
            packed Rice bitstream
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.file = open(
            self.path,
            "wb",
        )

        # Reserve four bytes for valid bit length.
        self.file.write(b"\x00\x00\x00\x00")

        self.current_byte = 0
        self.filled_bits = 0
        self.bit_count = 0
        self.closed = False

    def write_bit(self, bit: int) -> None:
        self.current_byte = (
            self.current_byte << 1
        ) | (
            1 if bit else 0
        )

        self.filled_bits += 1
        self.bit_count += 1

        if self.filled_bits == 8:
            self.file.write(
                bytes([self.current_byte])
            )

            self.current_byte = 0
            self.filled_bits = 0

    def write_ones(self, count: int) -> None:
        """
        Efficiently write a run of unary one-bits.

        This avoids constructing huge Python lists such as:
            [1] * quotient
        """

        count = int(count)

        # Complete a partially filled byte first.
        while count > 0 and self.filled_bits != 0:
            self.write_bit(1)
            count -= 1

        # Write complete 0xFF bytes directly.
        full_bytes = count // 8

        while full_bytes > 0:
            block_size = min(
                full_bytes,
                1_048_576,
            )

            self.file.write(
                b"\xff" * block_size
            )

            self.bit_count += block_size * 8
            full_bytes -= block_size

        # Write remaining one-bits.
        remainder_bits = count % 8

        for _ in range(remainder_bits):
            self.write_bit(1)

    def write_value(
        self,
        value: int,
        bit_count: int,
    ) -> None:
        for bit_index in reversed(range(bit_count)):
            bit = (int(value) >> bit_index) & 1
            self.write_bit(bit)

    def close(self) -> None:
        if self.closed:
            return

        if self.bit_count > MAX_OFFICIAL_EX2_BITS:
            self.file.close()
            self.closed = True

            raise OverflowError(
                "The official 4-byte .ex2 header cannot "
                "represent a bitstream longer than "
                f"{MAX_OFFICIAL_EX2_BITS:,} bits."
            )

        # Pad final byte with zeros.
        if self.filled_bits > 0:
            padded = self.current_byte << (
                8 - self.filled_bits
            )

            self.file.write(
                bytes([padded])
            )

        # Store actual valid bit count.
        self.file.seek(0)

        self.file.write(
            int(self.bit_count).to_bytes(
                4,
                "big",
            )
        )

        self.file.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()


# ============================================================
# MEMORY-SAFE BIT READER
# ============================================================


class BitReader:
    """
    Stream the packed Rice bitstream from an official .ex2 file.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.file = open(
            self.path,
            "rb",
        )

        header = self.file.read(4)

        if len(header) != 4:
            self.file.close()

            raise ValueError(
                "Invalid .ex2 file: missing four-byte header."
            )

        self.bit_length = int.from_bytes(
            header,
            "big",
        )

        self.bits_read = 0
        self.current_byte = 0
        self.remaining_bits = 0

    def _load_next_byte(self) -> None:
        data = self.file.read(1)

        if not data:
            raise EOFError(
                "Unexpected end of .ex2 bitstream."
            )

        self.current_byte = data[0]
        self.remaining_bits = 8

    def read_bit(self) -> int:
        if self.bits_read >= self.bit_length:
            raise EOFError(
                "No more valid bits."
            )

        if self.remaining_bits == 0:
            self._load_next_byte()

        bit_position = self.remaining_bits - 1
        bit = (
            self.current_byte
            >> bit_position
        ) & 1

        self.remaining_bits -= 1
        self.bits_read += 1

        return bit

    def read_unary_ones(self) -> int:
        """
        Read q one-bits followed by the terminating zero.

        Complete 0xFF bytes are consumed eight bits at a time.
        """

        quotient = 0

        while True:
            if self.bits_read >= self.bit_length:
                raise EOFError(
                    "Malformed Rice stream: "
                    "unterminated unary quotient."
                )

            if self.remaining_bits == 0:
                self._load_next_byte()

            valid_remaining = (
                self.bit_length
                - self.bits_read
            )

            # Fast path for a complete byte of one-bits.
            if (
                self.remaining_bits == 8
                and valid_remaining >= 8
                and self.current_byte == 0xFF
            ):
                quotient += 8
                self.bits_read += 8
                self.remaining_bits = 0
                continue

            bit = self.read_bit()

            if bit == 0:
                return quotient

            quotient += 1

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()


# ============================================================
# STREAMING RICE ENCODER
# ============================================================


def rice_encode_to_ex2(
    path: Path | str,
    values: np.ndarray,
    k: int,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 4096,
) -> int:
    """
    Rice encode values directly to an official .ex2 file.

    Returns the number of valid Rice bits.
    """

    if k < 0:
        raise ValueError(
            "Rice parameter K must be >= 0."
        )

    values = np.asarray(values)
    total = len(values)

    remainder_mask = (
        (1 << k) - 1
        if k > 0
        else 0
    )

    writer = BitWriter(path)

    try:
        for index, value in enumerate(
            values,
            start=1,
        ):
            mapped_value = signed_to_unsigned(
                int(value)
            )

            quotient = mapped_value >> k
            remainder = mapped_value & remainder_mask

            # Unary quotient.
            writer.write_ones(quotient)
            writer.write_bit(0)

            # K-bit remainder.
            if k > 0:
                writer.write_value(
                    remainder,
                    k,
                )

            if (
                progress_callback
                and (
                    index == total
                    or index % progress_interval == 0
                )
            ):
                progress_callback(
                    "encode",
                    index,
                    total,
                )

    finally:
        writer.close()

    return writer.bit_count


# ============================================================
# STREAMING RICE DECODER
# ============================================================


def rice_decode_from_ex2(
    path: Path | str,
    k: int,
    expected_sample_count: int | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 4096,
) -> np.ndarray:
    """
    Decode an official .ex2 Rice stream.

    If expected_sample_count is known, output is preallocated.
    Otherwise decoding continues until all valid bits have been
    consumed.
    """

    if k < 0:
        raise ValueError(
            "Rice parameter K must be >= 0."
        )

    reader = BitReader(path)

    try:
        if expected_sample_count is not None:
            total = int(expected_sample_count)

            decoded = np.empty(
                total,
                dtype=np.int32,
            )

            for sample_index in range(total):
                quotient = reader.read_unary_ones()

                remainder = 0

                for _ in range(k):
                    remainder = (
                        remainder << 1
                    ) | reader.read_bit()

                mapped_value = (
                    quotient << k
                ) | remainder

                decoded[sample_index] = unsigned_to_signed(
                    mapped_value
                )

                current = sample_index + 1

                if (
                    progress_callback
                    and (
                        current == total
                        or current % progress_interval == 0
                    )
                ):
                    progress_callback(
                        "decode",
                        current,
                        total,
                    )

            if reader.bits_read != reader.bit_length:
                raise ValueError(
                    "Decoded the expected number of samples, "
                    "but valid Rice bits remain. "
                    "K or the expected sample count may be incorrect."
                )

            return decoded

        # ----------------------------------------------------
        # UNKNOWN SAMPLE COUNT
        # ----------------------------------------------------

        decoded_values: list[int] = []

        while reader.bits_read < reader.bit_length:
            quotient = reader.read_unary_ones()

            remainder = 0

            for _ in range(k):
                remainder = (
                    remainder << 1
                ) | reader.read_bit()

            mapped_value = (
                quotient << k
            ) | remainder

            decoded_values.append(
                unsigned_to_signed(
                    mapped_value
                )
            )

            if (
                progress_callback
                and len(decoded_values) % progress_interval == 0
            ):
                progress_callback(
                    "decode_bits",
                    reader.bits_read,
                    reader.bit_length,
                )

        if progress_callback:
            progress_callback(
                "decode_bits",
                reader.bit_length,
                reader.bit_length,
            )

        return np.asarray(
            decoded_values,
            dtype=np.int32,
        )

    finally:
        reader.close()


# ============================================================
# .EX2 INSPECTION
# ============================================================


def inspect_ex2(path: Path | str) -> dict:
    """
    Inspect the official .ex2 header and payload size.
    """

    path = Path(path)

    with open(path, "rb") as file:
        header = file.read(4)

    if len(header) != 4:
        raise ValueError(
            "Invalid .ex2 file."
        )

    bit_length = int.from_bytes(
        header,
        "big",
    )

    file_size = path.stat().st_size
    payload_size = max(
        0,
        file_size - 4,
    )

    return {
        "valid_bit_length": bit_length,
        "file_size_bytes": file_size,
        "payload_bytes": payload_size,
        "padding_bits": (
            payload_size * 8
            - bit_length
        ),
    }


# ============================================================
# RESIDUAL ANALYSIS
# ============================================================


def residual_statistics(
    path: Path | str,
    prediction_mode: str = PREDICTION_FIRST_ORDER,
) -> dict:
    """
    Compute residual statistics for report and UI analysis.
    """

    samples, _ = read_wav(path)

    residuals = prediction_encode(
        samples,
        prediction_mode,
    )

    absolute = np.abs(
        residuals.astype(np.int64)
    )

    if absolute.size == 0:
        mean_abs = 0.0
        median_abs = 0.0
        max_abs = 0
        zero_percent = 0.0

    else:
        mean_abs = round(
            float(np.mean(absolute)),
            2,
        )

        median_abs = round(
            float(np.median(absolute)),
            2,
        )

        max_abs = int(
            np.max(absolute)
        )

        zero_percent = round(
            float(
                np.mean(residuals == 0)
                * 100
            ),
            2,
        )

    return {
        "file": Path(path).name,
        "prediction_method": prediction_name(prediction_mode),
        "sample_count": len(samples),
        "mean_abs_residual": mean_abs,
        "median_abs_residual": median_abs,
        "max_abs_residual": max_abs,
        "zero_residual_percent": zero_percent,
    }


# ============================================================
# BIT INSPECTOR
# ============================================================


def rice_code_details(
    values: Iterable[int],
    k: int,
    maximum_rows: int = 32,
) -> pd.DataFrame:
    """
    Educational representation of individual Rice codewords.
    """

    rows = []
    remainder_width = k

    for index, value in enumerate(values):
        if index >= maximum_rows:
            break

        value = int(value)
        mapped = signed_to_unsigned(value)

        quotient = mapped >> k

        remainder_mask = (
            (1 << k) - 1
            if k > 0
            else 0
        )

        remainder = mapped & remainder_mask

        unary = (
            "1" * quotient
            + "0"
        )

        if remainder_width > 0:
            remainder_bits = format(
                remainder,
                f"0{remainder_width}b",
            )

        else:
            remainder_bits = ""

        codeword = unary + remainder_bits

        displayed = (
            codeword
            if len(codeword) <= 96
            else codeword[:92] + "..."
        )

        rows.append(
            {
                "Index": index,
                "Signed value": value,
                "Mapped value": mapped,
                "q": quotient,
                "r": remainder,
                "Unary q": (
                    unary
                    if len(unary) <= 64
                    else unary[:60] + "..."
                ),
                "Remainder bits": remainder_bits,
                "Rice codeword": displayed,
                "Code length": len(codeword),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# COMPLETE ENCODE / DECODE / VERIFY
# ============================================================


def compress_wav(
    filepath: Path | str,
    k: int,
    encoded_dir: Path | str,
    decoded_dir: Path | str,
    strategy: str = STRATEGY_OFFICIAL,
    progress_callback: ProgressCallback | None = None,
) -> CompressionResult:
    """
    Complete Exercise 2 pipeline.

        WAV
        -> predictor
        -> Rice encode
        -> .ex2
        -> Rice decode
        -> inverse predictor
        -> decoded WAV
        -> sample-for-sample verification
    """

    filepath = Path(filepath)

    encoded_dir = Path(encoded_dir)
    decoded_dir = Path(decoded_dir)

    encoded_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    decoded_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = time.perf_counter()

    samples, wav_info = read_wav(filepath)

    mode = resolve_prediction_mode(
        strategy,
        filepath.name,
    )

    values = prediction_encode(
        samples,
        mode,
    )

    estimated_bits, _ = estimate_ex2_size_bytes(
        values,
        k,
    )

    if estimated_bits > MAX_OFFICIAL_EX2_BITS:
        raise OverflowError(
            "This experiment would exceed the four-byte "
            ".ex2 bit-length header used by the coursework."
        )

    encoded_path = (
        encoded_dir
        / f"{filepath.stem}_K{k}.ex2"
    )

    actual_bits = rice_encode_to_ex2(
        encoded_path,
        values,
        k,
        progress_callback=progress_callback,
    )

    if actual_bits != estimated_bits:
        raise AssertionError(
            "Internal Rice bit-length consistency check failed."
        )

    decoded_values = rice_decode_from_ex2(
        encoded_path,
        k,
        expected_sample_count=len(values),
        progress_callback=progress_callback,
    )

    decoded_samples = prediction_decode(
        decoded_values,
        mode,
    )

    decoded_path = (
        decoded_dir
        / f"{filepath.stem}_K{k}_Dec.wav"
    )

    save_wav(
        decoded_path,
        decoded_samples,
        wav_info,
    )

    original_16 = samples.astype(
        np.int16,
        copy=False,
    )

    reconstructed_16 = decoded_samples.astype(
        np.int16,
    )

    equality = original_16 == reconstructed_16

    differing_samples = int(
        np.count_nonzero(~equality)
    )

    if len(samples) > 0:
        errors = (
            original_16.astype(np.int32)
            -
            reconstructed_16.astype(np.int32)
        )

        maximum_absolute_error = int(
            np.max(
                np.abs(errors)
            )
        )

    else:
        maximum_absolute_error = 0

    lossless_verified = bool(
        np.array_equal(
            original_16,
            reconstructed_16,
        )
    )

    if not lossless_verified:
        raise AssertionError(
            "Decoded output does not match original samples: "
            f"{filepath.name}, K={k}"
        )

    original_size = filepath.stat().st_size
    compressed_size = encoded_path.stat().st_size

    compression_ratio = (
        compressed_size / original_size
        if original_size > 0
        else 0.0
    )

    percent_compression = (
        (
            original_size
            - compressed_size
        )
        / original_size
        * 100
        if original_size > 0
        else 0.0
    )

    bits_per_sample = (
        actual_bits / len(samples)
        if len(samples) > 0
        else 0.0
    )

    processing_time = (
        time.perf_counter()
        - start_time
    )

    return CompressionResult(
        file=filepath.name,
        k=int(k),
        method=prediction_name(mode),

        original_size_bytes=original_size,
        compressed_size_bytes=compressed_size,

        compression_ratio=round(
            compression_ratio,
            4,
        ),

        percent_compression=round(
            percent_compression,
            2,
        ),

        bitstream_length_bits=actual_bits,

        bits_per_sample=round(
            bits_per_sample,
            4,
        ),

        lossless_verified=lossless_verified,
        differing_samples=differing_samples,
        maximum_absolute_error=maximum_absolute_error,

        processing_time_seconds=round(
            processing_time,
            3,
        ),

        original_wav=filepath,
        encoded_ex2=encoded_path,
        decoded_wav=decoded_path,
    )


# ============================================================
# DECODE EXISTING .EX2
# ============================================================


def decode_ex2_to_wav(
    ex2_path: Path | str,
    k: int,
    output_path: Path | str,
    wav_info: WavInfo,
    prediction_mode: str,
    expected_sample_count: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    """
    Decode an existing coursework .ex2 file.

    Important:
    the .ex2 header stores only the number of valid Rice bits.
    It does not store K, prediction mode, sample rate, or WAV
    metadata. These must therefore be supplied externally.
    """

    decoded_values = rice_decode_from_ex2(
        ex2_path,
        k,
        expected_sample_count=expected_sample_count,
        progress_callback=progress_callback,
    )

    decoded_samples = prediction_decode(
        decoded_values,
        prediction_mode,
    )

    save_wav(
        output_path,
        decoded_samples,
        wav_info,
    )

    return decoded_samples


# ============================================================
# DATAFRAME HELPERS
# ============================================================


def results_dataframe(
    results: list[CompressionResult],
) -> pd.DataFrame:
    """
    Convert CompressionResult objects into a raw technical
    results DataFrame.
    """

    rows = []

    for result in results:
        rows.append(
            {
                "file": result.file,
                "K": result.k,
                "method": result.method,
                "original_size_bytes": result.original_size_bytes,
                "compressed_size_bytes": result.compressed_size_bytes,
                "compression_ratio": result.compression_ratio,
                "percent_compression": result.percent_compression,
                "bitstream_length_bits": result.bitstream_length_bits,
                "bits_per_sample": result.bits_per_sample,
                "lossless_verified": result.lossless_verified,
                "differing_samples": result.differing_samples,
                "maximum_absolute_error": result.maximum_absolute_error,
                "processing_time_seconds": result.processing_time_seconds,
                "encoded_ex2": str(result.encoded_ex2),
                "decoded_wav": str(result.decoded_wav),
            }
        )

    return pd.DataFrame(rows)


def build_coursework_table(
    results_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Reproduce the official coursework table where K=2 and K=4
    results are available.
    """

    if results_df.empty:
        return None

    required = {2, 4}

    available = set(
        results_df["K"].astype(int)
    )

    if not required.issubset(available):
        return None

    encoded_size_table = results_df.pivot(
        index="file",
        columns="K",
        values="compressed_size_bytes",
    )

    compression_table = results_df.pivot(
        index="file",
        columns="K",
        values="percent_compression",
    )

    originals = results_df.groupby("file")[
        "original_size_bytes"
    ].first()

    table = pd.DataFrame(
        {
            "Original size (bytes)": originals,
            "Rice (K = 4 bits)": encoded_size_table[4],
            "Rice (K = 2 bits)": encoded_size_table[2],
            "% Compression (K = 4 bits)": compression_table[4],
            "% Compression (K = 2 bits)": compression_table[2],
        }
    ).reset_index()

    table = table.rename(
        columns={
            "file": "Audio file",
        }
    )

    order = {
        "Sound1.wav": 0,
        "Sound2.wav": 1,
    }

    table["_sort"] = (
        table["Audio file"]
        .map(order)
        .fillna(999)
    )

    table = (
        table
        .sort_values(
            [
                "_sort",
                "Audio file",
            ]
        )
        .drop(
            columns=["_sort"]
        )
        .reset_index(drop=True)
    )

    return table


def build_lossless_table(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a lossless verification table.
    """

    if results_df.empty:
        return pd.DataFrame()

    columns = [
        "file",
        "K",
        "method",
        "lossless_verified",
        "differing_samples",
        "maximum_absolute_error",
        "encoded_ex2",
        "decoded_wav",
    ]

    return results_df[columns].copy()


# ============================================================
# RUN CONFIGURATION
# ============================================================


def save_run_configuration(
    path: Path | str,
    configuration: dict,
) -> None:
    """
    Save a run configuration JSON file.
    """

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2,
        )


# ============================================================
# SIZE FORMATTER
# ============================================================


def human_bytes(size: int | float) -> str:
    """
    Convert a byte count into a readable string.
    """

    size = float(size)

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    index = 0

    while (
        size >= 1024
        and index < len(units) - 1
    ):
        size /= 1024
        index += 1

    if index == 0:
        return (
            f"{int(size):,} "
            f"{units[index]}"
        )

    return (
        f"{size:.2f} "
        f"{units[index]}"
    )


# ============================================================
# STREAMLIT DISPLAY TABLES
# ============================================================


def build_final_actual_results_table(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a presentation-ready table for the Streamlit app.

    Unlike the preflight table, this table uses the actual
    .ex2 files written to disk, the decoded WAV files, and
    the sample-for-sample lossless verification result.
    """

    if results_df.empty:
        return pd.DataFrame()

    required_columns = {
        "file",
        "K",
        "method",
        "original_size_bytes",
        "compressed_size_bytes",
        "compression_ratio",
        "percent_compression",
        "bitstream_length_bits",
        "bits_per_sample",
        "lossless_verified",
        "differing_samples",
        "maximum_absolute_error",
        "processing_time_seconds",
    }

    missing_columns = (
        required_columns
        - set(results_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Cannot build final actual results table. "
            "Missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    rows = []

    ordered = (
        results_df
        .copy()
        .sort_values(
            [
                "file",
                "K",
            ]
        )
    )

    for _, row in ordered.iterrows():
        verified = bool(
            row["lossless_verified"]
        )

        rows.append(
            {
                "File": row["file"],
                "K": int(row["K"]),
                "Prediction": row["method"],
                "Original WAV": human_bytes(row["original_size_bytes"]),
                "Actual .ex2": human_bytes(row["compressed_size_bytes"]),
                "Compression ratio": f"{float(row['compression_ratio']):.4f}",
                "Compression": f"{float(row['percent_compression']):.2f}%",
                "Rice bits": f"{int(row['bitstream_length_bits']):,}",
                "Bits / sample": f"{float(row['bits_per_sample']):.4f}",
                "Lossless": "PASS" if verified else "FAIL",
                "Differing samples": f"{int(row['differing_samples']):,}",
                "Maximum error": int(row["maximum_absolute_error"]),
                "Runtime": f"{float(row['processing_time_seconds']):.3f} s",
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SAVED OFFICIAL COURSEWORK RESULTS
# ============================================================


def official_coursework_results_dataframe() -> pd.DataFrame:
    """
    Return the official Exercise 2 results from the final
    Jupyter notebook.

    This is intended for Streamlit demonstration and appendix
    screenshots. It avoids recomputing the extremely slow
    Sound2.wav, K=2 Rice decoding case during UI presentation.

    The actual notebook implementation remains the official
    computational evidence for the coursework.
    """

    rows = [
        {
            "file": "Sound1.wav",
            "K": 2,
            "method": "first-order delta + Rice",
            "original_size_bytes": 1002088,
            "compressed_size_bytes": 2442474,
            "compression_ratio": 2.4374,
            "percent_compression": -143.74,
            "bitstream_length_bits": 19539760,
            "bits_per_sample": 39.0000,
            "lossless_verified": True,
            "differing_samples": 0,
            "maximum_absolute_error": 0,
            "processing_time_seconds": 8.90,
            "encoded_ex2": "saved_official_results/Sound1_K2.ex2",
            "decoded_wav": "saved_official_results/Sound1_K2_Dec.wav",
        },
        {
            "file": "Sound1.wav",
            "K": 4,
            "method": "first-order delta + Rice",
            "original_size_bytes": 1002088,
            "compressed_size_bytes": 859795,
            "compression_ratio": 0.8580,
            "percent_compression": 14.20,
            "bitstream_length_bits": 6878328,
            "bits_per_sample": 13.7286,
            "lossless_verified": True,
            "differing_samples": 0,
            "maximum_absolute_error": 0,
            "processing_time_seconds": 4.81,
            "encoded_ex2": "saved_official_results/Sound1_K4.ex2",
            "decoded_wav": "saved_official_results/Sound1_K4_Dec.wav",
        },
        {
            "file": "Sound2.wav",
            "K": 2,
            "method": "first-order delta + Rice",
            "original_size_bytes": 1008044,
            "compressed_size_bytes": 226212282,
            "compression_ratio": 224.4072,
            "percent_compression": -22340.72,
            "bitstream_length_bits": 1809698224,
            "bits_per_sample": 3590.6711,
            "lossless_verified": True,
            "differing_samples": 0,
            "maximum_absolute_error": 0,
            "processing_time_seconds": 1086.05,
            "encoded_ex2": "saved_official_results/Sound2_K2.ex2",
            "decoded_wav": "saved_official_results/Sound2_K2_Dec.wav",
        },
        {
            "file": "Sound2.wav",
            "K": 4,
            "method": "first-order delta + Rice",
            "original_size_bytes": 1008044,
            "compressed_size_bytes": 56797172,
            "compression_ratio": 56.3439,
            "percent_compression": -5534.39,
            "bitstream_length_bits": 454377344,
            "bits_per_sample": 901.5423,
            "lossless_verified": True,
            "differing_samples": 0,
            "maximum_absolute_error": 0,
            "processing_time_seconds": 139.56,
            "encoded_ex2": "saved_official_results/Sound2_K4.ex2",
            "decoded_wav": "saved_official_results/Sound2_K4_Dec.wav",
        },
    ]

    return pd.DataFrame(rows)


def official_residual_statistics_dataframe() -> pd.DataFrame:
    """
    Return the official first-order residual statistics used in
    the Exercise 2 report.
    """

    rows = [
        {
            "file": "Sound1.wav",
            "prediction_method": "first-order delta + Rice",
            "sample_count": 501022,
            "mean_abs_residual": 72.49,
            "median_abs_residual": 12.0,
            "max_abs_residual": 3571,
            "zero_residual_percent": "not displayed",
        },
        {
            "file": "Sound2.wav",
            "prediction_method": "first-order delta + Rice",
            "sample_count": 504000,
            "mean_abs_residual": 7175.84,
            "median_abs_residual": 4857.0,
            "max_abs_residual": 50712,
            "zero_residual_percent": "not displayed",
        },
    ]

    return pd.DataFrame(rows)
