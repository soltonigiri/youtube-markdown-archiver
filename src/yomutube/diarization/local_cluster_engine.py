from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from yomutube.models import Segment

from .base import DiarizationEngineUnavailable, DiarizationSettings, SpeakerTurn


class LocalClusterDiarizationEngine:
    """Small local fallback diarizer based on audio features and k-means."""

    def __init__(self, settings: DiarizationSettings):
        self.settings = settings

    def diarize(self, audio_path: str | Path, *, segments: list[Segment] | None = None) -> list[SpeakerTurn]:
        sample_rate, audio = _read_wav(audio_path)
        if audio.size == 0:
            return []
        windows = _segment_windows(audio, sample_rate, self.settings, segments=segments)
        if not windows:
            return []
        features = np.vstack([_audio_features(audio[start:end], sample_rate) for start, end, _start_ms, _end_ms in windows])
        labels = _cluster(features, _speaker_count(self.settings, len(windows)))
        speaker_ids = _speaker_ids(labels)
        return [
            SpeakerTurn(
                speaker=speaker,
                start_ms=start_ms,
                end_ms=end_ms,
                metadata={"engine": "local_cluster"},
            )
            for (_start, _end, start_ms, end_ms), speaker in zip(windows, speaker_ids)
        ]


def _read_wav(audio_path: str | Path) -> tuple[int, np.ndarray]:
    try:
        with wave.open(str(audio_path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, OSError) as exc:
        raise DiarizationEngineUnavailable(f"local_cluster could not read wav: {exc}") from exc

    if sample_width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise DiarizationEngineUnavailable(f"local_cluster unsupported wav sample width: {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return sample_rate, data


def _segment_windows(
    audio: np.ndarray,
    sample_rate: int,
    settings: DiarizationSettings,
    *,
    segments: list[Segment] | None,
) -> list[tuple[int, int, int, int]]:
    if segments:
        return [_window_from_segment(segment, len(audio), sample_rate, settings.local_min_segment_ms) for segment in segments]

    window = max(1, round(sample_rate * settings.local_window_ms / 1000))
    hop = max(1, round(sample_rate * settings.local_hop_ms / 1000))
    windows: list[tuple[int, int, int, int]] = []
    for start in range(0, max(1, len(audio) - window + 1), hop):
        end = min(len(audio), start + window)
        chunk = audio[start:end]
        if chunk.size and float(np.sqrt(np.mean(chunk * chunk))) >= 0.003:
            windows.append((start, end, round(start / sample_rate * 1000), round(end / sample_rate * 1000)))
    return windows


def _window_from_segment(segment: Segment, audio_len: int, sample_rate: int, min_segment_ms: int) -> tuple[int, int, int, int]:
    start_ms = int(segment.start_ms)
    end_ms = int(segment.end_ms)
    if end_ms - start_ms < min_segment_ms:
        midpoint = (start_ms + end_ms) // 2
        start_ms = midpoint - min_segment_ms // 2
        end_ms = midpoint + min_segment_ms // 2
    start_ms = max(0, start_ms)
    end_ms = max(start_ms + 1, end_ms)
    start = min(audio_len, round(start_ms * sample_rate / 1000))
    end = min(audio_len, max(start + 1, round(end_ms * sample_rate / 1000)))
    return start, end, start_ms, round(end / sample_rate * 1000)


def _audio_features(chunk: np.ndarray, sample_rate: int) -> np.ndarray:
    if chunk.size == 0:
        return np.zeros(28, dtype=np.float32)
    chunk = chunk.astype(np.float32)
    chunk = chunk - float(np.mean(chunk))
    rms = float(np.sqrt(np.mean(chunk * chunk)))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(chunk))).astype(np.float32))) if chunk.size > 1 else 0.0
    windowed = chunk * np.hanning(chunk.size)
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(chunk.size, d=1.0 / sample_rate)
    power_sum = float(np.sum(spectrum)) + 1e-8
    centroid = float(np.sum(freqs * spectrum) / power_sum) / max(sample_rate / 2, 1)
    band_edges = np.linspace(80, min(sample_rate / 2, 8000), 25)
    bands: list[float] = []
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        value = float(np.mean(spectrum[mask])) if np.any(mask) else 0.0
        bands.append(np.log1p(value))
    return np.array([rms, zcr, centroid, *bands, float(np.max(np.abs(chunk)))], dtype=np.float32)


def _speaker_count(settings: DiarizationSettings, sample_count: int) -> int:
    if settings.min_speakers and settings.max_speakers and settings.min_speakers == settings.max_speakers:
        requested = settings.min_speakers
    else:
        requested = settings.local_num_speakers
    return max(1, min(int(requested or 1), sample_count))


def _cluster(features: np.ndarray, speaker_count: int) -> list[int]:
    if speaker_count <= 1 or len(features) <= 1:
        return [0 for _ in range(len(features))]
    normalized = _standardize(features)
    centers = _initial_centers(normalized, speaker_count)
    labels = np.zeros(len(normalized), dtype=np.int64)
    for _ in range(50):
        distances = np.linalg.norm(normalized[:, None, :] - centers[None, :, :], axis=2)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for index in range(speaker_count):
            members = normalized[labels == index]
            if len(members):
                centers[index] = members.mean(axis=0)
    return [int(label) for label in labels]


def _standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    return (features - mean) / np.where(std < 1e-6, 1.0, std)


def _initial_centers(features: np.ndarray, speaker_count: int) -> np.ndarray:
    centers = [features[0]]
    while len(centers) < speaker_count:
        current = np.vstack(centers)
        distances = np.min(np.linalg.norm(features[:, None, :] - current[None, :, :], axis=2), axis=1)
        centers.append(features[int(np.argmax(distances))])
    return np.vstack(centers)


def _speaker_ids(labels: list[int]) -> list[str]:
    mapping: dict[int, str] = {}
    speakers: list[str] = []
    for label in labels:
        if label not in mapping:
            mapping[label] = f"SPEAKER_{len(mapping):02d}"
        speakers.append(mapping[label])
    return speakers
