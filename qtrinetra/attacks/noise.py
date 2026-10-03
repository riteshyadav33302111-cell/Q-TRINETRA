"""Benign noise profiles (false-alarm measurement)."""
from __future__ import annotations

from qtrinetra.engine.teleport import PauliChannel

NOISE_PROFILES = {
    "none": lambda p: PauliChannel(),
    "depolarizing": PauliChannel.depolarizing,
    "dephasing": PauliChannel.dephasing,
    "bitflip": PauliChannel.bitflip,
    "amplitude_damping": PauliChannel.amplitude_damping_twirled,
}


def noise_channel(profile: str, strength: float) -> PauliChannel:
    if profile not in NOISE_PROFILES:
        raise ValueError(f"unknown noise profile {profile!r}; choose from {list(NOISE_PROFILES)}")
    return NOISE_PROFILES[profile](strength)
