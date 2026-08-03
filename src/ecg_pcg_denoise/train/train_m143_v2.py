"""Convenience entry point for the final M14.3-v2 training protocol.

The authoritative implementation retains its historical module name
``train_m142_imu``. Final behaviour is selected by a ``*_m143_*_v2.yaml``
configuration; this wrapper intentionally adds no model or optimisation logic.
"""

from ecg_pcg_denoise.train.train_m142_imu import main


if __name__ == "__main__":
    main()
