"""Convenience entry point for the final M14.3-v2 evaluation protocol.

The authoritative implementation retains its historical module name
``eval_m142_imu``. Final behaviour is selected by a ``*_m143_*_v2.yaml``
configuration; this wrapper intentionally adds no evaluation logic.
"""

from ecg_pcg_denoise.train.eval_m142_imu import main


if __name__ == "__main__":
    main()
