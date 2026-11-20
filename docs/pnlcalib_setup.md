# PnLCalib setup

The maintained instructions are now in [SETUP.md](SETUP.md#field-calibration-pnlcalib-required-for-processing).
The app expects `SV_kp` and `SV_lines` under `PNLCALIB_ROOT`. The calibration-only CLI also
accepts explicit `--weights-kp` and `--weights-line` paths.

For the current pipeline, read [ARCHITECTURE.md](ARCHITECTURE.md). The original phase-based
proposal is historical: [calibration_prototype_plan.md](calibration_prototype_plan.md).
