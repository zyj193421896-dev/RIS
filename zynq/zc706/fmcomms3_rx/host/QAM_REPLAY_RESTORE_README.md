# 256QAM bound replay in the three-area receiver

The receiver restores preserved user QAM/TDCM backend changes from the verified pre-pull backup. Four existing core modules were unchanged between upstream 9a825cb and 318dff5 before restoration. The 14 restored sources match the original backup manifest. Before-write copies and hashes are in `backups/before_qam_restore_20261110_104208` at repository project level. Current PWM and GUI modules were not replaced with old copies.

## GUI integration contract

`rx706_qam_replay_adapter.load_replay(path, input_path=None)` validates the original JSON before normalization, requires 256QAM and `iq_bound_tdcm_replay`, rejects unknown fields at every parameter level, checks selected capture path and full-file SHA-256, and returns a complete recipe. `validate_replay` performs schema/path checks without reading IQ. `run_replay(recipe, cancel_event=None)` returns `(spectrum, recovered)`, preserves every recovery setting, verifies the checksum before and after recovery, and rejects metadata sample-rate or FFT/averaging changes. It never writes the capture.

GUI workers must pass the complete recipe directly to `run_replay`, rather than reassembling a subset from visible controls. Load is not analysis. The redesigned front door supports its basic settings schema and strict bound replay. Portable `tdcm_parameter_config` remains a backend API, but is not silently converted into basic GUI settings. Old bounded/frozen/headerless recipe formats fail clearly in this front door; their restored backend implementations are not promises of generic blind acquisition.

## Verified real capture

- Recipe: `数据/rx_256qam_100slice_100khz_comp_4pliot_replay.json`
- Input: same filename stem, `.bin`, 122880000 bytes.
- SHA-256: `0d0894f28d99d6fcd2b4a0ea3924673b30e30288c3e2fc3b5cb57bc55096c5f1`
- Exact saved settings: RX1, 15.36 MHz, one-second recovery, 100 ksymbol/s, full compensation settings and four training symbols.
- Headless measured replay: 4.43 seconds; 377 frames, 376 matched training and one failed frame; 96256 candidate payload symbols.
- Nearest-neighbor EVM: 10.722239731975%. BER: not measured.

Training matching and nearest-neighbor EVM are not payload correctness or independent reference EVM. Labels follow the configured transmitter hypothesis; no new independent transmitter-reference verification was performed. No hardware test was run.

## Regression scope

`test_tdcm*.py`: 28 tests passed. Adapter, PWM autotune and frame-builder: 31 tests passed. Portable parameter tests were modernized to keep parameter validation, unchanged-input checks and independent two-capture decoding, replacing removed old-control assertions with strict new workflow load/failure tests. The final parameter/QAM PWM/preamble/four-corner suite passed all 26 tests. Its zero-search-window assertion now explicitly accepts immediate PWM-to-QAM transition while rejecting negative windows. These synthetic tests cover known labels independently of the real capture. Machine-readable metrics and all 15 backend source hashes are in `QAM_REPLAY_VALIDATION_20261110.json`.

Restored frozen special-capture and standalone diagnostic-window paths have not been exercised end-to-end in this delivery. GUI integration and layout tests are separate from the backend test runs. Exact source restoration does not imply all historical GUI entry points are restored.
