# Portable acceptance fixtures

`qam_replay_synthetic.json` retains the replay option structure but has synthetic placeholder input metadata. Replay adapter tests replace its path and SHA-256 with a temporary zero-filled IQ file. It is a contract-test fixture, not a usable real-capture recipe.

`../tx_qam_fixture.py` contains unchanged transmitter helper definitions extracted from the project's `ris_codebook_gui.py`, with source SHA-256 recorded in its header. Synthetic acceptance tests no longer import a file outside this repository. Payload truth remains confined to test assertions.

Portable logical frame profiles are under `../../host/frame_profiles/acceptance/`. No real IQ recordings are included.
