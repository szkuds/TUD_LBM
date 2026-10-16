"""Force-balance analysis of chemical-step droplet runs against the reduced ODE of the SI.

Run it with ``tud-lbm force-balance CONFIG.toml [-g inc -g tau ...] [--out DIR]``
(or ``python -m src.simulation_io.analysis.force_balance``). Modules:

* :mod:`.model` — the dimensionless ODE: closed form and numerical integrator
* :mod:`.geometry` — circular-cap relations and the predictive closure M6
* :mod:`.config` — the analysis TOML
* :mod:`.runs` — run loading, scales, duplicate detection, assumption log
* :mod:`.points` — the per-data-point force-balance table
* :mod:`.fits` — friction models M0..M6 and their run-bootstrap fits
* :mod:`.simulate` — forward integration and trajectory metrics
* :mod:`.analyses` — friction dependence, hysteresis, overshoot, collapse
* :mod:`.plots`, :mod:`.report`, :mod:`.pipeline`
"""
