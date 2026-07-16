"""
spatial_possibility_cbm.crosscheck — independent-solver validation utilities.

pasta_crosscheck: cross-validates ConceptComposer.wmc_interval()'s two-point
WMC against PASTA (github.com/damianoazzolini/pasta), an independent PASP
credal-semantics solver. Requires the `pastasolver` CLI on PATH — see
README for install instructions. Not required for core package use; this
is a correctness check, not a runtime dependency.
"""
