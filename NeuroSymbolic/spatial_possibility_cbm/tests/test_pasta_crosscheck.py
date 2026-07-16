"""
Optional: cross-validates pySDD's WMC interval against PASTA (an
independent PASP credal-semantics solver). Skipped automatically if the
`pastasolver` CLI isn't on PATH — install with:
    pip install "spatial-possibility-cbm[crosscheck]"
    pip install git+https://github.com/damianoazzolini/pasta
"""
import shutil
import pytest

pastasolver_available = shutil.which("pastasolver") is not None


@pytest.mark.skipif(not pastasolver_available, reason="pastasolver CLI not installed")
def test_pysdd_matches_pasta_at_both_corners():
    from spatial_possibility_cbm.crosscheck.pasta_crosscheck import run_pysdd, run_pasta

    pysdd_lo, pysdd_hi = run_pysdd()
    pasta_lo = run_pasta("lower")
    pasta_hi = run_pasta("upper")

    assert abs(pysdd_lo - pasta_lo) < 1e-4
    assert abs(pysdd_hi - pasta_hi) < 1e-4
