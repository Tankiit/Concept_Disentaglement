"""Daily gate check: one yes/no per kill rule, read from saved results. The first failing gate
blocks everything after it.   python gates.py
"""
import json, subprocess, sys
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parent/'results'


def _json(path):
    p = R/path/'summary.json'
    return json.loads(p.read_text()) if p.exists() else None


def gate_ball_resolution():
    rows = {'synthetic': (_json('response_l2_s128') or {}).get('ball_resolution_gate', {}).get('min_ball_resolution')}
    for name in ('audit_super4_label_shift_held_out', 'audit_classes_label_shift_held_out',
                 'audit_dinov2_super4_label_shift_held_out', 'audit_dinov2_classes_label_shift_held_out'):
        s = _json(name)
        rows[name] = s and s['ball_resolution_gate']['min_ball_resolution']
    ok = all(v is not None and v > 1 for v in rows.values())
    return ok, rows


def gate_invariants():
    tests = subprocess.run([sys.executable, 'main.py', 'test'], capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parent)
    m = pd.read_csv(R/'response_l2_s128'/'response_matrix.csv')
    exact = m[m.expected == 'exact']
    ok_exact = bool((exact.decision == 'exact_constant').all())
    return tests.returncode == 0 and ok_exact, dict(tests_pass=tests.returncode == 0,
                                                    exact_cells=len(exact), exact_ok=ok_exact)


def gate_a12_coverage():
    s = _json('response_l2_s128')
    ball = pd.DataFrame(s['ball_coverage'])
    real = {}
    for name in ('audit_super4_label_shift_held_out', 'audit_dinov2_super4_label_shift_held_out'):
        c = pd.DataFrame(_json(name)['n_axis_coverage'])
        real[name] = float(c[c.region == 'on_support'].joint_coverage.min())
    ok = bool(ball.demonstrates_at_least_nominal.all()) and all(v >= .95 for v in real.values())
    return ok, dict(synthetic_ball_cp_lower_min=float(ball['lower'].min()), nominal=.975,
                    real_joint_coverage_min=real)


def gate_response_matrix():
    s = _json('response_l2_s128')
    m = pd.read_csv(R/'response_l2_s128'/'response_matrix.csv')
    tested = m[m.expected.notna() & (m.expected != 'exact')]
    populated = bool(tested.slope.notna().all())
    return populated, dict(predicted_cells=s['predicted_cells'], agreeing=s['agreeing'],
                           all_populated=populated)


GATES = [('ball_resolution_gt_1', gate_ball_resolution), ('invariants_green', gate_invariants),
         ('a12_coverage_ok', gate_a12_coverage), ('response_matrix_done', gate_response_matrix)]


def gate_report():
    first_fail = None
    for name, fn in GATES:
        ok, detail = fn()
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {json.dumps(detail, default=float)}")
        if not ok and first_fail is None:
            first_fail = name
    print('first failing gate:', first_fail or 'none')
    return first_fail is None


if __name__ == '__main__':
    raise SystemExit(0 if gate_report() else 1)
