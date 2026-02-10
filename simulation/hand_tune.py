"""
Hand-tuning simulation for the 2-2-1 analog neural network.

Implements the manual tuning procedure from the README to find all 16
Boolean functions, using the math model for the inner loop and ngspice
for final verification.

Supports both precision rectifier (ideal ReLU) and bare diode (~0.6V
dead zone) configurations.
"""

import random
import sys

from simulate import math_model, simulate_point, compute_input_atten

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]

FUNC_NAMES = {
    0: "FALSE",           1: "NOR",
    2: "CONVERSE NONIMPL", 3: "NOT A",
    4: "NONIMPL",          5: "NOT B",
    6: "XOR",              7: "NAND",
    8: "AND",              9: "XNOR",
    10: "B",              11: "IMPL",
    12: "A",              13: "CONVERSE IMPL",
    14: "OR",             15: "TRUE",
}

MAX_ITERS = 100  # per sign attempt


def target_table(func_id):
    """Return dict mapping (x1,x2) logical inputs to 0/1 target outputs."""
    return {inp: (func_id >> i) & 1 for i, inp in enumerate(INPUTS)}


def logical_to_volts(bit):
    """Convert logical 0/1 to circuit voltage -9/+9."""
    return 9.0 if bit == 1 else -9.0


# ---------------------------------------------------------------------------
# Fast evaluation via math model
# ---------------------------------------------------------------------------
def evaluate(w1, w2, w3, w4, t_weight, rectifier='precision',
             opamp_model='ideal', input_atten=1.0):
    """Return dict of logical outputs for all 4 input combos using math_model."""
    results = {}
    for (lx1, lx2) in INPUTS:
        x1_v = logical_to_volts(lx1)
        x2_v = logical_to_volts(lx2)
        m = math_model(x1_v, x2_v, w1, w2, w3, w4, t_weight, rectifier,
                       opamp_model, input_atten)
        results[(lx1, lx2)] = 1 if m['y'] > 0 else 0
    return results


# ---------------------------------------------------------------------------
# Hand-tuning procedure (faithful to README steps 1-7)
# ---------------------------------------------------------------------------
def hand_tune(func_id, rng, rectifier='precision',
              opamp_model='ideal', input_atten=1.0):
    """Run the README tuning procedure for a given target function.

    Returns (w1, w2, w3, w4, t_weight, iterations, sign, success).
    """
    target = target_table(func_id)

    for sign in [+1, -1]:
        # Step 1: Set all weights to 0
        w1, w2, w3, w4 = 0.0, 0.0, 0.0, 0.0

        # Step 2: Set threshold so output LED is ON (all HIGH).
        # With all weights at 0, hy = 0 for all inputs.
        # Set t_out slightly below 0 (~1/16 to 1/8 turn past 0).
        t_weight = rng.uniform(1/16, 1/8)  # t_out ~ -0.56 to -1.13V

        for iteration in range(1, MAX_ITERS + 1):
            # Step 3: Cycle through inputs (randomized order), find first incorrect
            outputs = evaluate(w1, w2, w3, w4, t_weight, rectifier,
                               opamp_model, input_atten)

            order = list(INPUTS)
            rng.shuffle(order)
            incorrect_input = None
            for inp in order:
                if outputs[inp] != target[inp]:
                    incorrect_input = inp
                    break

            if incorrect_input is None:
                # All correct — done!
                return w1, w2, w3, w4, t_weight, iteration, sign, True

            lx1, lx2 = incorrect_input

            # Step 4: Select hidden neuron
            # x1=0 → neuron 1 (w1,w2); x1=1 → neuron 2 (w3,w4)
            use_neuron1 = (lx1 == 0)

            # Step 5: Compute gradient for each weight
            # Input ON (+9V) → gradient = +1; Input OFF (-9V) → gradient = -1
            grad_x1 = +1 if lx1 == 1 else -1
            grad_x2 = +1 if lx2 == 1 else -1

            # Step 6: Turn pots ~quarter-turn in the direction of the gradient
            step1 = rng.uniform(0.15, 0.35)
            step2 = rng.uniform(0.15, 0.35)

            if use_neuron1:
                w1 += sign * grad_x1 * step1
                w2 += sign * grad_x2 * step2
            else:
                w3 += sign * grad_x1 * step1
                w4 += sign * grad_x2 * step2

            # Clamp weights to [-1, 1] (physical pot limits)
            w1 = max(-1.0, min(1.0, w1))
            w2 = max(-1.0, min(1.0, w2))
            w3 = max(-1.0, min(1.0, w3))
            w4 = max(-1.0, min(1.0, w4))

            # Threshold stays fixed (not re-adjusted after each weight change)

    # Step 7: Failed both signs
    return 0.0, 0.0, 0.0, 0.0, 0.0, MAX_ITERS * 2, 0, False


# ---------------------------------------------------------------------------
# SPICE-calibrated threshold
# ---------------------------------------------------------------------------
def spice_calibrate_threshold(w1, w2, w3, w4, target, rectifier='precision',
                              opamp_model='ideal', input_div_r=0):
    """Measure actual hy values via ngspice and find optimal threshold.

    Returns (t_weight, success) where success means a separating threshold exists.
    """
    spice_hy = {}
    for (lx1, lx2) in INPUTS:
        x1_v = logical_to_volts(lx1)
        x2_v = logical_to_volts(lx2)
        try:
            s = simulate_point(x1_v, x2_v, w1, w2, w3, w4, 0.0, rectifier,
                               opamp_model, input_div_r)
            spice_hy[(lx1, lx2)] = s['hy']
        except Exception:
            return 0.0, False

    high_hys = [spice_hy[inp] for inp in INPUTS if target[inp] == 1]
    low_hys = [spice_hy[inp] for inp in INPUTS if target[inp] == 0]

    if not high_hys:
        t_out = max(spice_hy.values()) + 0.5
    elif not low_hys:
        t_out = min(spice_hy.values()) - 0.5
    else:
        upper = min(high_hys)
        lower = max(low_hys)
        if upper <= lower:
            return 0.0, False
        t_out = (upper + lower) / 2.0

    t_weight = -t_out / 9.0
    t_weight = max(-1.0, min(1.0, t_weight))
    return t_weight, True


# ---------------------------------------------------------------------------
# ngspice verification
# ---------------------------------------------------------------------------
def verify_with_ngspice(w1, w2, w3, w4, t_weight, target, rectifier='precision',
                        opamp_model='ideal', input_div_r=0):
    """Run ngspice for all 4 inputs, return (all_pass, results_list)."""
    results = []
    all_pass = True
    for (lx1, lx2) in INPUTS:
        x1_v = logical_to_volts(lx1)
        x2_v = logical_to_volts(lx2)
        try:
            s = simulate_point(x1_v, x2_v, w1, w2, w3, w4, t_weight, rectifier,
                               opamp_model, input_div_r)
            spice_out = 1 if s['y'] > 0 else 0
        except Exception:
            spice_out = -1
            s = {'hy': float('nan'), 'y': float('nan')}
            all_pass = False

        expected = target[(lx1, lx2)]
        ok = (spice_out == expected)
        if not ok:
            all_pass = False
        results.append((lx1, lx2, expected, spice_out, s.get('hy', float('nan')), ok))
    return all_pass, results


# ---------------------------------------------------------------------------
# Run all 16 functions for a given rectifier type
# ---------------------------------------------------------------------------
def run_all(rectifier, rng, opamp_model='ideal', input_atten=1.0, input_div_r=0):
    """Tune and verify all 16 functions. Returns (success_count, details)."""
    total = 16
    success_count = 0
    details = []

    for func_id in range(16):
        name = FUNC_NAMES[func_id]
        target = target_table(func_id)
        tt = "".join(str(target[inp]) for inp in INPUTS)

        print(f"\n{'─' * 72}")
        print(f"  Function {func_id:2d}: {name}")
        print(f"  Truth table: ({','.join(f'{a}{b}' for a,b in INPUTS)}) = ({','.join(tt)})")
        print(f"{'─' * 72}")

        w1, w2, w3, w4, t_weight, iters, sign, ok = hand_tune(
            func_id, rng, rectifier, opamp_model, input_atten)

        if not ok:
            print(f"  FAILED to converge after {iters} iterations")
            details.append((func_id, name, False, iters, None))
            continue

        sign_str = "+1" if sign == 1 else "-1"
        print(f"  Converged in {iters} iterations (sign={sign_str})")
        print(f"  Weights: w1={w1:+.3f}  w2={w2:+.3f}  w3={w3:+.3f}  w4={w4:+.3f}")
        print(f"  Threshold weight: {t_weight:+.4f}  (t_out={-9*t_weight:+.2f}V)")

        # SPICE-calibrated threshold for robust verification
        spice_t, cal_ok = spice_calibrate_threshold(w1, w2, w3, w4, target, rectifier,
                                                     opamp_model, input_div_r)
        if cal_ok:
            t_use = spice_t
            print(f"  SPICE-calibrated threshold: {t_use:+.4f}  "
                  f"(t_out={-9*t_use:+.2f}V)")
        else:
            t_use = t_weight

        # Verify with ngspice
        print(f"  ngspice verification:")
        spice_ok, results = verify_with_ngspice(w1, w2, w3, w4, t_use, target, rectifier,
                                                 opamp_model, input_div_r)
        for (lx1, lx2, exp, got, hy, match) in results:
            status = "PASS" if match else "FAIL"
            print(f"    ({lx1},{lx2}) -> expected={exp}  "
                  f"spice={got}  hy={hy:+.3f}  {status}")

        if spice_ok:
            print(f"  ngspice: ALL PASS")
            success_count += 1
        else:
            print(f"  ngspice: MISMATCH")

        details.append((func_id, name, spice_ok, iters, (w1, w2, w3, w4, t_use)))

    return success_count, details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = set(sys.argv[1:])
    rectifier = 'diode' if '--diode' in args else 'precision'
    opamp_model = 'tl07x' if '--tl07x' in args else 'ideal'
    input_div_r = 1800 if '--divider' in args else 0
    input_atten = compute_input_atten(input_div_r) if input_div_r > 0 else 1.0

    rect_label = "Bare Diode" if rectifier == 'diode' else "Precision Rectifier"
    opamp_label = "TL07x (phase inversion modeled)" if opamp_model == 'tl07x' else "Ideal"
    div_label = f"{input_div_r} ohm (atten={input_atten:.3f})" if input_div_r > 0 else "None"

    print("=" * 72)
    print(f"  Hand-Tuning Simulation — All 16 Boolean Functions")
    print(f"  Rectifier:   {rect_label}")
    print(f"  Op-amp model: {opamp_label}")
    print(f"  Input divider: {div_label}")
    print("=" * 72)

    rng = random.Random(42)
    success_count, details = run_all(rectifier, rng, opamp_model, input_atten, input_div_r)

    print(f"\n{'=' * 72}")
    print(f"  Result: {success_count}/16 functions found successfully")
    print(f"{'=' * 72}")
    return 0 if success_count == 16 else 1


if __name__ == '__main__':
    sys.exit(main())
