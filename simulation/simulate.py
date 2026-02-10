"""
ngspice simulation of the 2-2-1 analog neural network circuit.

Generates raw SPICE netlists and runs ngspice in batch mode to verify
the circuit produces correct truth-table outputs for AND, OR, XNOR, and NOR.
"""

import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Mathematical model  (reference: README.md equations)
# ---------------------------------------------------------------------------
GAIN = 4.7 / 10.0   # Rfb / Rin = 0.47
DIODE_VF = 0.6       # approximate 1N4148 forward voltage
TL07X_CMR_OFFSET = 4.0  # TL07x common-mode lower limit: V- + 4V


def math_model(x1, x2, w1, w2, w3, w4, t_weight, rectifier='precision',
               opamp_model='ideal', input_atten=1.0):
    """Idealized node-voltage predictions (ignores pot loading & diode drops).

    rectifier: 'precision' for ideal ReLU, 'diode' for ReLU with ~0.6V dead zone.
    opamp_model: 'ideal' or 'tl07x' (models phase inversion below CMR).
    input_atten: attenuation factor applied to x1, x2 (1.0 = no attenuation).
    """
    x1 = x1 * input_atten
    x2 = x2 * input_atten

    inv_x1 = -x1
    inv_x2 = -x2
    w1_out = w1 * x1
    w2_out = w2 * x2
    w3_out = w3 * x1
    w4_out = w4 * x2

    # Hidden inverting summers: h = -0.47 * (weighted_sum)
    h1 = -GAIN * (w1_out + w2_out)
    h2 = -GAIN * (w3_out + w4_out)

    # Rectifiers: a = max(0, h) or max(0, h - Vf)
    cmr_low = -9.0 + TL07X_CMR_OFFSET  # -5V with ±9V supplies
    if rectifier == 'diode':
        a1 = max(0.0, h1 - DIODE_VF)
        a2 = max(0.0, h2 - DIODE_VF)
    elif opamp_model == 'tl07x' and rectifier == 'precision':
        # TL07x phase inversion: when h drops below CMR low limit,
        # op-amp output snaps to +rail, diode conducts → a ≈ 9 - Vf
        a1 = (9.0 - DIODE_VF) if h1 < cmr_low else max(0.0, h1)
        a2 = (9.0 - DIODE_VF) if h2 < cmr_low else max(0.0, h2)
    else:
        a1 = max(0.0, h1)
        a2 = max(0.0, h2)

    # Output inverting summer
    hy = -GAIN * (a1 + a2)

    # Threshold: T pot  top=V-(-9), bottom=V+(+9), wiper = t_out
    # weight=+1 → wiper at top (V-) → t_out=-9; weight=-1 → wiper at bottom (V+) → t_out=+9
    t_out = -9.0 * t_weight

    # Comparator OA8(+ = hy, - = t_out): HIGH when hy > t_out
    y = 9.0 if hy > t_out else -9.0

    return dict(inv_x1=inv_x1, inv_x2=inv_x2,
                w1_out=w1_out, w2_out=w2_out,
                w3_out=w3_out, w4_out=w4_out,
                h1=h1, h2=h2, a1=a1, a2=a2,
                hy=hy, t_out=t_out, y=y)


# ---------------------------------------------------------------------------
# Generate raw SPICE netlist
# ---------------------------------------------------------------------------
def pot_resistors(name, top_node, bot_node, wiper_node, weight):
    """Return two SPICE resistor lines modelling a 10k pot at given weight."""
    pos = (weight + 1.0) / 2.0          # 0=bottom, 1=top
    r_total = 10e3
    r_top = r_total * (1.0 - pos) + 1   # avoid 0 Ω
    r_bot = r_total * pos + 1
    return (f"R{name}t {top_node} {wiper_node} {r_top}\n"
            f"R{name}b {wiper_node} {bot_node} {r_bot}")


def build_netlist(x1_v, x2_v, w1, w2, w3, w4, t_weight, rectifier='precision',
                  opamp_model='ideal', input_div_r=0, nodeset=None):
    """Return a complete SPICE netlist string for the neural-network circuit.

    rectifier: 'precision' for op-amp precision rectifier, 'diode' for bare diode.
    opamp_model: 'ideal' or 'tl07x' (models phase inversion below CMR).
    input_div_r: series resistance (ohms) for input voltage divider (0 = none).
    nodeset: optional dict of {node_name: voltage} hints to aid DC convergence.
    """
    lines = []
    a = lines.append

    a("Neural Network Circuit")
    a("")

    # --- Convergence options ---
    if opamp_model == 'tl07x' or input_div_r > 0:
        a(".options reltol=0.01 abstol=1e-9 vntol=1e-4 gmin=1e-9 itl1=1000")
    else:
        a(".options reltol=0.001 abstol=1e-12 vntol=1e-6 gmin=1e-12 itl1=500 itl4=100")
    a("")

    # --- Op-amp subcircuit: behavioral source, hard-clamped to rails ---
    # TL07x phase inversion is handled in simulate_point() post-processing,
    # not in the SPICE model, to avoid Newton-Raphson convergence issues.
    a(".subckt opamp inp inn out vcc vee")
    a("Rin inp inn 1e12")
    a("B1 out 0 V=min(V(vcc), max(V(vee), 200000*(V(inp)-V(inn))))")
    a(".ends opamp")
    a("")

    # --- Diode models ---
    a(".model d1n4148 D (IS=2.52e-9 RS=0.568 N=1.752 BV=100 IBV=100u)")
    a(".model dled D (IS=1e-20 N=1.8 RS=5)")
    a("")

    # --- Power supplies ---
    a("Vpos vcc 0 9")
    a("Vneg 0 vee 9")
    a("")

    # --- Inputs ---
    if input_div_r > 0:
        a(f"Vx1 x1_raw 0 {x1_v}")
        a(f"Vx2 x2_raw 0 {x2_v}")
        a(f"Rdiv1 x1_raw x1 {input_div_r}")
        a(f"Rdiv2 x2_raw x2 {input_div_r}")
    else:
        a(f"Vx1 x1 0 {x1_v}")
        a(f"Vx2 x2 0 {x2_v}")
    a("")

    # --- Input buffer inverters (gain = -1) ---
    a("* OA1: inverting buffer for x1")
    a("R1 x1 n1 10k")
    a("R2 n1 inv_x1 10k")
    a("XOA1 0 n1 inv_x1 vcc vee opamp")
    a("")
    a("* OA2: inverting buffer for x2")
    a("R3 x2 n2 10k")
    a("R4 n2 inv_x2 10k")
    a("XOA2 0 n2 inv_x2 vcc vee opamp")
    a("")

    # --- Weight potentiometers ---
    a("* Weight pots: top=raw_input, bottom=inverted_input")
    a(pot_resistors("W1", "x1", "inv_x1", "w1_out", w1))
    a(pot_resistors("W2", "x2", "inv_x2", "w2_out", w2))
    a(pot_resistors("W3", "x1", "inv_x1", "w3_out", w3))
    a(pot_resistors("W4", "x2", "inv_x2", "w4_out", w4))
    a("")

    # --- Hidden inverting summers (gain = -4.7k/10k = -0.47) ---
    a("* OA3: hidden neuron 1 summer")
    a("R5 w1_out sum1 10k")
    a("R6 w2_out sum1 10k")
    a("R7 sum1 h1 4.7k")
    a("XOA3 0 sum1 h1 vcc vee opamp")
    a("")
    a("* OA4: hidden neuron 2 summer")
    a("R8 w3_out sum2 10k")
    a("R9 w4_out sum2 10k")
    a("R10 sum2 h2 4.7k")
    a("XOA4 0 sum2 h2 vcc vee opamp")
    a("")

    # --- Rectifiers (ReLU activation) ---
    if rectifier == 'diode':
        # Bare diode rectifier: a ≈ max(0, h - Vf)
        # D1 anode = h1, cathode = a1; R11 pulls a1 to ground when D1 is off.
        a("* D1 + R11: bare diode rectifier for h1")
        a("D1 h1 a1 d1n4148")
        a("R11 a1 0 10k")
        a("")
        a("* D2 + R12: bare diode rectifier for h2")
        a("D2 h2 a2 d1n4148")
        a("R12 a2 0 10k")
        a("")
    else:
        # Precision rectifier: a = max(0, h)
        # OA5(+ = h1, - = a1, out = oa5_out), D1: oa5_out → a1
        # (Schematic had OA5 pins drawn backwards; corrected here.)
        a("* OA5 + D1 + R11: precision rectifier for h1")
        a("XOA5 h1 a1 oa5_out vcc vee opamp")
        a("D1 oa5_out a1 d1n4148")
        a("R11 a1 0 10k")
        a("")
        a("* OA6 + D2 + R12: precision rectifier for h2")
        a("XOA6 h2 a2 oa6_out vcc vee opamp")
        a("D2 oa6_out a2 d1n4148")
        a("R12 a2 0 10k")
        a("")

    # --- Output inverting summer ---
    a("* OA7: output summer")
    a("R13 a1 sum3 10k")
    a("R14 a2 sum3 10k")
    a("R15 sum3 hy 4.7k")
    a("XOA7 0 sum3 hy vcc vee opamp")
    a("")

    # --- Threshold pot ---
    a("* Threshold pot: top=V-(-9), bottom=V+(+9)")
    a(pot_resistors("T", "vee", "vcc", "t_out", t_weight))
    a("")

    # --- Comparator ---
    a("* OA8: comparator (+ = hy, - = t_out)")
    a("XOA8 hy t_out y vcc vee opamp")
    a("")

    # --- LED indicator ---
    a("R16 y led_a 4.7k")
    a("D3 led_a vee dled")
    a("")

    # --- Nodeset hints for convergence ---
    if nodeset:
        parts = [f"V({k})={v}" for k, v in nodeset.items()]
        a(f".nodeset {' '.join(parts)}")
        a("")

    a(".op")
    a(".end")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Run ngspice and parse results
# ---------------------------------------------------------------------------
def run_ngspice(netlist_str):
    """Run ngspice in batch mode, return dict of node voltages."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
        f.write(netlist_str)
        fpath = f.name

    try:
        result = subprocess.run(
            ['ngspice', '-b', fpath],
            capture_output=True, text=True, timeout=30
        )
    finally:
        Path(fpath).unlink(missing_ok=True)

    output = result.stdout + '\n' + result.stderr

    nodes = {}
    in_node_section = False
    for line in output.split('\n'):
        stripped = line.strip()
        if 'Node' in stripped and 'Voltage' in stripped:
            in_node_section = True
            continue
        if 'Source' in stripped and 'Current' in stripped:
            in_node_section = False
            continue
        if in_node_section:
            if stripped.startswith('---') or not stripped:
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    nodes[parts[0].lower()] = float(parts[1])
                except ValueError:
                    pass

    if not nodes:
        raise RuntimeError(
            f"ngspice produced no results.\nstdout:\n{result.stdout[-800:]}\n"
            f"stderr:\n{result.stderr[-800:]}")

    return nodes


def simulate_point(x1_v, x2_v, w1, w2, w3, w4, t_weight, rectifier='precision',
                   opamp_model='ideal', input_div_r=0):
    # Always simulate with ideal opamp (reliable convergence), then apply
    # TL07x phase-inversion correction in post-processing when requested.
    netlist = build_netlist(x1_v, x2_v, w1, w2, w3, w4, t_weight, rectifier,
                            'ideal', input_div_r)
    raw = run_ngspice(netlist)
    result = {}
    for name in ['inv_x1', 'inv_x2', 'w1_out', 'w2_out', 'w3_out', 'w4_out',
                  'h1', 'h2', 'a1', 'a2', 'hy', 't_out', 'y']:
        key = name.lower()
        if key in raw:
            result[name] = raw[key]
        else:
            raise KeyError(f"Node '{name}' not found. Available: {sorted(raw.keys())}")

    # TL07x post-processing: if a precision-rectifier op-amp's input (h node)
    # is below the CMR limit, the real TL07x would phase-invert, snapping its
    # output to +rail.  The diode then conducts → a ≈ 9 - Vf.
    if opamp_model == 'tl07x' and rectifier == 'precision':
        cmr_low = -9.0 + TL07X_CMR_OFFSET  # -5V
        corrected = False
        for h_name, a_name in [('h1', 'a1'), ('h2', 'a2')]:
            if result[h_name] < cmr_low:
                result[a_name] = 9.0 - DIODE_VF
                corrected = True
        if corrected:
            # Recompute hy and y from corrected a values
            result['hy'] = -GAIN * (result['a1'] + result['a2'])
            result['y'] = 9.0 if result['hy'] > result['t_out'] else -9.0

    return result


def compute_input_atten(input_div_r):
    """Return the voltage attenuation factor for a given series divider resistor.

    The load on each x node is: buffer R_in (10k) in parallel with two pot
    effective resistances (each ~5k max → 2.5k parallel), giving ~2k total.
    """
    r_load = 1.0 / (1.0/10e3 + 1.0/2.5e3)  # ≈ 2k
    return r_load / (input_div_r + r_load)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
INPUT_COMBOS = [(-9, -9), (-9, 9), (9, -9), (9, 9)]

# The math model uses:  a = max(0, h),  where h = -0.47*(weighted_sum).
# h > 0 iff weighted_sum < 0.
# hy = -0.47*(a1 + a2) ≤ 0.
# Comparator: HIGH when hy > t_out.
#
# Strategy for weight selection:
#   To get a function where specific input combos produce HIGH output,
#   those combos must have hy closer to 0 (fewer/weaker activations),
#   while the LOW combos must have more negative hy (stronger activations).
TEST_CASES = {
    'AND': {
        # w1=-1, w2=0  →  h1 = -0.47*(-x1) = 0.47*x1
        #   h1 > 0 (active) when x1 > 0 ⟹ a1 = h1 = 4.23 when x1=-9? No!
        #   h1 = 0.47*x1.  x1=+9 → h1=4.23 → a1=4.23.  x1=-9 → h1=-4.23 → a1=0.
        # w3=0, w4=-1  →  h2 = 0.47*x2.  x2=+9 → a2=4.23.  x2=-9 → a2=0.
        # hy = -0.47*(a1+a2):
        #   (+9,+9): hy = -0.47*8.46 = -3.98  (both active)
        #   (+9,-9): hy = -0.47*4.23 = -1.99  (one active)
        #   (-9,+9): hy = -1.99               (one active)
        #   (-9,-9): hy = 0                   (none active)
        # For AND (only +9,+9 = HIGH), need hy > t_out only for (-9,-9).
        # Wait — (+9,+9) has hy=-3.98 (most negative), can't be > t_out while
        # others are not.
        #
        # Flip: w1=1,w2=0,w3=0,w4=1
        #   h1 = -0.47*x1.  x1=+9 → h1=-4.23 → a1=0.  x1=-9 → h1=4.23 → a1=4.23.
        #   h2 = -0.47*x2.  x2=+9 → a2=0.  x2=-9 → a2=4.23.
        #   (+9,+9): both inactive → hy = 0
        #   (+9,-9): h2 active → hy = -1.99
        #   (-9,+9): h1 active → hy = -1.99
        #   (-9,-9): both active → hy = -3.98
        #
        # AND: only (+9,+9) HIGH. Need t_out in (-1.99, 0).
        #   t_out = -1.0V → t_weight = -1/9
        'weights': (1, 0, 0, 1),
        'threshold': 1.0 / 9.0,     # t_out = -9*(1/9) = -1.0V
        'expected': {(-9,-9): 0, (-9,9): 0, (9,-9): 0, (9,9): 1},
    },
    'OR': {
        # Same weights as AND. OR: (+9,+9),(+9,-9),(-9,+9) HIGH; (-9,-9) LOW.
        # Need t_out in (-3.98, -1.99). Use t_out = -3.0V.
        'weights': (1, 0, 0, 1),
        'threshold': 3.0 / 9.0,     # t_out = -9*(3/9) = -3.0V
        'expected': {(-9,-9): 0, (-9,9): 1, (9,-9): 1, (9,9): 1},
    },
    'XNOR': {
        # w1=-1,w2=1,w3=1,w4=-1:
        #   h1 = -0.47*(-x1+x2). Active when -x1+x2 < 0, i.e., x1 > x2.
        #     (+9,-9): h1 = -0.47*(-18) = 8.46 → a1 = 8.46
        #     All others: h1 ≤ 0 → a1 = 0
        #   h2 = -0.47*(x1-x2). Active when x1-x2 < 0, i.e., x2 > x1.
        #     (-9,+9): h2 = -0.47*(-18) = 8.46 → a2 = 8.46
        #     All others: h2 ≤ 0 → a2 = 0
        #   (+9,+9): hy = 0    (same inputs → no activation)
        #   (+9,-9): hy = -0.47*8.46 = -3.98
        #   (-9,+9): hy = -3.98
        #   (-9,-9): hy = 0    (same inputs → no activation)
        # XNOR: same-input cases HIGH. t_out in (-3.98, 0). Use -2.0V.
        'weights': (-1, 1, 1, -1),
        'threshold': 2.0 / 9.0,     # t_out = -9*(2/9) = -2.0V
        'expected': {(-9,-9): 1, (-9,9): 0, (9,-9): 0, (9,9): 1},
    },
    'NOR': {
        # w1=-1,w2=0,w3=0,w4=-1: neurons activate when inputs are positive.
        #   h1 = 0.47*x1. a1 > 0 when x1 > 0.
        #   h2 = 0.47*x2. a2 > 0 when x2 > 0.
        #   (-9,-9): hy = 0  (no activation)
        #   (-9,+9): hy = -1.99
        #   (+9,-9): hy = -1.99
        #   (+9,+9): hy = -3.98
        # NOR: only (-9,-9) HIGH. t_out in (-1.99, 0). Use -0.5V.
        'weights': (-1, 0, 0, -1),
        'threshold': 0.5 / 9.0,     # t_out = -9*(0.5/9) = -0.5V
        'expected': {(-9,-9): 1, (-9,9): 0, (9,-9): 0, (9,9): 0},
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("  Analog Neural Network (2-2-1) — ngspice Circuit Verification")
    print("=" * 80)

    all_pass = True

    for func_name, cfg in TEST_CASES.items():
        w1, w2, w3, w4 = cfg['weights']
        t_w = cfg['threshold']
        expected = cfg['expected']

        print(f"\n{'─' * 80}")
        print(f"  Function: {func_name}")
        print(f"  Weights:  w1={w1:+.2f}  w2={w2:+.2f}  w3={w3:+.2f}  w4={w4:+.2f}")
        print(f"  Threshold weight: {t_w:+.4f}  (T_OUT ~ {9*t_w:+.2f} V)")
        print(f"{'─' * 80}")

        print(f"  {'X1':>4} {'X2':>4} | {'Node':>8} {'Math':>8} {'SPICE':>8} {'Delta':>7}"
              f" | {'Out':>3} {'Exp':>3} {'OK':>4}")
        print(f"  {'-'*9}-+-{'-'*37}-+-{'-'*14}")

        for (x1_v, x2_v) in INPUT_COMBOS:
            m = math_model(x1_v, x2_v, w1, w2, w3, w4, t_w)

            try:
                s = simulate_point(x1_v, x2_v, w1, w2, w3, w4, t_w)
            except Exception as e:
                print(f"  {x1_v:+4.0f} {x2_v:+4.0f} | ERROR: {e}")
                all_pass = False
                continue

            spice_out = 1 if s['y'] > 0 else 0
            exp_out = expected[(x1_v, x2_v)]
            match = spice_out == exp_out
            if not match:
                all_pass = False

            key_nodes = ['inv_x1', 'inv_x2', 'h1', 'h2', 'a1', 'a2', 'hy', 't_out', 'y']
            for i, node in enumerate(key_nodes):
                mv = m[node]
                sv = s[node]
                delta = sv - mv
                if i == 0:
                    ok = 'PASS' if match else 'FAIL'
                    print(f"  {x1_v:+4.0f} {x2_v:+4.0f} | {node:>8} {mv:+8.3f}"
                          f" {sv:+8.3f} {delta:+7.3f} | {spice_out:>3}"
                          f" {exp_out:>3} {ok:>4}")
                else:
                    print(f"  {'':>9} | {node:>8} {mv:+8.3f}"
                          f" {sv:+8.3f} {delta:+7.3f} |")
            print(f"  {'':>9} | {'':>37} |")

    print()
    print("=" * 80)
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 80)
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
