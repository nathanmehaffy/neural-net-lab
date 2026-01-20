# 18100 Op-Amp Lab / Neural Network Lab

A manually tunable 2-input analog neural network capable of learning all 16 Boolean functions, including non-linearly separable ones like XOR and XNOR.

## Circuit Structure

This circuit implements a 2-2-1 feedforward neural network using analog circuit components in the following order:
- 2 op-amp inverting amplifiers for inverting inputs
- 4 potentiometers for weighting inputs by values in the range \[-1, 1\]
- 2 op-amp inverting summers for creating 2 hidden sums
- 2 op-amp precision rectifiers for applying ReLU activations to hidden sums
- 1 op-amp inverting summer for the output hidden sum
- 1 potentiometer for the output threshold
- 1 op-amp comparator for the final binary output

## Mathematical Description

```
Inputs: x1, x2 in {-9, 9}
Weights: w1, w2, w3, w4 in [-1, 1]
Threshold: t in [-9, 9]

Hidden layer:
h1 = (w1 * x1 + w2 * x2) * -0.47
h2 = (w3 * x1 + x4 * x2) * -0.47
a1 = max(0, h1)
a2 = max(0, h2)

Output layer:
hy = (a1 + a2) * 0.47
y = 1 if hy > t else 0
```

## Optimization

### Automatic

`index.html` contains a web app that applies gradient descent to find weights implementing any target function on the network.
Simply select your target function and press the optimize button, then transfer the weight values to the potentiometers on your circuit.
(Note that depending on how you wired your circuit, you may need to invert each of the weight values.)

(Can be used online at [nathanmehaffy.github.io/neural-net-lab](https://nathanmehaffy.github.io/neural-net-lab/))

### Manual

There is also a fairly simple procedure to find any 2-to-1 binary function on the implemented circuit by hand.

First, observe that each input weight is connected to one of the two inputs and one of the two hidden neurons.

Select a target function - that is, a particular mapping between the four possible input combinations ((0,0), (0,1), (1,0), (1,1)) and outputs.
Then, follow this tuning procedure:

1. Set all weights to 0 (the middle).
2. Find the spot where turning the threshold pot flips the output (should be near 0) and put it slightly past that so the output LED is on.
3. Cycle through the four possible inputs until you find one for which the output is incorrect. (If none are incorrect, you are done!)
4. First, determine which of the two hidden neurons to update via the following rule: if the input is (0, 0) or (0, 1), update the first hidden neuron.
    If the input is (1, 0) or (1, 1), update the second hidden neuron.
5. Now, find the gradient: for each of the two weight pots connected to the hidden neuron you selected, look at the input connected to that weight.
    If the input is on, the gradient for that weight is positive. If the input is off, the gradient for that weight is negative.
6. Turn each of the two pots for which you have identified the weight about a quarter-turn in the direction of the gradient. The exact amount doesn't matter.
7. Repeat from step 3 until your network represents the target function.
    If you end up with the inverse of your target function, your network is wired backwards. Don't worry, this isn't a problem -
    just repeat the process but turn the weights in the opposite direction in step 6.

## Repository Contents

This repo contains this `README.md` file, a tuning helper app `index.html`,
  a bill of materials to construct the circuit `BOM.csv`, and a schematic for the full circuit 'schematic.svg'.
