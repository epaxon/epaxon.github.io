# Spiking Neural Networks

## Robust Computation with Rhythmic Spike Patterns

### Summary

The basis for how we understand neural networks is linear algebra -- vectors and matrices. We see neurons in the brain signal through spikes, and we connect these spikes to vectors through the theory of rate-coding. This means that more spikes from a neuron is related to a larger number in a vector. The synapses between neurons has a synaptic weight, and the spikes influence the target neurons the way that a vector dot products with a weight matrix. 

In this paper, we flesh out an alternative neural code that uses spikes and spike-timing to represent information. The theoretical link between spikes and vectors is the same, except we are now considering complex-valued vectors. Similarly, we use a matrix-vector dot product to represent how neurons communicate with each other through synapses, but again the matrix is also complex-valued. 

The insight of the paper is that complex numbers are related to circles and oscillations. We see oscillations in many different nervous systems and at different frequencies. We show that mathematically, you can consider a single spike as representing a complex number, where the timing of the spike relative to some background oscillation indicates the phase of the complex number. You can also represent a complex-valued weight matrix by including synaptic delays. The spike being delayed by a synaptic delay is equivalent to the phase-shift induced when multiplying two complex numbers. 

In this paper, we use these mathematical connections to build a complex-valued version of the Hopfield network. The Hopfield network is widely known for its description of attractor-dynamics and many theoretical ideas as well as experimental studies point to the use of attractor-dynamics in neural computation. The complex-valued version of the Hopfield network we denoted as Threshold Phasor Associative Memory (TPAM), which allows one to store complex vectors as attractor-states. By providing a cue of a noisy or partial stored pattern, the dynamics of the network would restore the original pattern.

We then built spiking neural networks that implement these dynamics, which leads to a network that can produce stable, precisely-timed patterns of spiking activity. We explored two different ways of building such a network. The first network uses resonate-and-fire neurons, which have an intrinsic oscillatory dynamics, which is simple and more computationally friendly. The second network is more closely inspired by biology, where neurons are standard integrate-and-fire models and the oscillation is induced through recurrent feedback of inhibitory neurons. This version accounts for some neuroscientific principles, such as Dale's Law, mimics more closely what is known about neural circuits in the cortex, and is more directly relatable to electro-physiology experiments. 


### Links


[PNAS website](https://www.pnas.org/content/116/36/18050)

[PDF Available](https://www.researchgate.net/publication/335276895_Robust_computation_with_rhythmic_spike_patterns)

[TPAM Demo Notebook](tpam_demo.ipynb)


## Neuromorphic nearest neighbor search on Intel's Pohoiki Springs

### Summary


### Links

<a href="https://dl.acm.org/doi/pdf/10.1145/3381755.3398695?casa_token=0CyCI7vFe30AAAAA:WFOsToDNmS6rJeIUe4H0_ncDO5Ra5VCpAaMwZma8td7rBjcYV8j1BUHHU_ZkojrhZy4zogCz9PuH"> 
Neuromorphic KNN NICE 2020
</a>