#!/usr/bin/env python
# coding: utf-8

# # Causality and computation
# 
# ## Feedback loops break sensible causal chain
# 
# What is causality? When we typically want to trace the causal chain of events we can follow the energy.
# 
# If we imagine a pool break. The player strikes the cue ball, adding energy to the system. For now, we'll consider the player outside of the system.
# 
# The kinetic energy of the cue ball then get transfered to the other billiard balls when it strikes. The energy spreads out among the balls after many collisions. 
# 
# If we follow the transfer of kinetic energy, we can discern the causal chain of events. This gives us an easy understanding of what it means to have a causal chain of events.
# 
# As the billiard balls bounce around the pool table, the kinetic energy that was added to the system escapes the system as heat through friction and air resistance.
# 
# But lets consider a system for which there is no friction or air resistance (and no pockets). Again we are back to our particles in a box model of physics, where we are modeling the forces of mass and momentum.
# 
# However, we have to be careful about what exactly is happening the moment two pool balls collide. The collision is mediated by the electromagnetic forces between the atoms on the surfaces of the balls. As the balls approach each other, the negatively charged valence electrons push against each other with increasing strength as the balls get closer and closer. The force increases with proximity, and to us on a macro level it appears that nothing happens until the balls touch. However, the balls never really touch, but rather there is always a small (atomic scale) separation, and only because the EM force is strong at very small distances does it appear to us as a collision. The reality is that the balls are feeling the force of each other all the time, but it is very small until they are close. 
# 
# So this is very key to our understanding of causality -- particles in the Universe are always feeling the collective forces from all other particles. Particles are always in a feedback loop, where particle A is influenceed by particle B, and B is influenced it in return by A. This sets up a constant recurrent feedback system.
# 
# Further there is relativity, the notion that the rules of physics do not depend
# on the reference frame of the particle. Thus, if we were in particle A's reference frame, it would appear as if there was a collision from a high-energy particle B, and we would perhaps assign the change in kinetic energy of A as a causal chain of events from particle B. However, if we were in particle B's reference frame, then we would see energy transfer from A to B, assigning the causal chain as from A to B. 
# 
# But this is the key point, it does not really make sense to assign a causal chain of events in this context. A cause changes in B and B causes changes in A, but there is not a clear chain.  A and B are constantly influencing each other, and they are states of a feedback system. In this constant dynamical evolution of the states of the system and when state variables are coupled in a feedback loop, the notion of causal chain is non-sense.
# 
# Eventually in a chaotic system which does not leak any energy, we would see the billiard balls bouncing around indefinitely. We could apply a causal chain from when the outside person added energy to the system,. But once that energy is accounted for, the causal chain of events breaks down, and 
# 
# ## Causality in a computational system
# 
# 
# 
# ## Memory breaks temporal causal consistency
# 

# ## Computational and Inferential Thinking -- data science book, UC Berkeley course.
# 
# https://inferentialthinking.com/chapters/02/causality-and-experiments.html
# 
# Cool data science book. Covers John Snow cholera study. 
# 
# They build up from "observational" studies to "Randomized Control Trials", which are needed to establish causal relationships. 
# 
# Confounding factors make establishing causality hard. They bring up a study that found correlations between coffee drinking and cancer, but there was also a correlation between coffee drinking and smoking. 
# 
# Causality needs a from of active interference in the data set. A randomized control trial has the experimenter actually manipulating the system, interfering with it, which is required for understanding a causal relationship. 
# 
# Too often in science this is not possible. 
# 
# This doesn't much cover aspects of high-dimensionality, and collection of large-scale high-dimensional data. Granger causality would be good to cover. 

# In[ ]:




