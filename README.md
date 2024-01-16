
# Didac

Didac is a toy ground-up neural network framework in Julia. Its intention is to build neural net models with a focus on obvious code with no dependencies. Didac supports feed-forward neural networks with multiple activation functions, softmax, weight constraints, and dropout [TODO]. Didac also supports feed-forward or RNN neural nets. The goal is to support simple LMM models with Didac, preferably loading real trained models.

Note: Didac is a learning exercise. The intent is to build Didac to be as fast as possible to run real models, but only with such optimizations that don't impair obviousness.

## Getting Started

You will need Julia installed on your system. See [Julia Installation](https://julialang.org/downloads/) for more details. Otherwise, Didac currently has no other dependencies.

## Running Didac

Currently, the best examples as in the `examples/` folder. E.g. to run the traffic light model, simply run `julia examples/trafficlight.jl`. These models are intentionally very simple, but hopefully demonstrate the core functionality of NN techniques and models.

## Further Information

All code is based on the [former Coursera lecture series by Geoffrey Hinton](https://www.youtube.com/playlist?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9). I will try to add comments to the code to link to certain equations from the series that match certain lines of code.