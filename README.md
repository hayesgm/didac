
# Didac

Didac is a toy ground-up neural network framework in Julia. Its intention is to build neural net models with a focus on obvious code with no dependencies. Didac supports feed-forward neural networks with multiple activation functions, softmax, weight constraints, and dropout [TODO]. Didac also supports feed-forward or RNN neural nets. The goal is to support simple LMM models with Didac, preferably loading real trained models.

Note: Didac is a learning exercise. The intent is to build Didac to be as fast as possible to run real models, but only with such optimizations that don't impair obviousness.

## Getting Started

You will need Julia installed on your system. See [Julia Installation](https://julialang.org/downloads/) for more details. Otherwise, Didac currently has no other dependencies.

## Running Didac

Currently, the best examples as in the `examples/` folder. E.g. to run the traffic light model, simply run `julia examples/trafficlight.jl`. These models are intentionally very simple, but hopefully demonstrate the core functionality of NN techniques and models.

## Further Information

All code is based on the [former Coursera lecture series by Geoffrey Hinton](https://www.youtube.com/playlist?list=PLoRl3Ht4JOcdU872GhiYWf6jwrk_SNhz9). I will try to add comments to the code to link to certain equations from the series that match certain lines of code.

The majority of the code is in [didac.jl](/didac.jl), the core learning module. The code is heavily documented and should be helpful in understanding exactly what is occurring and why. The easiest way to get an understand of how it works is to run it against an example (e.g. `julia examples/colors.jl`) and observe how the net learns.

## Areas to Contribute

There are several items which I would hope to add:

- [x] Support for RNN to model sequences
- [ ] Support for Dropout to improve generalization
- [ ] Support for momentum
- [ ] Loading and saving of models and parameters
- [ ] Improving tracking parameters in "models"
- [ ] Experimental support of loading public models
- [ ] Visualizations of weights and activity in model inference

## Contributing and License

The intent of this library is to instruct more than it is to build a fully featured library. Thus, contributions will be judged against obviousness versus features. A clearer feature that helps instruct (e.g. adding `rmsprop` in a simple way) will be judged more favorably than say an optimization that vectorizes gradient descent but makes it significantly harder to understand. Feel free to create different modules (or even forks) for more complex learning procedures, if desired.

[MIT License](/LICENSE.md). Copyright 2024, Geoffrey Hayes
