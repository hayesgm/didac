using Colors
using Sixel
using LinearAlgebra
using Base.MathConstants
using PrettyTables
using Folds

## A simple neural-network which accepts embedded input

### Let's create two layers (1 hidden, 1 output)
### We'll initialize to random values between zero and one
neural_network = [
  #  N11   N12   N13   N14
  [  0.111 0.121 0.131 0.141 # Input 1
     0.112 0.122 0.132 0.142 # Input 2
     0.113 0.123 0.133 0.143 # Input 3
     0.114 0.124 0.134 0.144 # Input 4
  ],
   # N21   N22   N23
  [  0.211 0.221 0.231 # N11
     0.212 0.222 0.232 # N12
     0.213 0.223 0.233 # N13
     0.214 0.224 0.234 # N14
  ]
]

activation(x) = 1 / (1 + e^-x)
activation_derivative(x) = x * ( 1 - x )

# activation(x) = x
# activation_derivative(x) = 1

ϵ = 0.01
## Input grammer
grammar = Set(["red", "green", "very"])

# This could probably be made using an identity matrix and an index paramter
embedding = Dict(
  "red" =>      [1, 0, 0, 0],
  "green" =>    [0, 1, 0, 0],
  "blue" =>     [0, 0, 1, 0],
  "deep-red" => [0, 0, 0, 1]
)

(embedding::Dict)(k) = embedding[k]

# Inverse-embedding, not really used
inverse_embedding = Dict([(v, k) for (k, v) ∈ pairs(embedding)])

## This section is about training

# So first, we need to build some tuples to learn towards
# It's pretty funny since there aren't a lot of different examples
# Available given our grammer, but it doesn't mean we can't ain't learn.
# Also, since our grammar is so limited, we'll probably end up including
# all of our training cases in our results. But that's okay, this is all
# a toy at this phase anyway.
training = [
  ( "red", [0.9, 0.2, 0.2] ),
  ( "green", [0.2, 0.9, 0.2] ),
  ( "blue", [0.2, 0.2, 0.9] ),
  ( "deep-red", [0.8, 0.1, 0.1] )
]

# TODO: Everything below is really all about calculating results

# Note for word2vec, we have both embedding and context matrices.
# input = ["red", "red", "green"]

(input, target) = rand(training)
# (input, target) = training[1]

embedded_input = embedding.([input])


# Just do these one at a time until we parallelize

function calculate_next_layer(layer, weights)
  [activation.(dot(layer, w)) for w ∈ eachcol(weights)]
end

function calculate_value(layer0)
  Folds.reduce(calculate_next_layer, neural_network; init=layer0)
end

function calculate_next_layer_with_hidden(layers, weights)
  [layers; [[activation.(dot(layers[end], w)) for w ∈ eachcol(weights)]]]
end
 
function calculate_value_with_hidden(layer0)
  Folds.reduce(calculate_next_layer_with_hidden, neural_network; init=[layer0])
end

function show_output(output)
  max_length = maximum(length.(output))
  padded_vecs = [vcat(vec, fill(NaN, max_length - length(vec))) for vec in reverse(output)]
  mat = hcat(padded_vecs...)'
  pretty_table(mat, header=repeat(["Column"], max_length))
end

output_with_hidden = [calculate_value_with_hidden(i) for i ∈ embedded_input]
[show_output(o) for o ∈ output_with_hidden]

output = [calculate_value(i) for i ∈ embedded_input]

# TODO: We probably don't want to use a decoder at all
output_enc = [UInt32(UInt8(round(a[1] * 0xff))) << 16 + UInt32(UInt8(round(a[2] * 0xff))) << 8 + UInt32(UInt8(round(a[3] * 0xff))) for a ∈ output]

# Show as large pixels for now
const sz = 6 * 6
magnified = reshape(collect(Iterators.flatten(map(x -> x .* ones(UInt32, sz, sz), output_enc))), sz, sz * length(output_enc))
rgb(x) = reinterpret(RGB24, x)
output_rgb = rgb.(magnified)
sixel_encode(output_rgb)

# Now let's train, we have an output and a target. So let's use back propagation to figure out how our weights should be updated.

# First, let's calculate our error
actual = output_with_hidden[1][end]

# Calculate our loss or "error" function value
error = 1//2 * sum( ( target .- actual ) .^2 )

# We need to work backwards, so let's start on the last layer

# Note for linear neurons, this _should_ be the delta rule.

# We are calculating ∂E/∂yj which is defined as -(tj - yj)
# This is based entirely on the cost function, defined above
# TODO: Make the cost function swappable
# Note: j here means "all neurons in last layer", i is "first layer" and h is "input layer"
∂E∂yj = -( target .- actual )
∂E∂zj = activation_derivative.(actual) .* ∂E∂yj

# We can work this more methodically, but we have the values needed to start working
# the weights.
yi = output_with_hidden[1][2]
∆weightj=reshape([-ϵ*i*j for i in ∂E∂zj for j in yi], size(neural_network[2]))
weightj=neural_network[2] .+ ∆weightj

# Then to the next layer- hidden layer 1

∂E∂yi = [ sum(neuron .* ∂E∂zj) for neuron ∈ eachrow(neural_network[2]) ]
actuali = output_with_hidden[1][2]
∂E∂zi = activation_derivative.(actuali) .* ∂E∂yi

# Calculate weight changes
yj = output_with_hidden[1][1]
∆weighti=reshape([-ϵ*i*j for i in ∂E∂zi for j in yj], size(neural_network[1]))
weighti=neural_network[1] .+ ∆weighti

neural_network=[weighti, weightj]

[show_output(o) for o ∈ output_with_hidden]
input
sixel_encode(output_rgb)
target
actual
error

# I don't think we need this
# Then to the initial layer
# ∂E∂yh = [ sum(neuron .* ∂E∂zi) for neuron ∈ eachrow(neural_network[1]) ]
# actualh = output_with_hidden[1][1]
# ∂E∂zh = actualh .* ( ones(size(actualh)) - actualh ) .* ∂E∂yh

# neural_network_0 = neural_network
# neural_network_1 = [weighti, weightj]
# neural_network_2 = [weighti, weightj]

# julia> error
# 0.3832055164790629

# Error went up with new weights
# error = 1//2 * sum( ( target .- actual ) .^2 )
# 0.43654530839709255