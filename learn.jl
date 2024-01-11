using Colors
using Sixel
using LinearAlgebra
using Base.MathConstants

# [  0.111 0.121 0.131 0.141
#      0.112 0.122 0.132 0.142
#      0.113 0.123 0.133 0.143
#      0.114 0.124 0.134 0.144
#   ],
#    # N21   N22   N23
#   [  0.211 0.221 0.231
#      0.212 0.222 0.232
#      0.213 0.223 0.233
#      0.214 0.224 0.234
#   ]
  
## A simple neural-network which accepts embedded input

### Let's create two layers (1 hidden, 1 output)
### We'll initialize to random values between zero and one
neural_network = [
  # [ 1 5  9 13
  #   2 6 10 14
  #   3 7 11 15
  #   4 8 12 16
  # ],
  # [ 11 55 99
  #   22 66 11
  #   33 77 22 
  #   44 88 33
  # ]
  rand(4,4),
  rand(4,3)
]

activation(x) = 1 / (1 + e^-x)

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

# Note for word2vec, we have both embedding and context matrices.
input = ["red", "red", "green"]

embedded_input = embedding.(input)

# Just do these one at a time until we parallelize

function calculate_next_layer(layer, weights)
  [activation.(dot(layer, weights[:,n])) for n ∈ 1:size(weights)[2]]
end

function calculate_value(layer0)
  Folds.reduce(calculate_next_layer, neural_network; init=layer0)
end

output = [calculate_value(i) for i ∈ embedded_input]

# layer0 = embedded_input[3]
# layer1 = [activation.(dot(layer0, neural_network[1][:,n])) for n ∈ 1:size(neural_network[1])[2]]
# layer2 = [activation.(dot(layer1, neural_network[2][:,n])) for n ∈ 1:size(neural_network[2])[2]]

# TODO: We probably don't want to use a decoder at all
output_enc = [UInt32(UInt8(round(a[1] * 0xff))) << 16 + UInt32(UInt8(round(a[2] * 0xff))) << 8 + UInt32(UInt8(round(a[3] * 0xff))) for a ∈ output]

# Show as large pixels for now
const sz = 6 * 6
magnified = reshape(collect(Iterators.flatten(map(x -> x .* ones(UInt32, sz, sz), output_enc))), sz, sz * length(output_enc))
rgb(x) = reinterpret(RGB24, x)
output_rgb = rgb.(magnified)
sixel_encode(output_rgb)
