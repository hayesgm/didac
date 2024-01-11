using Colors
using Sixel
using LinearAlgebra
using Base.MathConstants
using PrettyTables
using Folds

function initialize_network(input_sz, output_sz, hidden_layers)

  network_shape = [input_sz; hidden_layers; output_sz]
  
  map(rand(x,y) for (x, y) ∈ zip(network_shape[1:end-1], network_shape[2:end]))
end

sigmoid(x) = 1 / (1 + e^-x)
sigmoid_derivative(x) = x * ( 1 - x )
squared_difference(target, actual) = 1//2 * sum( ( target .- actual ) .^2 )
squared_difference_derivative(target, actual) = -( target .- actual )

activation_fns = Dict(
  "sigmoid" => (sigmoid, sigmoid_derivative)
)

cost_fns = Dict(
  "squared-difference" => (squared_difference, squared_difference_derivative)
)

function show_network(neural_network)
  max_length = maximum(length.(neural_network))
  padded_vecs = [vcat(vec, fill(NaN, max_length - length(vec))) for vec in reverse(neural_network)]
  mat = hcat(padded_vecs...)'
  pretty_table(mat, header=repeat(["Column"], max_length))
end

function network(;network_shape, embedding, training, debug=false, show_fn=show_fn, activation_fn="sigmoid", cost_fn="squared-difference", ϵ=0.01)
  neural_network = initialize_network(network_shape)
  (activation, activation_derivative) = get(activation_fns, activation_fn, () -> error("unknown activation function: $(activation_fn)"))
  (cost, cost_derivative) = get(cost_fns, cost_fn, () -> error("unknown cost function: $(cost_fn)"))

  grammar = keys(embedding)

  apply_embedding(k) = embedding[k]
  inverse_embedding = Dict([(v, k) for (k, v) ∈ pairs(embedding)])

  function show_output(output)
    max_length = maximum(length.(output))
    padded_vecs = [vcat(vec, fill(NaN, max_length - length(vec))) for vec in reverse(output)]
    mat = hcat(padded_vecs...)'
    pretty_table(mat, header=repeat(["Column"], max_length))
  end

  function train(neural_network; case=nothing)
    (input, target) = if case == nothing
      rand(training)
    else
      (case, get(training, case, () -> error("unknown training case: $case")))
    end

    embedded_input = apply_embedding(input)

    function calculate_next_layer_with_hidden(layers, weights)
      [layers; [[activation.(dot(layers[end], w)) for w ∈ eachcol(weights)]]]
    end
     
    function calculate_value_with_hidden(layer0)
      Folds.reduce(calculate_next_layer_with_hidden, neural_network; init=[layer0])
    end

    output_with_hidden = calculate_value_with_hidden(embedded_input)

    if debug
      # TODO: Do we want to show our output in training mode?
      show_output(output_with_hidden)
    end

    # Now let's train, we have an output and a target. So let's use back propagation to figure out how our weights should be updated.

    # First, let's calculate our overall error
    error = cost(target, output_with_hidden[end])

    function backpropagate(acc, (j, (yi, yj)))
      # This reduction works backwards through the outputs of the
      # neural net. Each step computes the weight changes for that
      # layer and then propagates backwards. The result is several
      # named vectors (∂E∂yj=...,∂E∂zj=...,∆weightj=...,weightj=...)
      
      # Note: y refers to current layer, i to previous layer
      # Note: `i` could technically be the input layer

      # We are calculating ∂E/∂yj which is defined as -(tj - yj)
      # TODO: I believe this is only for squared-difference cost function
      # TODO: Maybe zip?
      weightsj = neural_network[j]

      if false
        show("acc=$acc")
        show("j=$j")
        show("yj=$yj")
        show("yi=$yi")
        show("weightsj=$weightsj")
      end
      # We need to peek at the next row of output to get `yi`

      # TODO: Is this accurate only for the first layer?
      # Again, maybe we can zip to combine data?

      ∂E∂yj = 
        if size(acc) == (0,)
          # Base case, output layer
          # TODO: Toy with this assumption
          cost_derivative(target, yj)
        else
          weightsk = neural_network[j+1]

          # This is the "caching" part of the back propagating algorithm
          # that uses previously calculated values
          [ sum(neuron .* acc[end].∂E∂zj) for neuron ∈ eachrow(weightsk) ]
        end

      ∂E∂zj = activation_derivative.(yj) .* ∂E∂yj      

      ∆weightj=reshape([-ϵ*i*j for i in ∂E∂zj for j in yi], size(weightsj))

      weightj_adj=weightsj .+ ∆weightj

      [acc; (∂E∂yj=∂E∂yj,∂E∂zj=∂E∂zj,∆weightj=∆weightj,weightj=weightj_adj)]
    end

    output_pairs = zip(output_with_hidden[1:end-1], output_with_hidden[2:end])

    res = Folds.reduce(backpropagate, Iterators.reverse(enumerate(output_pairs)); init=[])

    map(res -> res.weightj, reverse(res))
  end

  function infer(neural_network, input)
    embedded_input = apply_embedding.(input)

    # Just do these one at a time until we parallelize
    function calculate_next_layer(layer, weights)
      [activation.(dot(layer, w)) for w ∈ eachcol(weights)]
    end

    function calculate_value(layer0)
      Folds.reduce(calculate_next_layer, neural_network; init=layer0)
    end

    output = [calculate_value(i) for i ∈ embedded_input]

    show_fn(output)
  end

  (neural_network, train, infer)
end

# This could probably be made using an identity matrix and an index paramter
embedding = Dict(
  "red" =>      [1, 0, 0, 0],
  "green" =>    [0, 1, 0, 0],
  "blue" =>     [0, 0, 1, 0],
  "deep-red" => [0, 0, 0, 1]
)

## This section is about training

# So first, we need to build some tuples to learn towards
# It's pretty funny since there aren't a lot of different examples
# Available given our grammer, but it doesn't mean we can't ain't learn.
# Also, since our grammar is so limited, we'll probably end up including
# all of our training cases in our results. But that's okay, this is all
# a toy at this phase anyway.
training = Dict(
  "red" => [0.9, 0.2, 0.2],
  "green" => [0.2, 0.9, 0.2],
  "blue" => [0.2, 0.2, 0.9],
  "deep-red" =>  [0.8, 0.1, 0.1]
)

function show_sixel(output)
  # TODO: Do we want to do this all at once or singularly?

  # TODO: We probably don't want to use a decoder at all
  output_enc = [UInt32(UInt8(round(a[1] * 0xff))) << 16 + UInt32(UInt8(round(a[2] * 0xff))) << 8 + UInt32(UInt8(round(a[3] * 0xff))) for a ∈ output]

  # Show as large pixels for now
  sz = 6 * 6
  magnified = reshape(collect(Iterators.flatten(map(x -> x .* ones(UInt32, sz, sz), output_enc))), sz, sz * length(output_enc))
  rgb(x) = reinterpret(RGB24, x)
  output_rgb = rgb.(magnified)
  sixel_encode(output_rgb)
end

(nn, train, infer) = network(network_shape=(3,4), embedding=embedding, training=training, show_fn=show_sixel, ϵ=0.05)

for i ∈ 1:10
  for j ∈ 1:20000
    global nn = train(nn)
  end
  
  infer(nn, ["red", "green", "blue", "deep-red"])
end

show(nn)

##

# [show_output(o) for o ∈ output_with_hidden]
# input
# sixel_encode(output_rgb)
# target
# actual
# error

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