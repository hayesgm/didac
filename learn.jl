using Colors
using Sixel
using LinearAlgebra
using Base.MathConstants
using PrettyTables
using Folds

softmax(x)=e^x
softmax_derivative(x) = x * ( 1 - x ) # TODO: Fix this up
sigmoid(x) = 1 / (1 + e^-x)
sigmoid_derivative(x) = x * ( 1 - x )
squared_difference(target, actual) = 1//2 * sum( ( target .- actual ) .^2 )
squared_difference_derivative(target, actual) = -( target .- actual )

activation_fns = Dict(
  "sigmoid" => (sigmoid, sigmoid_derivative),
  "softmax" => (softmax, softmax_derivative)
)

function fetch!(dict, key, err="key not found")
  if haskey(dict, key)
    dict[key]
  else
    error(err)
  end
end

# TODO: Not always rand weights
function initialize_layer((layers, prev_sz), layer_config)
  activation_fn = layer_config[1]

  (layer_sz, weights, scaled) = if activation_fn == "softmax"
    (prev_sz, nothing, true)
  else
    sz = layer_config[2]

    (sz, rand(prev_sz, sz), false)
  end

  (activation, activation_derivative) = fetch!(activation_fns, activation_fn, "unknown activation function: $activation_fn")

  layer = (config=layer_config, weights=weights, activation=activation, activation_derivative=activation_derivative, scaled=scaled)

  ([layers; [layer]], layer_sz)
end

function initialize_network(input_sz, layer_configs)
  (layers, _) = Folds.reduce(initialize_layer, layer_configs, init=([], input_sz))

  layers
end

cost_fns = Dict(
  "squared-difference" => (squared_difference, squared_difference_derivative)
)

function show_network(nn)
  max_length = maximum(length.(nn))
  padded_vecs = [vcat(vec, fill(NaN, max_length - length(vec))) for vec in reverse(nn)]
  mat = hcat(padded_vecs...)'
  pretty_table(mat, header=repeat(["Column"], max_length))
end

function apply_layer(layer, prev_values)
  values = if layer.weights !== nothing
    [layer.activation.(dot(prev_values, w)) for w ∈ eachcol(layer.weights)]
  else
    layer.activation.(prev_values)
  end

  if layer.scaled
    values / sum(values)
  else
    values
  end
end

function build_nn(;network_layers, embedding, training, show_fn=show_fn, activation_fn="sigmoid", cost_fn="squared-difference", ϵ=0.01)
  apply_embedding(k) = embedding[k]

  input_sz = length(apply_embedding(rand(training)[begin]))
  nn = initialize_network(input_sz, network_layers)
  (activation, activation_derivative) = fetch!(activation_fns, activation_fn, "unknown activation function: $activation_fn")
  (cost, cost_derivative) = fetch!(cost_fns, cost_fn, "unknown cost function: $cost_fn")

  grammar = keys(embedding)

  inverse_embedding = Dict([(v, k) for (k, v) ∈ pairs(embedding)])

  function show_output(output)
    max_length = maximum(length.(output))
    padded_vecs = [vcat(vec, fill(NaN, max_length - length(vec))) for vec in reverse(output)]
    mat = hcat(padded_vecs...)'
    pretty_table(mat, header=repeat([""], max_length))
  end

  function train(nn; case=nothing, debug=false)
    (input, target) = if case == nothing
      rand(training)
    else
      (case, fetch!(training, case, "unknown training case: $case"))
    end

    embedded_input = apply_embedding(input)

    function calculate_next_layer_with_hidden(values, layer)
      [values; [apply_layer(layer, values[end])]]
    end

    function calculate_value_with_hidden(input)
      Folds.reduce(calculate_next_layer_with_hidden, nn; init=[input])
    end

    output_with_hidden = calculate_value_with_hidden(embedded_input)

    if debug
      # TODO: Do we want to show our output in training mode?
      show_output(output_with_hidden)
    end

    # Now let's train, we have an output and a target. So let's use back propagation to figure out how our weights should be updated.

    # First, let's calculate our overall error
    error = cost(target, output_with_hidden[end])
    ∂E∂yend=cost_derivative(target, output_with_hidden[end])

    function backpropagate(acc, (j, (layer, (yi, yj))))
      # This reduction works backwards through the outputs of the
      # neural net. Each step computes the weight changes for that
      # layer and then propagates backwards. The result is several
      # named vectors (∂E∂yj=...,∂E∂zj=...,∆weightj=...,weightj=...)
      
      # Note: y refers to current layer, i to previous layer

      if debug
        display("acc=$acc")
        display("j=$j")
        display("yj=$yj")
        display("yi=$yi")
        display("layer=$layer")
      end

      # By definition, ∂E∂yj = ∂E∂yi from the previous layer
      ∂E∂yj = acc[end].∂E∂yi
      ∂E∂zj = activation_derivative.(yj) .* ∂E∂yj

      # Calculate ∂E∂yi, which will be `∂E∂yj` for the next iteration
      (∂E∂yi, layer_adj) = if layer.config[1] == "softmax" 
        # I need to mull this, but I believe since there are no weights
        # we just pass this through directly?
        (∂E∂zj, layer)
      else
        # This is the "caching" part of the back propagating algorithm
        # that uses previously calculated values
        ∂E∂yi = [ sum(neuron .* ∂E∂zj) for neuron ∈ eachrow(layer.weights) ]
        ∆weightj=reshape([-ϵ*i*j for i in ∂E∂zj for j in yi], size(layer.weights))
        weightj_adj=layer.weights .+ ∆weightj
        layer_adj=(; layer..., weights=weightj_adj)

        (∂E∂yi, layer_adj)
      end

      [acc; (∂E∂yi=∂E∂yi,∂E∂zj=∂E∂zj,layer_adj=layer_adj)]
    end

    output_pairs = zip(nn, zip(output_with_hidden[1:end-1], output_with_hidden[2:end]))

    res = Folds.reduce(backpropagate, Iterators.reverse(enumerate(output_pairs)); init=[(∂E∂yi=∂E∂yend,)])

    map(res -> res.layer_adj, reverse(res[2:end]))
  end

  function infer(nn, input)
    embedded_input = apply_embedding.(input)

    # Just do these one at a time until we parallelize
    function calculate_next_layer(values, layer)
      apply_layer(layer, values)
    end

    function calculate_value(input)
      Folds.reduce(calculate_next_layer, nn; init=input)
    end

    output = [calculate_value(i) for i ∈ embedded_input]

    show_fn(output)
  end

  (nn, train, infer)
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

(nn, train, infer) = build_nn(
  network_layers=[
    ("sigmoid", 4),
    ("sigmoid", 3),
    ("softmax", 3)
  ],
  embedding=embedding,
  training=training,
  show_fn=show_sixel,
  ϵ=0.05
)

for i ∈ 1:10
  for j ∈ 1:20000
    global nn = train(nn)
  end
  
  infer(nn, ["red", "green", "blue", "deep-red"])
end

show(nn)
