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
cross_entropy(target, actual) = -sum( target .* log.(actual) )
cross_entropy_derivative(target, actual) = actual .- target

activation_fns = Dict(
  "sigmoid" => (sigmoid, sigmoid_derivative),
  "softmax" => (softmax, softmax_derivative)
)

cost_fns = Dict(
  "squared-difference" => (squared_difference, squared_difference_derivative),
  "cross-entropy" => (cross_entropy, cross_entropy_derivative),
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

function build_nn(;network_layers, embedding, training, show_fn=show_fn, cost_fn="squared-difference", ϵ=0.01)
  apply_embedding(k) = embedding[k]

  input_sz = length(apply_embedding(rand(training)[begin]))
  nn = initialize_network(input_sz, network_layers)
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
    actual = output_with_hidden[end]
    error = cost(target, actual)
    ∂E∂yifinal=cost_derivative(target, actual)
    ∂E∂zifinal = nn[end].activation_derivative.(actual) .* ∂E∂yifinal

    function derive_gradients((∂E∂yis, ∂E∂zis, ∂E∂wijs), (layer, yprev))
      # This reduction works backwards through the outputs of the
      # neural net. Each step computes the weight changes for that
      # layer and then propagates backwards. The result is several
      # named vectors (∂E∂yj=...,∂E∂zj=...,∆weightj=...,weightj=...)
      
      # Note: i refers to current layer, y to next layer up
      # Note: Due to backpropagation, y was calculated before i

      if debug
        display("layer=$layer")
        display("∂E∂yis=$∂E∂yis")
        display("∂E∂zis=$∂E∂zis")
        display("∂E∂wijs=$∂E∂wijs")
        display("yprev=$yprev")
      end

      ∂E∂zj=∂E∂zis[end]

      ∂E∂wij = if layer.config[1] == "softmax"
        nothing
      else
        reshape([zj*y for zj in ∂E∂zj for y in yprev], size(layer.weights))
      end

      # Calculate ∂E∂yi, which will be `∂E∂yj` for the next iteration
      # 4 in length
      ∂E∂yi = if layer.config[1] == "softmax" 
        # I need to mull this, but I believe since there are no weights
        # we just pass this through directly?
        # Note: this is really ∂zj∂yi
        ∂E∂zj
      else
        # This is the "caching" part of the back propagating algorithm
        # that uses previously calculated values
        [ sum(neuron .* ∂E∂zj) for neuron ∈ eachrow(layer.weights) ]
      end

      # display("∂E∂yi=$∂E∂yi")

      # For each of these 4
      ∂E∂zi = layer.activation_derivative.(yprev) .* ∂E∂yi

      ([∂E∂yis; [∂E∂yi]], [∂E∂zis; [∂E∂zi]], [∂E∂wijs; [∂E∂wij]])
    end

    layer_prev_values = zip(nn, output_with_hidden[1:end-1])

    (∂E∂yis_rev, ∂E∂zis_rev, ∂E∂wijs_rev) = Folds.reduce(derive_gradients, Iterators.reverse(layer_prev_values); init=([∂E∂yifinal], [∂E∂zifinal], [nothing]))
    ∂E∂yis=Iterators.reverse(∂E∂yis_rev)
    ∂E∂zis=Iterators.reverse(∂E∂zis_rev)
    ∂E∂wijs=Iterators.reverse(∂E∂wijs_rev)

    function reweight((layer, ∂E∂wij))
      if ∂E∂wij == nothing
        layer
      else
        ∆weight=-ϵ .* ∂E∂wij
        weights_adj=layer.weights .+ ∆weight

        (; layer..., weights=weights_adj)
      end
    end

    # display("∂E∂wijs=$([∂E∂wijs...])")

    [map(reweight, zip(nn, ∂E∂wijs))...]
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
