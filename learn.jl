using Colors
using Sixel
using LinearAlgebra
using Base.MathConstants
using PrettyTables
using Folds
using Statistics

# Next items: 1- dropout, 2- RNN

# Activation Functions
softmax(x) = e^x
softmax_derivative(x) = x * (1 - x) # TODO: Fix this up
relu(x) = x > 0 ? x : 0
relu_derivative(x) = x > 0 ? 1 : 0
sigmoid(x) = 1 / (1 + e^-x)
sigmoid_derivative(x) = x * (1 - x)

# Cost Functions
squared_difference(target, actual) = 1 // 2 * sum((target .- actual) .^ 2)
squared_difference_derivative(target, actual) = -(target .- actual)
cross_entropy(target, actual) = -sum(target .* log.(actual))
cross_entropy_derivative(target, actual) = actual .- target

activation_fns = Dict(
  "sigmoid" => (sigmoid, sigmoid_derivative),
  "relu" => (relu, relu_derivative),
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

function get_opt(opts, key, default)
  if opts !== nothing && key in fieldnames(typeof(opts))
    opts[key]
  else
    default
  end
end

has_opt(opts, key) = opts !== nothing && key in fieldnames(typeof(opts))

# TODO: Not always rand weights
function initialize_layer((layers, layer_num, prev_sz, network_config), layer_config)
  (layer_sz, weights, scaled) = if layer_config.type == "softmax"
    (prev_sz, nothing, true)
  else
    sz = layer_config.nodes
    if has_opt(network_config, :weights)
      (sz, network_config.weights[layer_num], false)
    else
      (sz, rand(prev_sz, sz), false)
    end
  end

  activation_fn = layer_config.type # currently 1:1
  (activation, activation_derivative) = fetch!(activation_fns, activation_fn, "unknown activation function: $activation_fn")

  apply_weight_constraints = if :constraints in fieldnames(typeof(layer_config))
    function (∂E∂wijs)
      for ((x1, y1), (x2, y2)) in layer_config.constraints
        ∂E∂wijs[x1, y1] = ∂E∂wijs[x2, y2] =
          ((∂E∂wijs[x1, y1] + ∂E∂wijs[x2, y2]) / 2)
      end
      ∂E∂wijs
    end
  else
    x -> x
  end

  layer = (
    config=layer_config,
    weights=apply_weight_constraints(weights),
    activation=activation,
    activation_derivative=activation_derivative,
    scaled=scaled,
    apply_weight_constraints=apply_weight_constraints
  )

  ([layers; [layer]], layer_num + 1, layer_sz, network_config)
end

function initialize_network(input_sz, layer_configs; network_config=nothing)
  (layers, _) = Folds.reduce(initialize_layer, layer_configs, init=([], 1, input_sz, network_config))

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

function build_nn(; network_layers, embedding, training, show_fn=show_fn, cost_fn="squared-difference", ϵ=0.01, network_config=nothing)
  apply_embedding(k) = embedding[k]

  input_sz = length(apply_embedding(rand(training)[begin]))
  nn = initialize_network(input_sz, network_layers, network_config=network_config)
  (cost, cost_derivative) = fetch!(cost_fns, cost_fn, "unknown cost function: $cost_fn")

  grammar = keys(embedding)

  inverse_embedding = Dict([(v, k) for (k, v) ∈ pairs(embedding)])

  function show_output(output)
    max_length = maximum(length.(output))
    padded_vecs = [vcat(vec, fill(NaN, max_length - length(vec))) for vec in reverse(output)]
    mat = hcat(padded_vecs...)'
    pretty_table(mat, header=repeat([""], max_length))
  end

  function get_case(case)
    fetch!(training, case, "unknown training case: $case")
  end

  function train(nn; case=nothing, batch=nothing, debug=false)
    inputs = if batch == nothing
      if case == nothing
        [rand(training), rand(training)]
      else
        [(case, get_case(case))]
      end
    else
      map(case -> (case, get_case(case)), batch)
    end

    function calculate_next_layer_with_hidden(values, layer)
      [values; [apply_layer(layer, values[end])]]
    end

    function calculate_value_with_hidden(input)
      Folds.reduce(calculate_next_layer_with_hidden, nn; init=[input])
    end

    function derive_gradients((input, target))
      embedded_input = apply_embedding(input)

      output_with_hidden = calculate_value_with_hidden(embedded_input)

      if debug
        # TODO: Do we want to show our output in training mode?
        show_output(output_with_hidden)
      end

      # Now let's train, we have an output and a target. So let's use back propagation to figure out how our weights should be updated.

      # First, let's calculate our overall error
      actual = output_with_hidden[end]
      error = cost(target, actual)
      # ∂E∂yifinal = 
      # ∂E∂zifinal = nn[end].activation_derivative.(actual) .* ∂E∂yifinal

      function backpropagate((∂E∂yjs, ∂E∂zjs, ∂E∂wijs, prev_layer), ((j, layer), yj, yi))
        # This reduction works backwards through the outputs of the
        # neural net. Each step computes the weight changes for that
        # layer and then propagates backwards. The result is several
        # named vectors (∂E∂yj=...,∂E∂zj=...,∆weightj=...,weightj=...)

        # Note: i refers to current layer, y to next layer up
        # Note: Due to backpropagation, y was calculated before i

        if debug
          display("j=$j")
          display("layer=$layer")
          display("prev_layer=$prev_layer")
          display("∂E∂yjs=$∂E∂yjs")
          display("∂E∂zjs=$∂E∂zjs")
          display("∂E∂wijs=$∂E∂wijs")
          display("yj=$yj")
          display("yi=$yi")
        end

        ∂E∂yj = if prev_layer === nothing
          # Output layer's gradient is defined by the cost function
          cost_derivative(target, yj)
        else
          # Other layer's are defined by backprop from the previous layer
          ∂E∂zk = ∂E∂zjs[end]

          # Calculate ∂E∂yj, which will be `∂E∂yk` for the next iteration
          if prev_layer.config.type == "softmax"
            # I need to mull this, but I believe since there are no weights
            # we just pass this through directly?
            # Note: this is really ∂zj∂yi
            ∂E∂zk
          else
            # This is the "caching" part of the back propagating algorithm
            # that uses previously calculated values
            # There might be room to make this comprehension even
            # a little faster.
            [dot(neuron, ∂E∂zk) for neuron ∈ eachrow(prev_layer.weights)]
          end
        end

        # display("∂E∂yj=$∂E∂yj")

        # For each of these 4
        ∂E∂zj = layer.activation_derivative.(yj) .* ∂E∂yj

        # display("∂E∂zj=$∂E∂zj")

        ∂E∂wij = if layer.config.type == "softmax"
          nothing
        else
          yi * ∂E∂zj'
        end

        # display("size(∂E∂wij)=$(size(∂E∂wij))")

        ([∂E∂yjs; [∂E∂yj]], [∂E∂zjs; [∂E∂zj]], [∂E∂wijs; [∂E∂wij]], layer)
      end

      layer_values = zip(enumerate(nn), output_with_hidden[2:end], output_with_hidden[1:end-1])

      # display("layer_values=$([layer_values...])")

      (∂E∂yis_rev, ∂E∂zis_rev, ∂E∂wijs_rev) = Folds.reduce(backpropagate, Iterators.reverse(layer_values); init=([], [], [], nothing))
      ∂E∂yis = Iterators.reverse(∂E∂yis_rev)
      ∂E∂zis = Iterators.reverse(∂E∂zis_rev)
      ∂E∂wijs = Iterators.reverse(∂E∂wijs_rev)

      [∂E∂wijs...]
    end

    gradients = map(derive_gradients, inputs)

    gradient_layers = [[gradient[i] for gradient in gradients] for i in 1:length(gradients[1])]

    # display("nn=$nn")
    # display("gradient_layers=$gradient_layers")
    # display("gradient_layers_sz=$(map(x -> size(x[1]), gradient_layers))")

    function reweight((layer, ∂E∂wijs))
      if layer.weights == nothing
        layer
      else
        # display("weights=$(layer.weights),∂E∂wijs=$(mean(∂E∂wijs))")
        ∆weight = -ϵ .* layer.apply_weight_constraints(mean(∂E∂wijs))
        weights_adj = layer.weights .+ ∆weight

        (; layer..., weights=weights_adj)
      end
    end

    # display("∂E∂wijs=$([∂E∂wijs...])")

    [map(reweight, zip(nn, gradient_layers))...]
  end

  function try_check_error((input, actual))
    if haskey(training, input)
      cost(get(training, input, nothing), actual)
    else
      nothing
    end
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

    mean(filter(x -> x !== nothing, map(try_check_error, zip(input, output))))
  end

  (nn, train, infer)
end
