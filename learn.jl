using Colors
using Sixel
using LinearAlgebra
using Base.MathConstants
using PrettyTables
using Folds
using Statistics

### TODO Items
# 1. Add Dropout
# 2. Add RNN

### Activation Functions
#
# Activation functions are used to map the weighted
# inputs of a neuron to its own output value, which
# is then passed to the next neurons in the the net.
#
# We include their derivitive functions here, as well,
# which is used to gather gradients during the
# training process.
#
# Common activation functions include:
#
# ## The Sigmoid Function
#
# Defined as `1 / (1 + e^-x)`, the sigmoid function
# has a nice value of neatly mapping values onto a
# scale of [-1,1] with a smooth derivative.
#
# The derivative is `x(1 - x)`.
# 
# ## The Rectified Linear Unit (ReLU)
#
# Defined as `max[0, x]`, this unit is zero
# unless x is greater than zero, and x otherwise.
#
# The derivative is 0 if x ≤ 0, and x otherwise.
#
# ## The Softmax Function
#
# The softmax function is `e^x / ∑e^x` for all units
# in the current layer. This has the nice property
# of summing to 1.0, which makes it appropriate to use
# for the weights of probabilistic outputs.
#
# The derivative is `x(1-x)` # TODO: Verify this
#
# Note: the equations below assume the scaling term,
#       which is not directly included in the equations.
#
sigmoid(x) = 1 / (1 + e^-x)
sigmoid_derivative(x) = x * (1 - x)
relu(x) = x > 0 ? x : 0
relu_derivative(x) = x > 0 ? 1 : 0
softmax(x) = e^x
softmax_derivative(x) = x * (1 - x) # TODO: Fix this up

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

function initialize_layer((layers, layer_num, prev_sz, network_config), layer_config)
  # In a RNN, we need to track a previous activation
  # we're going to track it with the RNN. When we build
  # the RNN, we need to build a base activation. For now
  # that's going to be random.

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

  tag = get_opt(layer_config, :tag, "layer-$layer_num")

  initial_activation = if get_opt(layer_config, :feedback, false)
    activation_fn.(rand(prev_sz, sz))
  else
    nothing
  end

  recurrent_weights = if get_opt(layer_config, :recurrent, false)
    rand(size(weights))
  else
    nothing
  end

  layer = (
    tag=tag,
    initial_activation=initial_activation,
    config=layer_config,
    weights=apply_weight_constraints(weights),
    recurrent_weights=recurrent_weights,
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

function apply_layer(layer, input_values, recurrent_values=nothing)
  values = if layer.weights !== nothing
    if true
      [layer.activation.(dot(input_values, w)) for w ∈ eachcol(layer.weights)]
    else
      weighted_inputs = [dot(input_values, w) for w ∈ eachcol(layer.weights)]
      weighted_recurrents = [dot(recurrent_values, w) for w ∈ eachcol(layer.recurrent_weights)]

      layer.activation.(weighted_inputs .+ weighted_recurrents)
    end
  else
    layer.activation.(input_values)
  end

  if layer.scaled
    values / sum(values)
  else
    values
  end
end

function build_nn(; network_layers, embedding, training, show_fn=show_fn, cost_fn="squared-difference", ϵ=0.01, network_config=nothing, recurrent=false)
  # Embedding maps the user-defined input (grammar) to a vector space.
  # This function is a helper so we can broadcast over inputs,
  # e.g. `apply_embedding.(["a", "b"]) = [[1, 0, 0], [0, 1, 0]]`
  apply_embedding(k) = embedding[k]

  # We want to determine the size of the input embedding. The current
  # strategy here is to grab the first one from the training set,
  # since each one should be the same. There's probably a better
  # way to do this.
  input_sz = length(apply_embedding(rand(training)[begin]))

  # We initialize the neural network. The neural network is an array
  # of layers in the network. This includes loading weights for each
  # layer, etc.
  nn = initialize_network(input_sz, network_layers, network_config=network_config)

  # Grab the cost function and make sure it exists. See `Cost Functions` above.
  (cost, cost_derivative) = fetch!(cost_fns, cost_fn, "unknown cost function: $cost_fn")

  # The grammar is simply the pre-embedded inputs, e.g. "a", "b"
  grammar = keys(embedding)

  # If for any reason we want to take an embedded input and convert
  # it back to the grammar. This isn't currently used.
  inverse_embedding = Dict([(v, k) for (k, v) ∈ pairs(embedding)])

  # This is a quick and dirty function to show the activations of
  # each neuron in the net for debugging. This should probably be
  # improved.
  function show_output(output)
    max_length = maximum(length.(output))
    padded_vecs = [vcat(vec, fill(NaN, max_length - length(vec))) for vec in reverse(output)]
    mat = hcat(padded_vecs...)'
    pretty_table(mat, header=repeat([""], max_length))
  end

  # Helper function to pull a specific training case from our
  # training dictionary.
  get_case(case) = fetch!(training, case, "unknown training case: $case")

  ### Feedforward Calculations
  #
  # The following functions are used to calculate
  # the output of the network in a feed-forward approach.
  # We separate these into multiple functions, since
  # calculating the output of a recurrent net is slightly
  # different than calculating the output of a simple
  # feed-forward network.
  #
  # As each layer depends on the activation of the previous
  # layer, we use a reduction to pass the calculation to
  # the next layer in the net. For inference, we track
  # the activation of each layer since we'll need it for
  # gradient descent. For recurrent nets, we only track
  # the value of layers which are marked recurrent, meaning
  # they are allowed to be passed to earlier layers in the
  # net.
  #
  # Note: These splits may end up being a premature
  #       optimization and maybe it would make more sense to
  #       simply always track the output of each layer, even
  #       during inference.

  # This function will activate a single next layer
  # in the foward pass of the neural net based on
  # the activation of the previous layer.
  #
  # Note: for a recurrent neural network, we will
  #       store the context (activation) of any
  #       feedback layers in the net for the next
  #       input's pass on the net.
  function calculate_next_layer(values, layer)
    apply_layer(layer, values)
  end

  # This function walks the neural net, activating
  # one layer at a time until it reaches the output.
  #
  # Note: for a recurrent neural network, this function
  #       will store all contexts (that is, the output
  #       of a hidden layer that is the input to another
  #       layer) in an accumulator for the next step
  #       in the input array.
  function calculate_value(nn, input)
    Folds.reduce(calculate_next_layer, nn; init=input)
  end

  # Inner loop of reduction. Simple forward pass on the
  # neural net during the inference part of the training.
  #
  # Note: for training, we collect the activation of each
  #       layer in the neural net, as they are necessary
  #       for determining the gradient for learning.
  function calculate_next_layer_with_hidden(values, layer)
    [values; [apply_layer(layer, values[end])]]
  end

  # Calculates the forward pass on the neural net during
  # the inference part of training. Returns an array with
  # the activations in each layer of the net.
  function calculate_value_with_hidden(nn, input)
    Folds.reduce(calculate_next_layer_with_hidden, nn; init=[input])
  end

  # Calculate a step in the forward pass, passing around a
  # context value, which is the activation on recurrent layers.
  function calculate_next_layer_with_context((values, context), layer)
    values = if layer.recurrent
      recurrent_values = fetch!(context, layer.recurrent, "missing recurrent layer $(layer.recurrent) in context")
      next_values = apply_layer(layer, values, recurrent_values)
    else
      apply_layer(layer, values)
    end

    if layer.feedback
      # TODO: Is it good to do this by tag?
      (next_values, merge(context, Dict(layer.tag => next_values)))
    else
      (next_values, context)
    end
  end

  # Calculates the forward pass on the neural net given
  # a context, which is the activation on recurrent layers.
  # We return both the output and a new context that can
  # be used in the next iteration.
  function calculate_value_with_context(nn, input, context)
    Folds.reduce(calculate_next_layer_with_context, nn; init=(input, context))
  end

  # The `train` function can be used to run a single training case
  # or a batch of training cases. This function will return an
  # updated neural net having undergone a single step of
  # stochastic gradient decent. Note: for a batch of training
  # cases, we'll still only take one step in the average direction
  # from the batch of cases. If no case or batch is provided,
  # this function will grab a random batch of two training cases
  # from the `training` dictionary.
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

    function derive_gradients((input, target))
      embedded_input = apply_embedding(input)

      output_with_hidden = calculate_value_with_hidden(nn, embedded_input)

      if debug
        show_output(output_with_hidden)
      end

      # Now let's train, we have an output and a target. So let's use back propagation to figure out how our weights should be updated.

      # First, let's calculate our overall error
      actual = output_with_hidden[end]
      error = cost(target, actual)

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
        # display("layer=$(layer.tag), weights=$(layer.weights),∂E∂wijs=$(mean(∂E∂wijs))")
        ∆weight = -ϵ .* layer.apply_weight_constraints(mean(∂E∂wijs))
        weights_adj = layer.weights .+ ∆weight
        # display("weights_adj=$weights_adj")

        (; layer..., weights=weights_adj)
      end
    end

    # display("∂E∂wijs=$([∂E∂wijs...])")

    [map(reweight, zip(nn, gradient_layers))...]
  end

  # When we're inferencing, it's nice to output our error,
  # though we only can calculate it if we have a target,
  # (this is, the input was part of our training set).
  #
  # This function returns the error if that's the case, and
  # otherwise `nothing`.
  function try_check_error((input, actual))
    if haskey(training, input)
      cost(get(training, input, nothing), actual)
    else
      nothing
    end
  end

  # The inference function calculates the output of the net
  # for a given input case. For a feed-forward neural network
  # this will run the inference for each item in the input
  # vector. For a recurrent neural network, this function will
  # take one step for each value in the input vector.
  #
  # Note: this function currently display the output, it doesn't
  #       return the value. It does return a potential error size.
  function infer(nn, input)
    embedded_input = apply_embedding.(input) # TODO: Accept single input?

    output = if recurrent
      # For recurrent neural nets, we need to feed the context
      # forward from each run of the neural net.

      # The context is the running context between inputs
      # It's only set for recurrent layers (that is, layers
      # which are allowed to feed back into other layers).
      #
      # This `initial_activation` field is currently set in the
      # `initialize_layer` function to a random activation.
      local context = Dict([
        (layer.tag, layer.initial_activation)
        for layer in nn
        if layer.initial_activation !== nothing
      ])

      # The `ys` is a collection of outputs of the net
      local ys = []

      for i ∈ embedded_inputs
        # Run a single input, storing the output from this
        # run, as well as a context to use for the next input.
        (y, context) = calculate_value_with_context(nn, i, context)

        append!(ys, y)
      end

      ys
    else
      # For feed-forward neural nets, we can calulate each output
      # in parallel.
      [calculate_value(nn, i) for i ∈ embedded_input]
    end

    show_fn(output)

    mean(filter(x -> x !== nothing, map(try_check_error, zip(input, output))))
  end

  (nn, train, infer)
end
