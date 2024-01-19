using Colors
using Sixel
using LinearAlgebra
using Base.MathConstants
using PrettyTables
using Folds
using Statistics

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

# Helper function to get a field from a named tuple or else raise
function fetch!(dict, key, err="key not found")
  if haskey(dict, key)
    dict[key]
  else
    error(err)
  end
end

# Helper function to get a field from a named tuple or else return default
function get_opt(opts, key, default)
  if opts !== nothing && key in fieldnames(typeof(opts))
    opts[key]
  else
    default
  end
end

# Helper function to check if named tuple contains given field
has_opt(opts, key) = opts !== nothing && key in fieldnames(typeof(opts))

# Helper function to split a vector at a given index
split(vec, idx) = (vec[begin:idx-1], vec[idx:end])

# Helper function to drop first n elements from an array
drop(vec, n) = vec[begin+n:end]

# Helper function to drop last n elements from an array
drop_right(vec, n) = vec[end-length(vec)+1:end-n]

# This is a quick and dirty function to show the activations of
# each neuron in the net for debugging. This should probably be
# improved.
function show_output(output)
  max_length = maximum(length.(output))
  padded_vecs = [vcat(vec, fill(NaN, max_length - length(vec))) for vec in reverse(output)]
  mat = hcat(padded_vecs...)'
  pretty_table(mat, header=repeat([""], max_length))
end

"""
    initialize_layer((layers, layer_num, prev_sz, network_config), layer_config)

Build a layer as part of a neural net. For use in reduction from `initialize_network`.
"""
function initialize_layer((layers, layer_num, prev_sz, layer_sizes, network_config), (tag, layer_config))
  # We want to figure out the layer size, since we'll need to know
  # how big the layer is if it's fully connected to the next layer.
  #
  # Additionally, we get the weights from config or randomize them.
  #
  # `scaled` is our current cheat for use in layers like softmax
  # where we need to divide by the activation of the entire layer. 
  (layer_sz, weights, scaled) = if layer_config.type == "softmax"
    (prev_sz, nothing, true)
  else
    sz = layer_config.nodes
    total_prev_sz = if has_opt(layer_config, :recurrent)
      recurrent_sz = fetch!(layer_sizes, layer_config.recurrent, "unknown recurrent layer `$(layer_config.recurrent)`")

      # For a recurrent net, we need weights from both
      # our input layer and the recurrent layer
      prev_sz + recurrent_sz
    else
      prev_sz
    end

    if has_opt(network_config, :weights)
      weights = network_config.weights[layer_num]
      if size(weights) !== (total_prev_sz, sz)
        error("invalid weights for layer $tag, expected size=$((total_prev_sz, sz)), got weight matrix of size=$(size(weights))")
      end

      # Use the weights if specified
      (sz, network_config.weights[layer_num], false)
    else
      # Otherwise randomize them
      (sz, rand(total_prev_sz, sz), false)
    end
  end

  activation_fn = layer_config.type # currently layer type is 1:1 with activation functions
  (activation, activation_derivative) = fetch!(activation_fns, activation_fn, "unknown activation function: $activation_fn")

  # A function to help us apply weight constraints after weight adjustments
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

  # For layers which can be fed back in an RNN, we need to build
  # an initial activation. Currently, we simply apply random
  # inputs to the activation function.
  initial_activation = if has_opt(layer_config, :feedback)
    activation.(rand(sz))
  else
    nothing
  end

  layer = (
    tag=tag,
    initial_activation=initial_activation,
    config=layer_config,
    weights=apply_weight_constraints(weights),
    recurrent=get_opt(layer_config, :recurrent, nothing),
    feedback=get_opt(layer_config, :feedback, false),
    activation=activation,
    activation_derivative=activation_derivative,
    scaled=scaled,
    apply_weight_constraints=apply_weight_constraints,
    skip=0
  )

  ([layers; [layer]], layer_num + 1, layer_sz, layer_sizes, network_config)
end

"""
    initialize_network(input_sz, layer_configs; network_config=nothing)

Build a neural net from layer configurations.

Given the parameters of the layers, which may include theirs weights,
builds the layers that will be part of a neural net, which itself
is simply a collection of these layers. If weights aren't specified,
they will generated randomly. We also suss out certain items like the
activation function, to make it easier to call when we're inferring
a value from the net.

Note: only a single recurrent layer is allowed at this time.

# Examples
```julia-repl
julia> initialize_network(2, [(type="sigmoid", nodes=2, tag="l1-hidden")])
42 # TODO
```
"""
function initialize_network(input_sz, layer_configs; network_config=nothing)
  # Tag each layer with a friendly name
  tagged_layer_configs = [(get_opt(layer_config, :tag, "layer-$i"), layer_config) for (i, layer_config) in enumerate(layer_configs)]

  # Tally each layer's node size in a dict keyed by tags
  layer_sizes = Dict([(tag, layer_config.nodes) for (tag, layer_config) in tagged_layer_configs])

  # Build each layer in a big reduction
  (layers, _) = reduce(initialize_layer, tagged_layer_configs, init=([], 1, input_sz, layer_sizes, network_config))

  layers
end

"""
    initial_context(nn)

Build the initial context for a (potentially) recursive neural net

Note: this is based on the `initial_activation` of the layers in the
      net, which is build in `initialize_network`.
"""
initial_context(nn) = Dict([
  (layer.tag, layer.initial_activation)
  for layer in nn
  if layer.initial_activation !== nothing
])

"""
    get_previous_layers(nn, layer)

Returns every layer in the nn that came before given layer including the layer itself.

Note: Technically, we're allowed to skip, say, skip layers here,
      but right now we might just literally return every previous layer.

TODO: Kill this fn?
"""
function get_previous_layers(nn, layer)
  layer_index = findindex(l -> l.tag == layer.tag, nn)
  nn[1:layer_index]
end

"""
    get_layer_by_tag!(nn, tag)

Gets a layer by tag or raises.
"""
function get_layer_by_tag!(nn, tag)
  layer_index = findfirst(l -> l.tag == tag, nn)
  if layer_index == nothing
    error("layer not found \"$tag\"")
  else
    nn[layer_index]
  end
end

"""
  apply_layer(layer, input_values)

Apply a next layer activation based on previous activations.

Note: for RNN, the inputs may be a mix of input nodes and feedback
      signals from recurrent nodes.
"""
function apply_layer(layer, input_values)
  values = if layer.weights !== nothing
    #display("weights=$(layer.weights), inputs=$(input_values)")
    [layer.activation.(dot(input_values, w)) for w ∈ eachcol(layer.weights)]
  else
    layer.activation.(input_values)
  end

  if layer.scaled
    values / sum(values)
  else
    values
  end
end

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
# Note: These splits may end up being a pre-mature
#       optimization and maybe it would make more sense to
#       simply always track the output of each layer, even
#       during inference.

"""
    calculate_next_layer(values, layer)

Inner function to calculate forward one pass of a feed-forward network

This function will activate a single next layer
in the foward pass of the neural net based on
the activation of the previous layer.

Note: for a recurrent neural network, we will
      store the context (activation) of any
      feedback layers in the net for the next
      input's pass on the net.
"""
function calculate_next_layer(values, layer)
  apply_layer(layer, values)
end

"""
    calculate_value(nn, input)

Calculate output value of a feed-forward network

This function walks the neural net, activating
one layer at a time until it reaches the output.

Note: for a recurrent neural network, this function
      will store all contexts (that is, the output
      of a hidden layer that is the input to another
      layer) in an accumulator for the next step
      in the input array.
"""
function calculate_value(nn, input)
  reduce(calculate_next_layer, nn; init=input)
end

"""
    calculate_next_layer_with_hidden(values, layer)

Inner function to calculate layer of a network, tracking all intermediate activations.

Inner loop of reduction. Simple forward pass on the
neural net during the inference part of the training.

Note: for training, we collect the activation of each
      layer in the neural net, as they are necessary
      for determining the gradient for learning.
"""
function calculate_next_layer_with_hidden((values, context), layer)
  (next_values, next_context) = calculate_next_layer_with_context((values[end], context), layer)
  ([values; [next_values]], next_context)
end

"""
    calculate_value_with_hidden(nn, input, context)

Calculate all network activations for given input

Calculates the forward pass on the neural net during
the inference part of training. Returns an array with
the activations in each layer of the net.
"""
function calculate_value_with_hidden(nn, input, context)
  (values, next_context) = reduce(calculate_next_layer_with_hidden, nn; init=([input], context))
  
  # Just handle weirdness of inputs. This can be smarter!
  if nn[1].recurrent !== nothing
    recurrent_values = fetch!(context, nn[1].recurrent, "missing recurrent layer `$(nn[1].recurrent)` in context")
    # display("values[1]=$(values[1]); recurrent_values=$recurrent_values")
    values[1] = [values[1]; recurrent_values]
  else
    values
  end

  (values, next_context)
end

"""
    calculate_next_layer_with_context((values, context), layer)

Calculate single step of a forward pass in a recurrent network

Calculates a step in the forward pass, passing around a
context value, which is the activation on recurrent layers.
"""
function calculate_next_layer_with_context((values, context), layer)
  input_values = if layer.recurrent !== nothing
    # display("context=$context")
    recurrent_values = fetch!(context, layer.recurrent, "missing recurrent layer `$(layer.recurrent)` in context")
    # display(values)
    # display(recurrent_values)
    [values; recurrent_values]
  else
    values
  end

  next_values = apply_layer(layer, input_values)

  if layer.feedback
    # TODO: Context is currently being indexed by tags, could we make that better/faster?
    (next_values, merge(context, Dict(layer.tag => next_values)))
  else
    (next_values, context)
  end
end

"""
    calculate_value_with_context(nn, input, context)

Calculate output value and next context from a recurrent neural network, given the input.

Calculates the forward pass on the neural net given
a context, which is the activation on recurrent layers.
We return both the output and a new context that can
be used in the next iteration.
"""
function calculate_value_with_context(nn, input, context)
  reduce(calculate_next_layer_with_context, nn; init=(input, context))
end

"""
    backpropagate((debug, cost_derivative, target, ∂E∂yjs, ∂E∂zjs, ∂E∂wijs, prev_layer), (j, (layer, yj, yi)))

Backpropagate one layer of the net in determining gradients

This reduction works backwards through the outputs of the
neural net. Each step computes the gradients for that
layer, which then propagate backwards.

For context: `k` refers to the deepest layer, `j` refers to
the current layer, and `i` refers to the next (less deep)
layer (which may be the input layer).

## Intermediate Values

We compute several important values in this function. You can
see most here: https://youtu.be/LOc_y67AzCA?si=daVtsgy9J5V4NGtP&t=634

### ∂E∂yj

First, we calculate `∂E∂yj`, which is the derivative of the
error function with respect to the output of the current (`j`)
layer's activation (`yj`) values.

For the output layer, this is simply the derivative of the
error function applied to the actual (output) values.

For other layers of the net, this is `∑(w_jk)∂E∂zk` for
all outgoing connections from `j` to the `k` layer.

This can be easily calculated, thus, using a dot product.

### ∂E∂zj

Next, we calculate `∂E∂zj`, which is the derivative of the
error function based on the inputs into layer `j`. This is
the derivative of the activation function times `∂E∂yj`, based
on the chain rule.

We use `activation_derivative`, which we stored in the layer
configuration in `initialize_network` for the derivative
function.

### ∂E∂wij

Finally, we calculate `∂E∂wij`, which is the derivative of
the error function based on the weights into layer `j`. This
is the most important value of the backpropagation function,
as these gradients will be used to change the weights during
our stochastic gradient descent step. This value is simply
`(y_i)(∂E∂zj)`. That is, the output of the neuron from layer
`i` times the `∂E∂zj` of the neuron its connected to. We can
use a big outer multiplication to calculate these all in one
step.
"""
function backpropagate((debug, cost_derivative, target, ∂E∂yjs, ∂E∂zjs, ∂E∂wijs, offsets, prev_layer), (j, (layer, yj, yi)))
  if debug
    display("j=$j")
    display("layer=$layer")
    display("prev_layer=$prev_layer")
    display("∂E∂yjs=$∂E∂yjs")
    display("∂E∂zjs=$∂E∂zjs")
    display("∂E∂wijs=$∂E∂wijs")
    display("yj=$yj")
    display("yi=$yi")
    # display("offsets=$offsets")
  end

  load_layer = length(∂E∂zjs) - layer.skip

  ∂E∂yj = if prev_layer === nothing
    # Output layer's gradient is defined by the cost function
    cost_derivative(target, yj)
  else
    # Other layer's are defined by backprop from the previous layer
    # Previous
    # TODO: We really need a diff "prev layer" here for skip layers, right?
    ∂E∂zk = ∂E∂zjs[load_layer]
    # display("prev_layer_offset=$prev_layer_offset,prev_layer_offset_end=$prev_layer_offset_end,∂E∂zk=$∂E∂zk")

    # Calculate `∂E∂yj`, which will be `∂E∂yk` for the next iteration
    if prev_layer.config.type == "softmax"
      # I need to mull this, but I believe since there are no weights
      # we just pass this through directly?
      # Note: this is really ∂zj∂yi
      ∂E∂zk
    else
      # prev_weights=eachrow(prev_layer.weights[prev_layer_offset:prev_layer_offset_end,:])
      # display("prev_weights=$prev_weights")
      # This is the "caching" part of the back propagating algorithm
      # that uses previously calculated values
      # There might be room to make this comprehension even
      # a little faster.
      [dot(neuron, ∂E∂zk) for neuron ∈ eachrow(prev_layer.weights)]
    end
  end

  # display("load_layer=$load_layer")

  # display("∂E∂yj=$∂E∂yj")

  # TODO: Think about this more
  ∂E∂yj_slim = if length(∂E∂yj) > length(yj)
    drop(∂E∂yj, length(∂E∂yj) - length(yj))
  else
    ∂E∂yj
  end

  # display("∂E∂yj_slim=$∂E∂yj_slim")

  ∂E∂zj = layer.activation_derivative.(yj) .* ∂E∂yj_slim

  # display("∂E∂zj=$∂E∂zj")

  ∂E∂wij = if layer.config.type == "softmax"
    nothing
  else
    yi * ∂E∂zj'
  end

  # display("size(∂E∂wij)=$(size(∂E∂wij))")

  (debug, cost_derivative, target, [∂E∂yjs; [∂E∂yj]], [∂E∂zjs; [∂E∂zj]], [∂E∂wijs; [∂E∂wij]], [offsets; [0]], layer)
end

"""
    derive_gradients(nn, embedded_input, target, context, cost, cost_derivative, recurrent_layers, recurrent, debug=false)

Build the gradients on the training case

Given a network, an input value, and a target (expected)
value, this function produces the gradients for each
layer of the neural network. These gradients can be used
in stochastic gradient descent to train the network.

We also return the next context, which is used for the next
run of a RNN.
"""
function derive_gradients(nn, embedded_input, target, context, cost, cost_derivative, recurrent_layers, recurrent, debug=false)
  # display("input=$input")
  # display("target=$target")
  #display("embedded_input=$embedded_input")

  # display("recurrent_layers=$recurrent_layers")

  (output_with_hidden, next_context) = calculate_value_with_hidden(nn, embedded_input, context)

  # display("output_with_hidden=$output_with_hidden")
  # display("recurrent_layers=$recurrent_layers")
  # if length(recurrent_layers) > 0
  #   # Append output from hidden layer
  #   # But note: this is super hacky
  #   output_with_hidden[1] = [output_with_hidden[1]; recurrent_layers[end][end]]
  # end
  # display("output_with_hidden_post=$output_with_hidden")
  # TODO: Think about this a bit more
  # recurrent_layers


  if debug
    show_output(output_with_hidden)
  end

  # Now let's train, we have an output and a target. So let's use back propagation to figure out how our weights should be updated.

  # First, let's calculate our overall error
  actual = output_with_hidden[end]
  error = cost(target, actual)

  # TODO: I can probably just track the output here a bit differently
  #       and make this a bit less hacky
  layer_values = [zip(nn, output_with_hidden[2:end], output_with_hidden[1:end-1])...]
  all_layer_values = vcat(recurrent_layers, layer_values)
  # display("all_layer_values=$all_layer_values")
  # display("layer_values=$([layer_values...])")
  (_, _, _, ∂E∂yis_rev, ∂E∂zis_rev, ∂E∂wijs_rev) = reduce(backpropagate, Iterators.reverse(enumerate(all_layer_values)); init=(debug, cost_derivative, target, [], [], [], [], nothing))
  ∂E∂yis = Iterators.reverse(∂E∂yis_rev)
  ∂E∂zis = Iterators.reverse(∂E∂zis_rev)
  ∂E∂wijs = Iterators.reverse(∂E∂wijs_rev)

  next_recurrent_layers = if recurrent
    # Attach any recurrent layers
    rlayers = map(nn) do layer
      if layer.recurrent !== nothing
        # We want to add this layer and then the deeper layers
        # The question is: we're getting into parallel layers
        # which means we might need to treat this more like a graph
        # but that's _also bad_ since it's more complicated.
        # We should probably attach the layers at the end, which is... ~fine
        # But we need to make sure they grab the correct previous layer values!
        recurrent_layer = get_layer_by_tag!(nn, layer.recurrent) # gets that layer
        recurrent_layer_index = findfirst(l -> l.tag == recurrent_layer.tag, nn)
        # display("recurrent_layer_index=$recurrent_layer_index")
        # display("output_with_hidden=$output_with_hidden")
        # nn_layers = map(nn) do layer
        #   (; layer..., skip=1, recurrent=false, tag="$(layer.tag)'")
        # end
        # reverse([zip(nn_layers, output_with_hidden[2:end])...][1:recurrent_layer_index])

        # TODO: Ack, how do I mark these as skip layers now?
        map(layer_values[1:recurrent_layer_index]) do (layer, yj, yi)
          (
            (; layer..., skip=0, recurrent=false, tag="$(layer.tag)'"),
            yj,
            yi
          )
        end


        # walk that recurrent layer back to the inputs
        # we'll need to also attach the outputs for it to our outputs that we'll calcuate
        # but everything should work fine if we attach everything correctly
        # which we'll... might be hard. Basically, the best thing to do would be to describe
        # how the weights connect between layers, even if we process them layer by layer,
        # we only need to make sure that we're always pre-caching the data we need, but the
        # exact order isn't that important.
        # This will also allow us to implement skip-layer connections, which is really what we're doing here!

        # Now, we might need to modify those a bit, since they are false layers
        # so a) the tags are probably off, and b) we need to attach outputs to them, since
        # otherwise we won't track that

        # We'll need to map over them to get them to be normalish, and ... this should work
        # Except for the final piece: we'll need to recombine the layers when we derive gradients

        # I could try to do this via real weight constraints, and then via tag combine things
        # This would be more versitile, but overall, it's just super hard with matrix sizes, etc
        # So for now it's probably easier to just handle it layer by layer, since averaging gradients
        # is super easy to do.
      else
        []
      end
    end

    vcat(rlayers...)
  else
    []
  end

  ([∂E∂wijs...], next_context, vcat(recurrent_layers, next_recurrent_layers))
end

"""
  build_nn(; network_layers, embedding, training, show_fn=show_fn, cost_fn="squared-difference", ϵ=0.01, network_config=nothing, recurrent=false)

Build a neural network with given parameters.

The function returns a triple `(nn, train, infer)`. `nn` is the
neural network in its initial state. `train` is a function to
train the neural net (e.g. `train(nn)`) and `infer` is a function
to run inference on the net (e.g. `infer(nn, ["red"])`). Generally,
you will use a pattern like so:

```julia-repl
(nn, train, infer) = build_nn()
nn = train(nn)
infer(nn, ["red"])
```

This pattern allows you to train the network many times to get updated
weights, and then to infer on the final network.

Note: we don't currently have any framework to store the weights
      of a network, but we'll expect to build that soon.s
"""
function build_nn(; network_layers, embedding, training, show_fn=show_fn, cost_fn="squared-difference", ϵ=0.01, network_config=nothing, recurrent=false)
  # Embedding maps the user-defined input (grammar) to a vector space.
  # This function is a helper so we can broadcast over inputs,
  # e.g. `apply_embedding.(["a", "b"]) = [[1, 0, 0], [0, 1, 0]]`
  apply_embedding(k) = embedding[k]

  # We want to determine the size of the input embedding. The current
  # strategy here is to grab the a random embedding's value.
  input_sz = length(iterate(embedding)[begin][end])

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

  # Helper function to pull a specific training case from our
  # training dictionary.
  get_case(case) = rand(filter(c -> c.first == case, training))

  """
      train(nn; case=nothing, batch=nothing, debug=false)

  Train a network with a single or batch set of cases

  The `train` function can be used to run a single training case
  or a batch of training cases. This function will return an
  updated neural net having undergone a single step of
  stochastic gradient decent. Note: for a batch of training
  cases, we'll still only take one step in the average direction
  from the batch of cases. If no case or batch is provided,
  this function will grab a random batch of two training cases
  from the `training` dictionary.
  """
  function train(nn; case=nothing, batch=nothing, debug=false)
    inputs = if recurrent
      # TODO: Handle this better?
      # display("training=$training")
      # display("case=$case")
      case = if case == nothing
        rand(training)[begin]
      else
        case
      end

      (steps, targets) = get_case(case)
      [zip(steps, targets)...]
    else
      if batch == nothing
        if case == nothing
          [rand(training), rand(training)]
        else
          [(case, get_case(case))]
        end
      else
        map(case -> (case, get_case(case)), batch)
      end
    end

    # display("inputs=$inputs $(length(inputs))")

    gradients = if recurrent
      (gradients, _) = reduce(inputs, init=([], initial_context(nn), [])) do (acc, context, recurrent_layers), (input, target)
        # We're going to build back the entire net as if it were feed-forward on subsequent steps
        # We'll then have to piece it back together down below
        (gradient, next_context, next_recurrent_layers) = derive_gradients(nn, apply_embedding(input), target, context, cost, cost_derivative, recurrent_layers, recurrent, debug)
        
        # TODO: This is obviously all a mess!
        # display("gradient=$gradient")
        all_layers=map(l -> l.config.tag, [map(layer -> layer[begin], recurrent_layers); nn])
        # display("all_layers=$all_layers")
        gradient_pairs = drop_right([zip(all_layers, gradient)...], length(nn))
        gradient_full = map(enumerate(nn)) do (j, layer)
          # display("gradient_pairs=$gradient_pairs")
          gradient_j = gradient[j+length(recurrent_layers)]
          subs = [gradient for (tag, gradient) ∈ gradient_pairs if tag == layer.tag]
          # display("$(layer.tag) gradient[$j]=$(gradient_j),subs=$(subs)")
          if length(subs) > 0
            # display("gradient[$j] with subs=$(mean([subs; [gradient_j]]))")
            mean([subs; [gradient_j]])
          else
            gradient_j
          end
        end
        # We're going to merge gradients here for recurrent layers
        # display("acc=$acc,gradient=$gradient,gradient_full=$gradient_full")
        ([acc; [gradient_full]], next_context, next_recurrent_layers)
      end

      gradients
    else
      map(inputs) do (input, target)
        (gradient, _) = derive_gradients(nn, apply_embedding(input), target, Dict(), cost, cost_derivative, debug)
        # display("gradient=$gradient")
        gradient
      end
    end

    # display("gradients=$gradients")

    gradient_layers = [[gradient[i] for gradient in gradients] for i in 1:length(gradients[1])]

    # display("nn=$nn")
    # display("gradient_layers=$gradient_layers")
    # display("gradient_layers_sz=$(map(x -> size(x[1]), gradient_layers))")

    function reweight((layer, ∂E∂wijs))
      if layer.weights == nothing
        layer
      else
        # display("∂E∂wijs=$∂E∂wijs")
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

  """
      try_check_error((input, actual))

  Try to check the error, but only if the input is in the training set.

  When we're inferencing, it's nice to output our error,
  though we only can calculate it if we have a target,
  (this is, the input was part of our training set).

  This function returns the error if that's the case, and
  otherwise `nothing`.
  """
  function try_check_error((input, actual))
    if haskey(training, input)
      cost(get(training, input, nothing), actual)
    else
      nothing
    end
  end

  """
      infer(nn, input)

  Infer a value from a given neural net.

  The inference function calculates the output of the net
  for a given input case. For a feed-forward neural network
  this will run the inference for each item in the input
  vector. For a recurrent neural network, this function will
  take one step for each value in the input vector.

  Note: this function currently display the output, it doesn't
        return the value. It does return a potential error size.
  """
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
      local context = initial_context(nn)

      # The `ys` is a collection of outputs of the net
      local ys = []

      # display("embedded_input=$embedded_input")

      for i ∈ embedded_input
        # Run a single input, storing the output from this
        # run, as well as a context to use for the next input.
        (y, context) = calculate_value_with_context(nn, i, context)

        # display("i=$i")
        # display("y=$y")
        # display("ys=$ys")

        append!(ys, [y])
      end

      ys
    else
      # For feed-forward neural nets, we can calulate each output
      # in parallel.
      [calculate_value(nn, i) for i ∈ embedded_input]
    end

    # display("output=$output")
    show_fn(output)

    if recurrent
      # TODO: Consider calculating error for recurrent
      0
    else
      mean(filter(x -> x !== nothing, map(try_check_error, zip(input, output))))
    end
  end

  (nn, train, infer)
end
