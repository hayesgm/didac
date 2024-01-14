include("learn.jl")

# https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exlimit=max&explaintext&exintro&titles=Steam+Engine&redirects=

texts = [
  "the dog jumped over the house",
  "a cat ran up a pipe",
]

# This could probably be made using an identity matrix and an index paramter
embedding = Dict(
  "the" =>    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "a" =>      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  "dog" =>    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  "cat" =>    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  "jumped" => [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
  "ran" =>    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
  "over" =>   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
  "up" =>     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
  "house" =>  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
  "pipe" =>   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
)

inverted_embedding = Dict(v => k for (k, v) in embedding)

# TODO: Even for the joke set, we need a lot more training cases!
training = Dict(
  ["the", "dog", "over", "the"] => "jumped",
  ["dog", "jumped", "the", "house"] => "over",
  ["a", "cat", "up", "a"] => "ran",
  ["cat", "ran", "a", "pipe"] => "up",
)

function get_word(vector)
  _, max_index = findmax(vector)

  # Create a one-hot vector
  one_hot_vector = [i == max_index ? 1 : 0 for i in 1:length(vector)]

  get(inverted_embedding, one_hot_vector, nothing)
end

function show_word(output)
  display(get_word.(output))
end

# TODO: This needs to use the cross-entropy function
# But otherwise, we're basically set
(nn, train, infer) = build_nn(
  network_layers=[
  # TODO: Consider layers better
    ("sigmoid", 4),
    ("sigmoid", 4),
    ("softmax", 4)
  ],
  embedding=embedding,
  training=training,
  show_fn=show_word,
  cost_fn="cross-entropy",
  ϵ=0.05
)

for i ∈ 1:10
  for j ∈ 1:10000
    global nn = train(nn)
  end
  
  infer(nn, ["dog", "cat", "house", "boat"])
end

show(nn)
