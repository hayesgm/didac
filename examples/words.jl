include("../didac.jl")

# This could probably be made using an identity matrix and an index paramter
embedding = Dict(
  "dog" => [1, 0, 0, 0],
  "cat" => [0, 1, 0, 0],
  "house" => [0, 0, 1, 0],
  "boat" => [0, 0, 0, 1]
)

inverted_embedding = Dict(v => k for (k, v) in embedding)

training = Dict(
  "dog" => [0, 1, 0, 0],
  "cat" => [1, 0, 0, 0],
  "house" => [0, 0, 0, 1],
  "boat" => [0, 0, 1, 0]
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

(nn, train, infer) = build_nn(
  network_layers=[
    (type="sigmoid", nodes=4),
    (type="sigmoid", nodes=4),
    (type="softmax", nodes=4)
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
