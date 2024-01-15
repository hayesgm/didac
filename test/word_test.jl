using Test

include("../learn.jl")

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

@testset "Test single color run on multi-typed network" begin
  # Note: mixed layers is producing the wrong answer
  #       which means we're likely applying the wrong
  #       derivative function.
  (nn, train, infer) = build_nn(
    network_layers=[
      (type="sigmoid", nodes=4),
      (type="relu", nodes=4),
      (type="softmax", nodes=4)
    ],
    embedding=embedding,
    training=training,
    show_fn=show_word,
    ϵ=0.05,
    network_config=(; weights=
    [
      #  Layer 1 [Hidden Sigmoid]
      #  N11   N12   N13   N14  | Signals
      [0.111 0.121 0.131 0.141  # Input 1
        0.112 0.122 0.132 0.142 # Input 2
        0.113 0.123 0.133 0.143 # Input 3
        0.114 0.124 0.134 0.144 # Input 4
      ],
      # Layer 2 [Hidden ReLU]
      # N21   N22   N23   N24   | Signals
      [0.211 0.221 0.231 0.241  # N11
        0.212 0.222 0.232 0.242 # N12
        0.213 0.223 0.233 0.243 # N13
        0.214 0.224 0.234 0.244 # N14
      ],
      # Layer 3 [Output Softmax]
      nothing
    ]
    )
  )

  nn = train(nn; case="dog", debug=false)
  error = infer(nn, ["dog"])
  display("error=$error")
  show(nn)

  @test error == 0.37580315703006945
end
