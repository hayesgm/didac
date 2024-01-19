using Test

include("../learn.jl")

# This could probably be made using an identity matrix and an index paramter
embedding = Dict(
  "red" => [1, 0, 0, 0],
  "green" => [0, 1, 0, 0],
  "blue" => [0, 0, 1, 0],
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
  "deep-red" => [0.8, 0.1, 0.1]
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

@testset "Test single color run on multi-typed network" begin
  # Note: mixed layers is producing the wrong answer
  #       which means we're likely applying the wrong
  #       derivative function.
  (nn, train, infer) = build_nn(
    network_layers=[
      (type="sigmoid", nodes=4),
      (type="relu", nodes=3),
      (type="sigmoid", nodes=3)
    ],
    embedding=embedding,
    training=training,
    show_fn=show_sixel,
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
      # N21   N22   N23   | Signals
      [0.211 0.221 0.231  # N11
        0.212 0.222 0.232 # N12
        0.213 0.223 0.233 # N13
        0.214 0.224 0.234 # N14
      ],
      # Layer 3 [Output Sigmoid]
      # N31   N32   N33   | Signals
      [0.311 0.321 0.331  # N21
        0.312 0.322 0.332 # N22
        0.313 0.323 0.333 # N23
      ]
    ]
    )
  )

  nn = train(nn; case="red", debug=false)
  error = infer(nn, ["red"])
  display("error=$error")
  show(nn)

  @test error == 0.2126231568899236
end
