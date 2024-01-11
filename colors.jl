include("learn.jl")

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
# a toy at this phase anyway.jul
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
