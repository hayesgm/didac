include("../learn.jl")

# This could probably be made using an identity matrix and an index paramter
embedding = Dict(
  "continue" => [0],
  "stop" => [1]
)

colors = Dict(
  "red" => [0.9, 0.2, 0.2],
  "yellow" => [1.0, 0.91, 0.0],
  "green" => [0.2, 0.9, 0.2],
)

color(x) = colors[x]

# TODO: Should we learn a starting state?
# TODO: We probably still need a lot more training cases
training = Dict(
  ["continue", "continue", "continue", "continue"] =>
    color.(["green", "yellow", "red", "green"]),
  ["continue", "continue", "continue", "continue"] =>
    color.(["yellow", "red", "green", "yellow"]),
  ["continue", "continue", "continue", "continue"] =>
    color.(["red", "green", "yellow", "red"]),
  ["continue", "continue", "continue", "continue", "continue", "continue", "continue"] =>
    color.(["red", "green", "yellow", "red", "green", "yellow", "red"]),
  ["continue", "stop", "continue", "continue", "continue"] =>
    color.(["green", "red", "green", "yellow", "red"])
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
    (type="sigmoid", nodes=2, recurrent="l2-context", tag="l1-hidden"),
    (type="sigmoid", nodes=2, feedback=true, tag="l2-context"),
    (type="sigmoid", nodes=3, tag="l3-output")
  ],
  recurrent=true,
  embedding=embedding,
  training=training,
  show_fn=show_sixel,
  ϵ=0.05,
  network_config=(;)
)

display(nn)
# for i ∈ 1:10
#   for j ∈ 1:20000
#     global nn = train(nn)
#   end

err = infer(nn, ["continue", "continue", "continue"])
display("err=$err")
# end

# show(nn)
