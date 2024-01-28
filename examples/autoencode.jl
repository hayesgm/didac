include("../didac.jl")

# We need to find some data that has some good definition
# of distance, so that we can evaluate and train the
# auto-encoder. That is, if it's just "right" or "wrong,"
# then the auto-encoder will simply optimize for getting
# a few definitively right, versus a mix of nearby, which
# is the goal of the auto-encoder.

# Sentences can be a fun one, since we could just verify
# on getting the most correct words right.

# Okay, the idea will be identifying song genres by their
# chord progressions for now. So to start, we'll come up
# with a dummy list of songs and their chords and label
# the song. Once that's done, we could try to load in some
# real data.

# Well, if we want a better dataset, etc, we can do that
# after we understand how this works.

pretrain = true

genres = Dict(
  "Jazz" => [1, 0],
  "Pop" => [0, 1]
)

chords = Dict(
  "C" => 1.0,
  "C♯" => 2.0,
  "D" => 3.0,
  "E♭" => 4.0,
  "E" => 5.0,
  "F" => 6.0,
  "F♯" => 7.0,
  "G" => 8.0,
  "G♯" => 9.0,
  "A" => 10.0,
  "B♭" => 11.0,
  "B" => 12.0
)

songs = [
  (name="Song 1", chords=["A", "B", "A", "G"], genre="Jazz"),
  (name="Song 2", chords=["A", "G", "A", "B"], genre="Pop"),
  (name="Song 3", chords=["F", "F♯", "F", "F♯"], genre="Pop"),
  (name="Song 4", chords=["G", "F♯", "G", "G"], genre="Pop")
]

encode_chord(c) = get(chords, c, nothing) / 12.0
encode_genre(g) = get(genres, g, nothing)

show_chord(c) = round(c * 12.0)
show_chords(chords) = display(show_chord.(chords))

# show_genres(genre) = display(show_genre.(genres))
# show_genres(genre) = display(show_genre.(genres))

song_training = map(songs) do song
  encode_chord.(song.chords) => encode_genre(song.genre)
end

display("song_training=$song_training")

automap((x, y)) = x => x

initial_layers = if pretrain
  auto_encode_training = automap.(song_training)

  display("auto_encode_training=$auto_encode_training")

  # TODO: We might realllly need biases added before this works
  # TODO: We're using squared difference error, is that best?
  # TODO: Embedding is a bit weird to be a huge part of the input
  # Note: Bag of words could also technically be better as a real task, but this is more fun
  (ae_nn, ae_train, ae_infer) = build_nn(
    network_layers=[
      (type="sigmoid", nodes=8, tag="encoder-1"),
      (type="sigmoid", nodes=4, tag="encoder-2"),
      (type="sigmoid", nodes=2, tag="code"),
      (type="sigmoid", nodes=4, tag="decoder-1")
    ],
    training=auto_encode_training,
    show_fn=(x) -> show_chords.(x),
    ϵ=50,
    network_config=(;)
  )

  display(ae_nn)

  for i ∈ 1:10
    for j ∈ 1:10000
      global ae_nn = ae_train(ae_nn)
    end

    err = ae_infer(ae_nn, [encode_chord.(["A", "B", "A", "G"])])
    display("err=$err")
  end

  map(songs) do song
    # TODO: I need to be able to extract the code layer from
    # the inference!

    # TODO: We really need biases, I believe

    # TODO: Next step is to then apply labels and train a new nn that incorporates this NN
    err = ae_infer(ae_nn, [encode_chord.(song.chords)])
    display("err=$err")
    display("song=$(song.name), err=$err")
  end

  [
    (type="sigmoid", nodes=8, tag="encoder-1", weights=ae_nn[1].weights),
    (type="sigmoid", nodes=4, tag="encoder-2", weights=ae_nn[2].weights)
  ]
else
  [
    (type="sigmoid", nodes=8, tag="encoder-1"),
    (type="sigmoid", nodes=4, tag="encoder-2"),
  ]
end

# Next move is to take some of the early layers
# and to share them with a new nn with those same
# early layers, but now with fine-tuning.
(nn, train, infer) = build_nn(
  network_layers=[
    initial_layers...,
    (type="sigmoid", nodes=2, tag="output-1"),
    (type="softmax", nodes=2, tag="softmax-1")
  ],
  training=song_training,
  ϵ=0.05,
  cost_fn="cross-entropy",
  network_config=(;)
)

display(nn)

for i ∈ 1:10
  for j ∈ 1:10000
    global nn = train(nn)
  end

  err = infer(nn, [encode_chord.(["A", "B", "A", "G"])])
  display("err=$err")
end

# show(nn)

# So, for the first phase, we are just looking to auto-encode the chords of the song
# So we'll build out this net first, just to make sure we understand.
