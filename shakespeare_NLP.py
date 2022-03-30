# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GRU
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import Sequential

# import pandas as pd
# import matplotlib.pyplot as plt

# from tensorflow.keras.models import load_model

# %%
path_to_file = "../06-NLP-and-Text-Data/shakespeare.txt"
text = open(path_to_file).read()
vocab = sorted(set(text))
vocab_size = len(vocab)
char_to_ind = {char: ind for ind, char in enumerate(vocab)}
ind_to_char = np.array(vocab)

encoded_text = np.array([char_to_ind[c] for c in text])
# %%
seq_len = 120  # Three lines-ish
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
sequences = char_dataset.batch(batch_size=seq_len + 1, drop_remainder=True)


dataset = sequences.map(
    tf.autograph.experimental.do_not_convert(lambda seq: (seq[:-1], seq[1:]))
)
batch_size = 128
buffer_size = 10000

dataset = dataset.shuffle(buffer_size=buffer_size).batch(
    batch_size=batch_size, drop_remainder=True
)

# %%
# sparse_cross = lambda y_true, y_pred: sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)


def sparse_cross(y_true, y_pred):
    return sparse_categorical_crossentropy(
        y_true=y_true, y_pred=y_pred, from_logits=True
    )


def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
    model = Sequential()
    model.add(
        layer=Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            batch_input_shape=[batch_size, None],
        )
    )
    model.add(
        layer=GRU(
            units=rnn_neurons,
            return_sequences=True,
            stateful=True,
            recurrent_initializer="glorot_uniform",
        )
    )
    model.add(layer=Dense(units=vocab_size))
    model.compile(optimizer="adam", loss=sparse_cross)

    model.summary()
    return model


def generate_text(model, start_seed, gen_size=500, temp=1.0):
    num_generate = gen_size
    input_eval = [char_to_ind[s] for s in start_seed]
    input_eval = tf.expand_dims(input_eval, axis=0)
    text_generated = []
    temperature = temp
    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, axis=0)
        predictions /= temperature
        predicted_id = tf.random.categorical(logits=predictions, num_samples=1)[
            -1, 0
        ].numpy()
        input_eval = tf.expand_dims(input=[predicted_id], axis=0)
        text_generated.append(ind_to_char[predicted_id])
    return start_seed + "".join(text_generated)


# %%


embed_dim = 64  # Features to embed input --> vector
rnn_neurons = 1026
epochs = 30


model = create_model(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    rnn_neurons=rnn_neurons,
    batch_size=batch_size,
)

# %%
model.fit(x=dataset, epochs=epochs)
my_model_path = (
    "/home/marcel/Udemy/TF_2_Keras_deep_bootcamp/06-NLP-and-Text-Data/my_model.h5"
)
model.save(my_model_path)

# %%

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
sampled_indices = tf.random.categorical(
    logits=example_batch_predictions[0], num_samples=1
)

sampled_indices = tf.squeeze(input=sampled_indices, axis=1)
# %%
loaded_model = create_model(
    vocab_size=vocab_size, embed_dim=embed_dim, rnn_neurons=rnn_neurons, batch_size=1
)


load_model_path = "/home/marcel/Udemy/TF_2_Keras_deep_bootcamp/06-NLP-and-Text-Data/shakespeare_gen.h5"
loaded_model.load_weights(load_model_path)
loaded_model.build(tf.TensorShape([1, None]))

# %%
print(generate_text(model=loaded_model, start_seed="Sandra", gen_size=1000))

# %%
