import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM,
    Dense, Dropout, GlobalAveragePooling1D,
    GlobalMaxPooling1D, concatenate, Layer, Conv1D
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


# --------------------------
# Custom Attention Layer
# --------------------------
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# --------------------------
# Build Deep BiLSTM + Attention Model
# --------------------------
def build_bilstm_attention_model(embedding_matrix, max_len, num_words, embed_dim, num_classes):
    """
    Creates a Deep BiLSTM model with Attention and CNN feature extraction.
    """
    inputs = Input(shape=(max_len,), name="input_layer")

    # Embedding layer using pre-trained GloVe embeddings
    embedding_layer = Embedding(
        input_dim=num_words,
        output_dim=embed_dim,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=False,
        name="embedding"
    )(inputs)

    # Add a CNN layer for feature extraction
    conv = Conv1D(filters=128, kernel_size=3, padding="same", activation="relu")(embedding_layer)
    conv = Dropout(0.3)(conv)

    # Bidirectional LSTM with dropout
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(conv)

    # Attention layer
    attn = Attention()(lstm_out)

    # Global pooling layers for additional feature diversity
    avg_pool = GlobalAveragePooling1D()(lstm_out)
    max_pool = GlobalMaxPooling1D()(lstm_out)

    merged = concatenate([attn, avg_pool, max_pool])
    dense1 = Dense(128, activation="relu")(merged)
    dense1 = Dropout(0.4)(dense1)
    outputs = Dense(num_classes, activation="softmax")(dense1)

    model = Model(inputs=inputs, outputs=outputs)
    return model
