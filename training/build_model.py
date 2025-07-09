from keras.layers import (
    Dense,
    Input,
    LSTM,
    Embedding,
    GRU,
)
from keras.layers import (
    Bidirectional,
    GlobalMaxPooling1D,
)
from keras.layers import Input, Embedding, Dense
from keras.layers import Concatenate, SpatialDropout1D
from keras.models import Model
from keras import backend as K
from keras import optimizers
def build_model(embedding_matrix, nb_words, embedding_size=300, max_length=55):
    inp = Input(shape=(max_length,))
    x = Embedding(
        nb_words, embedding_size, weights=[embedding_matrix], trainable=False
    )(inp)
    x = SpatialDropout1D(0.3)(x)
    x1 = Bidirectional(LSTM(256, return_sequences=True))(x)
    x2 = Bidirectional(GRU(128, return_sequences=True))(x1)
    max_pool1 = GlobalMaxPooling1D()(x1)
    max_pool2 = GlobalMaxPooling1D()(x2)
    conc = Concatenate()([max_pool1, max_pool2])
    predictions = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=predictions)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
