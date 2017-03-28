from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, GlobalAveragePooling1D, Activation, TimeDistributed, BatchNormalization
from keras.layers import GlobalMaxPooling1D, Convolution1D, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.merge import Concatenate, Add, Dot, Multiply
from keras.optimizers import Adam
from keras.layers.advanced_activations import PReLU
from keras import backend as K

from keras_decomposable_attention import build_model

from utils import get_data_and_embeddings


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def tdd_network(x1_shape, input_dims, embedding_matrix):
    input_a = Input(shape=x1_shape[1:])
    input_b = Input(shape=x1_shape[1:])

    def create_base_network(x1_shape, input_dims, embedding_matrix):
        inp = Input(x1_shape[1:])
        embedding_block = Embedding(
            input_dim=input_dims,
            output_dim=300,
            weights=[embedding_matrix],
            input_length=40,
            trainable=False
        )(inp)
        embedding_block = TimeDistributed(Dense(300, activation='relu'))(embedding_block)
        embedding_block = Lambda(lambda x: K.sum(x, axis=1))(embedding_block)
        embedding_block = Model(inputs=inp, outputs=embedding_block)
        return embedding_block

    base_network = create_base_network(x1_shape, input_dims, embedding_matrix)
    processed_a = base_network(input_a)  # shared
    processed_b = base_network(input_b)  # layers
    
    distance1 = Lambda(euclidean_distance,
                       output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    
    distance2 = Dot(axes=1, normalize=True)([processed_a, processed_b])
    
    merged = Concatenate()([processed_a, processed_b, distance2])
    print(merged.shape)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(200)(merged)
    merged = PReLU()(merged)

    merged = BatchNormalization()(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(200)(merged)
    merged = PReLU()(merged)

    merged = BatchNormalization()(merged)
    merged = Dense(1, activation="sigmoid")(merged)
    model = Model(inputs=[input_a, input_b], outputs=merged)
    model.compile(loss="binary_crossentropy", optimizer=Adam(1e-3), metrics=["accuracy"])
    return model


def conv_network(x1_shape, input_dims, embedding_matrix):
    input_a = Input(shape=x1_shape[1:])
    input_b = Input(shape=x1_shape[1:])

    def create_base_network(x1_shape, input_dims, embedding_matrix):
        inp = Input(x1_shape[1:])
        embedding_block = Embedding(
            input_dim=input_dims,
            output_dim=300,
            weights=[embedding_matrix],
            input_length=40,
            trainable=False
        )(inp)
        embedding_block = Convolution1D(64, 5,)(embedding_block)
        embedding_block = PReLU()(embedding_block)
        embedding_block = Dropout(0.2)(embedding_block)
        embedding_block = Convolution1D(64, 5,)(embedding_block)
        embedding_block = PReLU()(embedding_block)
        embedding_block = GlobalMaxPooling1D()(embedding_block)
        embedding_block = BatchNormalization()(embedding_block)
        embedding_block = Model(inputs=inp, outputs=embedding_block)
        return embedding_block

    base_network = create_base_network(x1_shape, input_dims, embedding_matrix)
    processed_a = base_network(input_a)  # shared
    processed_b = base_network(input_b)  # layers
    
    distance1 = Lambda(euclidean_distance,
                       output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    
    distance2 = Dot(axes=1, normalize=True)([processed_a, processed_b])
    
    merged = Concatenate()([processed_a, processed_b, distance2])

    merged = BatchNormalization()(merged)
    merged = Dense(64)(merged)
    merged = PReLU()(merged)
    merged = Dropout(0.1)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(64)(merged)
    merged = PReLU()(merged)

    merged = BatchNormalization()(merged)
    merged = Dense(1, activation="sigmoid")(merged)
    model = Model(inputs=[input_a, input_b], outputs=merged)
    model.compile(loss="binary_crossentropy", optimizer=Adam(1e-3), metrics=["accuracy"])
    return model


def lstm_network(x1_shape, input_dims, embedding_matrix):
    input_a = Input(shape=x1_shape[1:])
    input_b = Input(shape=x1_shape[1:])
    
    def create_base_network(x1_shape, input_dims, embedding_matrix):
        inp = Input(x1_shape[1:])
        embedding_block = Embedding(
            input_dim=input_dims,
            output_dim=300,
            weights=[embedding_matrix],
            input_length=40,
            trainable=False
        )(inp)
        #embedding_block = LSTM(32, return_sequences=True)(embedding_block)
        #embedding_block = BatchNormalization()(embedding_block)
        embedding_block = LSTM(32, return_sequences=True)(embedding_block)
        embedding_block = Lambda(lambda x: K.sum(x, axis=1))(embedding_block)
        embedding_block = Model(inputs=inp, outputs=embedding_block)
        return embedding_block

    base_network = create_base_network(x1_shape, input_dims, embedding_matrix)
    processed_a = base_network(input_a)  # shared
    processed_b = base_network(input_b)  # layers
    
    distance1 = Lambda(lambda x: K.mean(K.abs(x[1] - x[0]), axis=1, keepdims=True))([
        processed_a,
        processed_b,
    ])
    distance1 = BatchNormalization()(distance1)
    distance2 = Dot(axes=1, normalize=True)([
        processed_a,
        processed_b,
    ])
    
    merged = Concatenate()([processed_a, processed_b, distance2])

    merged = Dense(32)(merged)
    merged = PReLU()(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(32)(merged)
    merged = PReLU()(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(1, activation="sigmoid")(merged)
    model = Model(inputs=[input_a, input_b], outputs=merged)
    model.compile(loss="binary_crossentropy", optimizer=Adam(1e-3), metrics=["accuracy"])
    return model


def siamese_network(x1_shape, input_dims, embedding_matrix):
    input_a = Input(shape=x1_shape[1:])
    input_b = Input(shape=x1_shape[1:])
    
    def create_base_network(x1_shape, input_dims, embedding_matrix):
        inp = Input(x1_shape[1:])
        embedding_block = Embedding(
            input_dim=input_dims,
            output_dim=300,
            weights=[embedding_matrix],
            input_length=40,
            trainable=False
        )(inp)
        #embedding_block = LSTM(32, return_sequences=True)(embedding_block)
        #embedding_block = BatchNormalization()(embedding_block)
        embedding_block = Bidirectional(LSTM(32, return_sequences=True))(embedding_block)
        embedding_block = Lambda(lambda x: K.sum(x, axis=1))(embedding_block)
        embedding_block = Model(inputs=inp, outputs=embedding_block)
        return embedding_block

    base_network = create_base_network(x1_shape, input_dims, embedding_matrix)
    processed_a = base_network(input_a)  # shared
    processed_b = base_network(input_b)  # layers
    
    distance1 = Lambda(lambda x: K.abs(x[1] - x[0]))([
        processed_a,
        processed_b,
    ])
    distance1 = BatchNormalization()(distance1)
    distance2 = Multiply()([
        processed_a,
        processed_b,
    ])
    distance2 = BatchNormalization()(distance2)
    
    merged = Concatenate()([distance1, distance2])

    merged = Dense(64)(merged)
    merged = PReLU()(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(64)(merged)
    merged = PReLU()(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(1, activation="sigmoid")(merged)
    model = Model(inputs=[input_a, input_b], outputs=merged)
    model.compile(loss="binary_crossentropy", optimizer=Adam(1e-3), metrics=["accuracy"])
    return model


def decomposable_attention_network(x1_shape, input_dims, embedding_matrix):
    input_a = Input(shape=x1_shape[1:])
    input_b = Input(shape=x1_shape[1:])
    
    def create_base_network(x1_shape, input_dims, embedding_matrix):
        inp = Input(x1_shape[1:])
        embedding_block = Embedding(
            input_dim=input_dims,
            output_dim=300,
            weights=[embedding_matrix],
            input_length=40,
            trainable=False
        )(inp)
        embedding_block = TimeDistributed(Dense(100, activation='relu'))(embedding_block)
        embedding_block = Model(inputs=inp, outputs=embedding_block)
        return embedding_block

    base_network = create_base_network(x1_shape, input_dims, embedding_matrix)
    processed_a = base_network(input_a)  # shared
    processed_b = base_network(input_b)  # layers

    def attention(m1, m2):
        def outer_prod(AB):
            att_ji = K.batch_dot(AB[1], K.permute_dimensions(AB[0], (0, 2, 1)))
            return K.permute_dimensions(att_ji,(0, 2, 1))

        td = TimeDistributed(Dense(100, activation="relu"))
        td1 = td(m1)
        td1 = TimeDistributed(Dropout(0.1))(td1)
        td2 = td(m2)
        td2 = TimeDistributed(Dropout(0.1))(td2)
        model = Lambda(outer_prod)([td1, td2])
        return model

    attend = attention(processed_a, processed_b)
    print(attend.shape)

    def align(m, attmat, transpose=False):
        def normalise_attention(attmat):
            att = attmat[0]
            mat = attmat[1]
            if transpose:
                att = K.permute_dimensions(att,(0, 2, 1))
            e = K.exp(att - K.max(att, axis=-1, keepdims=True))
            s = K.sum(e, axis=-1, keepdims=True)
            sm_att = e / s
            return K.batch_dot(sm_att, mat)
        return Lambda(normalise_attention)([attmat, m])
        

    align1 = align(processed_a, attend)
    align2 = align(processed_b, attend, transpose=True)
    print(align1.shape, align2.shape)

    compare = TimeDistributed(Dense(100, activation="relu"))
    
    compare1 = compare(Concatenate()([processed_a, align1]))
    compare1 = Concatenate()([GlobalAveragePooling1D()(compare1), GlobalMaxPooling1D()(compare1)])
    compare2 = compare(Concatenate()([processed_b, align2]))
    compare2 = Concatenate()([GlobalAveragePooling1D()(compare2), GlobalMaxPooling1D()(compare2)])

    merged = Concatenate()([BatchNormalization()(compare1), BatchNormalization()(compare2)])
    merged = Dropout(0.1)(merged)
    #merged = Dense(64)(merged)
    #merged = PReLU()(merged)
    #merged = BatchNormalization()(merged)
    merged = Dense(100)(merged)
    merged = PReLU()(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(1, activation="sigmoid")(merged)
    model = Model(inputs=[input_a, input_b], outputs=merged)
    model.compile(loss="binary_crossentropy", optimizer=Adam(1e-3), metrics=["accuracy"])
    return model

    
model_list = [
    ("decomposable_attention", decomposable_attention_network),
    ("siamese", siamese_network),
    ("conv", conv_network),
    ("lstm", lstm_network),
    ("tdd", tdd_network)
]
