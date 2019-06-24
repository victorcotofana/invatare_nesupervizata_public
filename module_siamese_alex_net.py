import keras
import keras.backend as keras_backend
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Concatenate, ZeroPadding2D

# all the hyper-parameters are taken from the github repo of the article

DROPOUT_RATE = 0.5
MARGIN_DELTA = 1
BASE_LEARNING_RATE = 0.001
GAMMA = 0.1
MOMENTUM_RATE = 0.9

WEIGHT_INIT_STD = 0.01
WEIGHT_DECAY = 0.005
BIAS_INIT_CONSTANT = 0.1

INPUT_SHAPE = (227, 227, 3)
QUERY_INPUT = Input(INPUT_SHAPE)
COMPARED_INPUT = Input(INPUT_SHAPE)
DIFF_VIDEO_FRAME_FLAG = Input((1,))


def contrastive_loss_function(y_true, y_pred):
    # y_true = label, target tensor, not used in our case
    # y_pred = prediction, model output tensor

    # just so pep won't annoy me
    _ = y_true

    # the flag for the same video or not is concatenated for the prediction and the both encodings
    # WARN: these are tf.Tensor not np.array
    diff_video_frame_flag = y_pred[0][0]
    query_encoding = y_pred[0][1:4097]
    compared_encoding = y_pred[0][4097:]

    euclidian_distance = keras_backend.sqrt(
        keras_backend.sum(keras_backend.square(query_encoding - compared_encoding), axis=-1, keepdims=True))

    alt_distance = keras_backend.maximum(MARGIN_DELTA - euclidian_distance, 0)

    # simulate the if condition, one side will always be zero
    return (diff_video_frame_flag * euclidian_distance) + ((1 - diff_video_frame_flag) * alt_distance)


def initialize_model():
    # create the Siamese model made from two AlexNet CNNs

    # instantiate an empty model
    alex_net = Sequential()

    # weight initializer: "gaussian" == "normal" distribution
    weight_init = keras.initializers.RandomNormal(stddev=WEIGHT_INIT_STD)

    # aka L2 regularization
    weight_decay = keras.regularizers.l2(WEIGHT_DECAY)

    # bias initializer: consant
    bias_init = keras.initializers.Constant(value=BIAS_INIT_CONSTANT)

    # 1st Convolutional Layer
    alex_net.add(Conv2D(input_shape=INPUT_SHAPE, filters=96, kernel_size=(7, 7), strides=(2, 2),
                        kernel_initializer=weight_init, kernel_regularizer=weight_decay, bias_initializer=bias_init))
    alex_net.add(Activation('relu'))
    alex_net.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 2nd Convolutional Layer
    alex_net.add(Conv2D(filters=384, kernel_size=(5, 5), strides=(2, 2),
                        kernel_initializer=weight_init, kernel_regularizer=weight_decay, bias_initializer=bias_init))
    alex_net.add(Activation('relu'))
    alex_net.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd Convolutional Layer
    # add padding of 1
    alex_net.add(ZeroPadding2D(padding=(1, 1)))
    alex_net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                        kernel_initializer=weight_init, kernel_regularizer=weight_decay, bias_initializer=bias_init))
    alex_net.add(Activation('relu'))

    # 4th Convolutional Layer
    alex_net.add(ZeroPadding2D(padding=(1, 1)))
    alex_net.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                        kernel_initializer=weight_init, kernel_regularizer=weight_decay, bias_initializer=bias_init))
    alex_net.add(Activation('relu'))

    # 5th Convolutional Layer
    alex_net.add(ZeroPadding2D(padding=(1, 1)))
    alex_net.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                        kernel_initializer=weight_init, kernel_regularizer=weight_decay, bias_initializer=bias_init))
    alex_net.add(Activation('relu'))
    alex_net.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # passing it to a Fully Connected layer, being the length of 6x6x256 = 9216
    alex_net.add(Flatten())

    # 1st Fully Connected Layer, 6th Overall Layer
    alex_net.add(Dense(4096,
                       kernel_initializer=weight_init, kernel_regularizer=weight_decay,
                       bias_initializer=bias_init))
    alex_net.add(Activation('relu'))
    # add Dropout to prevent overfitting
    alex_net.add(Dropout(DROPOUT_RATE))

    # 2nd Fully Connected Layer, 7th Overall Layer
    alex_net.add(Dense(4096,
                       kernel_initializer=weight_init, kernel_regularizer=weight_decay,
                       bias_initializer=bias_init))
    alex_net.add(Activation('relu'))
    alex_net.add(Dropout(DROPOUT_RATE))

    # encode each of the two inputs into a vector with the alexnet
    query_encoded = alex_net(QUERY_INPUT)
    compared_encoded = alex_net(COMPARED_INPUT)

    # add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Concatenate()([DIFF_VIDEO_FRAME_FLAG, query_encoded, compared_encoded])

    # input: flag if the two frames are from the same video or not
    # output: the same flag, and the feature spaces of the frames
    siamese_net = Model(inputs=[DIFF_VIDEO_FRAME_FLAG, QUERY_INPUT, COMPARED_INPUT], outputs=prediction)

    print('SIAMESE ALEXNET SUMMARY:')
    siamese_net.summary()

    print('SIAMESE INPUT:')
    print(siamese_net.input_shape)

    sgd_optimizer = keras.optimizers.SGD(lr=BASE_LEARNING_RATE,
                                         decay=GAMMA,
                                         momentum=MOMENTUM_RATE,
                                         nesterov=False)
    siamese_net.compile(loss=contrastive_loss_function,
                        optimizer=sgd_optimizer)

    return siamese_net


def get_saved_model(path_to_model):
    siamese_model = load_model(path_to_model, custom_objects={
        'contrastive_loss_function': contrastive_loss_function
    })

    return siamese_model
