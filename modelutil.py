# import os 
# from tensorflow.keras.models import Sequential 
# from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

# def load_model() -> Sequential: 
#     model = Sequential()

#     model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1,2,2)))

#     model.add(Conv3D(256, 3, padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1,2,2)))

#     model.add(Conv3D(75, 3, padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1,2,2)))

#     model.add(TimeDistributed(Flatten()))

#     model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
#     model.add(Dropout(.5))

#     model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
#     model.add(Dropout(.5))

#     model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

#     model.load_weights(os.path.join('..','models','checkpoint.weights.h5'))
#     # model.load_weights('C:/path/to/your/models/checkpoint.weights.h5')

#     return model

























import os
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import (
    Conv3D, LSTM, Dense, Dropout, Bidirectional, 
    MaxPool3D, Activation, Reshape, SpatialDropout3D, 
    BatchNormalization, TimeDistributed, Flatten
)

def load_model() -> Sequential: 
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    # Add error handling when loading weights
    try:
        model.load_weights(os.path.join('..', 'models', 'checkpoint'))  # Change this if needed
    except ValueError as e:
        print("Error loading model weights:", e)
        # You might want to handle different cases here or set default weights
    except Exception as e:
        print("An unexpected error occurred:", e)

    return model





