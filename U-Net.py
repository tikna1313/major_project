from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as k
import pickle
import numpy as np

f1 = open('Input_images.m', 'rb')# training input image saved in pickle Object file(SIZE is too big to Upload).
j = pickle.load(f1)
f2 = open('mask_images.m', 'rb')# training output mask_images same as above.
m = pickle.load(f2)

input_image = Input(shape=[256, 256, 3])

# encoder

l = Conv2D(2 ** 10, (3, 3), activation="relu", padding='same')(input_image)
l = MaxPooling2D((2, 2), padding='same')(l)
l = Conv2D(2 ** 9, (3, 3), activation="relu", padding='same')(l)
l = MaxPooling2D((2, 2), padding='same')(l)
l = Conv2D(2 ** 8, (3, 3), activation="relu", padding='same')(l)
l = MaxPooling2D((2, 2), padding='same')(l)
l = Conv2D(2 ** 7, (3, 3), activation="relu", padding='same')(l)
l = MaxPooling2D((2, 2), padding='same')(l)
l = Conv2D(2 ** 6, (3, 3), activation="relu", padding='same')(l)
l = MaxPooling2D((2, 2), padding='same')(l)
l = Conv2D(2 ** 5, (3, 3), activation="relu", padding='same')(l)
l = MaxPooling2D((2, 2), padding='same')(l)
l = Conv2D(2 ** 4, (3, 3), activation="relu", padding='same')(l)
l = MaxPooling2D((2, 2), padding='same')(l)
l = Conv2D(2 ** 3, (3, 3), activation="relu", padding='same')(l)
l = MaxPooling2D((2, 2), padding='same')(l)
encoder = Conv2D(2 ** 2, (3, 3), activation="relu", padding='same')(l)
#encoder = MaxPooling2D((2, 2), padding='same')(l)

# decoder
l = Conv2D(2 ** 2, (3, 3), activation="relu", padding='same')(encoder)
l = UpSampling2D((2, 2))(l)
l = Conv2D(2 ** 3, (3, 3), activation="relu", padding='same')(l)
l = UpSampling2D((2, 2))(l)
l = Conv2D(2 ** 4, (3, 3), activation="relu", padding='same')(l)
l = UpSampling2D((2, 2))(l)
l = Conv2D(2 ** 5, (3, 3), activation="relu", padding='same')(l)
l = UpSampling2D((2, 2))(l)
l = Conv2D(2 ** 6, (3, 3), activation="relu", padding='same')(l)
l = UpSampling2D((2, 2))(l)
l = Conv2D(2 ** 7, (3, 3), activation="relu", padding='same')(l)
l = UpSampling2D((2, 2))(l)
l = Conv2D(2 ** 8, (3, 3), activation="relu", padding='same')(l)
l = UpSampling2D((2, 2))(l)
l = Conv2D(2 ** 9, (3, 3), activation="relu", padding='same')(l)
l = UpSampling2D((2, 2))(l)
decoder = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(l)
print('Training Start')
auto_encoder = Model(input_image, decoder)
auto_encoder.compile(optimizer='Adam', loss='mean_squared_error')
auto_encoder.summary()
#images = [f for f in listdir('train/train')]
#image_masks = [k for k in listdir('train_masks/train_masks')]
auto_encoder.fit((np.reshape(np.array(j[:501], dtype="float64"), newshape=[501, 256, 256, 3]))/255, (np.reshape(np.array(m[:501], dtype="float64"), newshape=[501, 256, 256, 3]))/255,
                 epochs=1000,
                 batch_size=50,
                 shuffle=True,
                 validation_data=(np.reshape(np.array(j[501:669], dtype="float64"), newshape=[168, 256, 256, 3])/255, np.reshape(np.array(m[501:669], dtype="float64"), newshape=[168, 256, 256, 3])/255),
                 verbose=1,
                 )
auto_encoder.save("Mask_Gen.hdf5")