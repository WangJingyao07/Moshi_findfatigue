#__author__ = 'Wang Jingyao'
from keras.applications import *
from keras.layers import Input, Lambda, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
import h5py
height = 224
width = 224
x = Input((width, height, 3))
base_model = VGG19(input_tensor=x, input_shape=(width, height, 3), include_top=False, pooling="avg", weights="imagenet")
imgGen = ImageDataGenerator()
train_gen = imgGen.flow_from_directory("data/train", target_size=(width, height), shuffle=True, batch_size=32)
valid_gen = imgGen.flow_from_directory("data/valid", target_size=(width, height), shuffle=True, batch_size=32)
# train_gen = imgGen.flow_from_directory("train", target_size=(width, height), shuffle=True, batch_size=32)
# valid_gen = imgGen.flow_from_directory("valid", target_size=(width, height), shuffle=True, batch_size=32)
new_output = Dropout(0.5)(base_model.output)
new_output = Dense(10, activation="softmax")(new_output)
new_model = Model(inputs=base_model.input, outputs=new_output)
for layer in new_model.layers[:-9]:
  layer.trainable = False
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
new_model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
#new_model.summary()
new_model.load_weights("drive/vgg19.hdf5")
#这里的checkpoint可以在每个epoch后检测val_loss如果比之前的小，那么把权重存下来
m_checkpoint = ModelCheckpoint("drive/vgg19.hdf5", monitor="val_loss", mode="min", save_best_only=True, save_weights_only=True, verbose=1, period=1)
#这里的early stopping可以在5个epoch个没有改进就停止训练，防止过拟合
m_earlyStopping = EarlyStopping(monitor="val_loss", min_delta=0.0003, patience=5, verbose=1, mode="min")
new_model.fit_generator(train_gen, 19544 // 32, 15, validation_data=valid_gen, validation_steps=2880 // 32, callbacks=[m_checkpoint, m_earlyStopping])