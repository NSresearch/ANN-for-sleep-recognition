import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def train_network(input_size,output_size, epo, x_train, y_train, training_data) :
 model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(input_size,)),
  tf.keras.layers.Dense(output_size, activation='sigmoid', use_bias=True)
 ])
 loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
 model.compile(optimizer='adam',
               loss=loss_fn,
               metrics=['accuracy'])
 history = model.fit(x_train, y_train, validation_split=0, epochs=epo)

 model.save('model/my_model')

 plt.plot(history.history['accuracy'])
 #plt.plot(history.history['val_accuracy'])
 plt.title('model accuracy')
 plt.ylabel('accuracy')
 plt.xlabel('epoch')
 #plt.legend(['train', 'test'], loc='upper left')
 plt.savefig("training_acc_" + training_data+".png")
 plt.show()

 plt.plot(history.history['loss'])
 #plt.plot(history.history['val_loss'])
 plt.title('model loss')
 plt.ylabel('loss')
 plt.xlabel('epoch')
 #plt.legend(['train', 'test'], loc='upper right')
 plt.savefig("training_loss_" + training_data +".png")
 plt.show()

 return history
