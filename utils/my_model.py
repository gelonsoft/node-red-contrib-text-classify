import tensorflow_hub as hub
import tensorflow as tf
import os

from keras import optimizers
from official.nlp import optimization
import tensorflow_text as text

def create_new_model(value_count):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(os.path.dirname(os.path.realpath(__file__)) +'/../bert/preprocessor', name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(os.path.dirname(os.path.realpath(__file__)) +'/../bert/encoder', trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    output = tf.keras.layers.Dense(units=value_count,kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.,stddev=1.), name='y', activation='softmax')(net)
    model=tf.keras.Model(text_input, output)

    epochs = 1
    steps_per_epoch = 25
    num_train_steps = steps_per_epoch * epochs
    num_train_steps = 5
    num_warmup_steps = int(0.1*num_train_steps)
    num_warmup_steps = 5
    init_lr = 3e-5
    #optimizer = optimization.create_optimizer(init_lr=init_lr,
    #                                          num_train_steps=num_train_steps,
    #                                          num_warmup_steps=num_warmup_steps,
    #                                          optimizer_type='adamw')

    #loss = {'y': tf.keras.losses.SparseCategoricalCrossentropy()}
    #metrics = {'y': tf.metrics.SparseCategoricalAccuracy('accuracy')}

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                         metrics=[tf.metrics.SparseCategoricalAccuracy('accuracy')])
    return model