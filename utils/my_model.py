import tensorflow_hub as hub
import tensorflow as tf
import os

from keras import optimizers
from official.nlp import optimization
import tensorflow_text as text

def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.getenv('TC_BERT_PREPROCESSOR_DIR',"cointegrated/LaBSE-en-ru"))

def create_new_model(value_count):
    if os.getenv('TC_USE_HUBBLE_HUG','0')=='1':
        from transformers import TFAutoModelForSequenceClassification
        model = TFAutoModelForSequenceClassification.from_pretrained(os.getenv('TC_BERT_ENCODER_DIR',"cointegrated/LaBSE-en-ru"),num_labels=value_count)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-3))
    else:
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(os.getenv('TC_BERT_PREPROCESSOR_DIR',os.path.dirname(os.path.realpath(__file__)) +'/../bert/preprocessor'), name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(os.getenv('TC_BERT_ENCODER_DIR',os.path.dirname(os.path.realpath(__file__)) +'/../bert/encoder'), trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        output = tf.keras.layers.Dense(units=value_count,kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.,stddev=1.), name='y', activation='softmax')(net)
        model=tf.keras.Model(text_input, output)

        model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                             metrics=[tf.metrics.SparseCategoricalAccuracy('accuracy')])
    return model