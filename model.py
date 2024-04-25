import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

def create_model():
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    return model, tokenizer

def compile_and_train_model(model, train_ds, valid_ds):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                  metrics=[tf.keras.metrics.BinaryAccuracy('accuracy')])

    model.fit(train_ds, epochs=2, validation_data=valid_ds)