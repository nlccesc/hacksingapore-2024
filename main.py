import tensorflow as tf
from init import init
from preprocessor import preprocess_data
from model import create_model, compile_and_train_model
from feature_converter import convert_examples_to_features
from mapper import map_example_to_dict

# Initialize data
train, valid, test = init()

# Preprocess data
X_train, y_train, X_valid, y_valid, X_test, y_test = preprocess_data(train, valid, test)

# Create model and tokenizer
model, tokenizer = create_model()

train_features = convert_examples_to_features(X_train, y_train, tokenizer)
valid_features = convert_examples_to_features(X_valid, y_valid, tokenizer)

# Prepare our training dataset
train_ds = tf.data.Dataset.from_tensor_slices((train_features['input_ids'], train_features['attention_mask'], train_features['token_type_ids'], y_train)).map(map_example_to_dict).shuffle(100).batch(32)

# Prepare our validation dataset
valid_ds = tf.data.Dataset.from_tensor_slices((valid_features['input_ids'], valid_features['attention_mask'], valid_features['token_type_ids'], y_valid)).map(map_example_to_dict).batch(64)

# Compile and train the model
compile_and_train_model(model, train_ds, valid_ds)