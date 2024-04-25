from transformers import InputFeatures

def convert_examples_to_features(examples, labels, tokenizer, max_length=128, task=None, label_list=None):
    features = []
    for i, example in enumerate(examples):
        inputs = tokenizer.encode_plus(
            example,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True
        )

        feature = InputFeatures(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids'],
            label=labels[i]
        )

        features.append(feature)

    return features