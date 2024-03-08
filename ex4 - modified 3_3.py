import datasets
###################################################
# Exercise 4 - Natural Language Processing 67658  #
###################################################
import numpy as np

# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }

def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion*len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


# Q1
def linear_classification(portion=1.):
    """
    Perform linear classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    tf = TfidfVectorizer(stop_words='english', max_features=1000)
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Add your code here
    tf_x_train, tf_x_test = tf.fit_transform(x_train), tf.fit_transform(x_test)
    logisticRegr = LogisticRegression()
    logisticRegr.fit(tf_x_train, y_train)
    y_pred = logisticRegr.predict(tf_x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Q2
def transformer_classification(portion=1.):
    """
    Transformer fine-tuning.
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    import torch

    class Dataset(torch.utils.data.Dataset):
        """
        Dataset object
        """
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    from datasets import load_metric
    # added trust_remote_code to prevent a warning while running
    metric = load_metric("accuracy", trust_remote_code=True)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    from transformers import Trainer, TrainingArguments
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base', cache_dir=None)
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base',
                                                               cache_dir=None,
                                                               num_labels=len(category_dict),
                                                               problem_type="single_label_classification")

    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Add your code here
    # see https://huggingface.co/docs/transformers/v4.25.1/en/quicktour#trainer-a-pytorch-optimized-training-loop
    # Use the DataSet object defined above. No need for a DataCollator

    x_test = x_test[:10]  # TODO: remove later. temporary for testing speed
    y_test = y_test[:10]  # TODO: remove later. temporary for testing speed
    training_args = TrainingArguments(
        output_dir="results",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,

    )
    # training data
    tokenized_x_train = tokenize_dataset(tokenizer, x_train)  # tokenizing sentences
    train_dataset = Dataset(tokenized_x_train, y_train)
    # test data
    tokenized_x_test = tokenize_dataset(tokenizer, x_test)  # tokenizing sentences
    test_dataset = Dataset(tokenized_x_test, y_test)

    trainer = Trainer(
        model=model.to('cuda'),
        args=training_args,
        train_dataset=train_dataset.to('cuda'),
        eval_dataset=test_dataset.to('cuda'),
        compute_metrics=compute_metrics
    )
    trainer.train()
    eval_predictions = trainer.evaluate(test_dataset)
    # TODO: log loss per epoch!
    print(eval_predictions)
    return eval_predictions['eval_accuracy']


# ####### helper func ###########
def tokenize_dataset(tokenizer, x_dataset):
    tokenized_x_dataset = tokenizer(x_dataset, padding=True, truncation=True, return_tensors="pt")
    return tokenized_x_dataset
# ####### end of helper func ###########


# Q3
def zeroshot_classification(portion=1.):
    """
    Perform zero-shot classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from transformers import pipeline
    from sklearn.metrics import accuracy_score
    import torch
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    clf = pipeline("zero-shot-classification", model='cross-encoder/nli-MiniLM2-L6-H768')
    candidate_labels = list(category_dict.values())

    # Add your code here
    # see https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline
    return


if __name__ == "__main__":
    portions = [0.001, 0.1, 0.5, 1.]
    # Q1
    # print("Logistic regression results:")
    # for p in portions:
    #     print(f"Portion: {p}")
    #     print(linear_classification(p))

    # Q2
    print("\nFinetuning results:")
    for p in portions:
        print(f"Portion: {p}")
        print(transformer_classification(portion=p))

    # Q3
    print("\nZero-shot result:")
    print(zeroshot_classification())

