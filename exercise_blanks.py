import datetime
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=True):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    w2v_list = []
    for leaf in sent.get_leaves():  # for each word in the sentence, add to a growing list the one-hot vector for it
        if leaf.text[0] in word_to_vec.keys():
            if word_to_vec[leaf.text[0]].shape == (0,):
                w2v_list.append(np.zeros(embedding_dim))
            else:
                w2v_list.append(word_to_vec[leaf.text[0]])
        else:
            w2v_list.append(np.zeros(embedding_dim))
    if len(w2v_list) > 0:
        w2v = np.vstack(w2v_list)  # stack the one_hot vectors
        return np.sum(w2v, 0) / len(w2v)  # get the average of the one_hot vectors
    return np.zeros(embedding_dim)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    return np.array([0] * ind + [1] + [0] * (size - ind - 1))


def get_one_hots(size):
    result = []
    for i in range(size):
        result.append(get_one_hot(size, i))
    return result


def average_one_hots(sent, word_to_ind, one_hots):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    one_hots_list = []
    for leaf in sent.get_leaves():  # for each word in the sentence, add to a growing list the one-hot vector for it
        one_hots_list.append(one_hots[word_to_ind[leaf.text[0]]])
    one_hots = np.vstack(one_hots_list)  # stack the one_hot vectors
    return np.sum(one_hots, 0) / len(one_hots)  # get the average of the one_hot vectors


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    result = {}
    for i in range(len(words_list)):  # for each word, add an entry with it as a key and its index as its value
        result[words_list[i]] = i
    return result


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    w2v_sequence = []
    leaves = sent.get_leaves()
    for i in range(seq_len):
        if i < len(leaves):
            if leaves[i].text[0] in word_to_vec.keys():
                w2v_sequence.append(word_to_vec[leaves[i].text[0]])
            else:
                w2v_sequence.append(np.zeros(embedding_dim))
        else:
            w2v_sequence.append(np.zeros(embedding_dim))
    return np.array(w2v_sequence)


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs, data_role, data_type):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs
        self.data_role = data_role
        self.data_type = data_type
        self.sentence_embeddings, self.sentence_labels = self._get_sentence_labeled_embeddings()  # instead of keeping
        # the sentences as data, We store All the labels in the GPU memory to make things more efficient

    def _get_sentence_labeled_embeddings(self):  # returns the embeddings and labels for the data
        if os.path.exists(self.data_type + "_" + self.data_role + "_huge_tensor.pt") and \
         os.path.exists(self.data_type + "_" + self.data_role + "_tiny_tensor.pt"):  # We check for existence of the
            # saved data, and if it exists We load it
            embeddings = torch.load(self.data_type + "_" + self.data_role + "_huge_tensor.pt")
            labels = torch.load(self.data_type + "_" + self.data_role + "_tiny_tensor.pt")
        else:  # Otherwise, We do the conversion process
            embedding_blocks = []  # Initialize the block lists
            label_blocks = []
            embedding_block = []  # Initialize the blocks
            label_block = []
            for sent in self.data:  # For each sentence, We apply the embedding function
                new_embedding = self.sent_func(sent, **self.sent_func_kwargs)
                # if new_embedding.shape != (300,):
                #    new_embedding = np.zeros(300)
                embedding_block.append(new_embedding)
                label_block.append(sent.sentiment_class)
                if (len(label_block) % 100) == 0:  # Every 100 sentences, to prevent a huge list from forming,
                    # We add the 100 sentence block to the block list
                    embedding_blocks.append(embedding_block)
                    label_blocks.append(label_block)
                    embedding_block = []
                    label_block = []
                    print("Processed ", len(label_blocks), " sentence blocks!")
            embedding_blocks_arr = np.array(embedding_blocks)
            embeddings_raw = torch.tensor(embedding_blocks_arr)  # convert embeddings to tensor
            embeddings = torch.cat((embeddings_raw.view(-1, len(embedding_blocks[0][0])), torch.tensor(embedding_block)), dim=0)  # Add the remaining non-full block to the tensor
            labels = torch.cat((torch.tensor(label_blocks).view(-1), torch.tensor(label_block)), dim=0)  # Labels tensor
            torch.save(embeddings, self.data_type + "_" + self.data_role + "_huge_tensor.pt")  # Save Our results for future use
            torch.save(labels, self.data_type + "_" + self.data_role + "_tiny_tensor.pt")
        return embeddings.to(get_available_device()), labels.to(get_available_device())  # Convert to device and return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sent = self.data[idx]
        # sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        # sent_label = sent.sentiment_class
        return self.sentence_embeddings[idx], self.sentence_labels[idx]  # We changed this function to return the appropriate data


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list),
                                     "one_hots": get_one_hots(len(words_list))}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs, k, data_type) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape




# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        self.lstm = nn.LSTM()

    def forward(self, text):
        return

    def predict(self, text):
        return


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.layer = nn.Linear(embedding_dim, 1, dtype=torch.float64)  # initialize one affine layer
        self.activation = nn.Sigmoid()  # initialize sigmoid activation

    def forward(self, x):
        return self.layer(x)  # return the non-activated result of the layer for one input

    def predict(self, x):
        output = self.activation(self.layer(x))
        result = torch.zeros_like(output)
        result[output > 0.6] = 1
        result[(output >= 0.4) & (output <= 0.6)] = -1
        return result


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    all_preds = len(preds)
    rounded_preds = np.round(preds)  # round to either 0 or 1
    accurate_preds = np.count_nonzero(rounded_preds == y)  # check how many correct predictions
    return accurate_preds/all_preds


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    model.train()  # Set model to train mode
    for inputs, targets in data_iterator:  # Iterate over the data
        optimizer.zero_grad()  # Zero the gradient
        partial_outputs = model(inputs.to(torch.float64))  # Get the values the model returns for the data (representing a prediction of positive sentiment)
        complementary_outputs = -partial_outputs[:, 0].unsqueeze(1)  # Get the complement output for negative sentiments
        outputs = torch.cat((complementary_outputs, partial_outputs[:, 0].unsqueeze(1)), dim=1)  # Connect the two
        loss = criterion(outputs, targets.to(torch.int64))  # Calculate loss
        loss.backward()  # Calculate gradient of loss
        optimizer.step()  # Move the parameters of the model according to the gradient
    return evaluate(model, data_iterator, criterion)  # We evaluate What We've trained and return the accuracy and loss


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    model.eval()  # Similarly, set model to evaluation mode
    total_loss = 0  # Initialize values
    correct_predictions = 0
    total_examples = 0
    with torch.no_grad():  # No gradient for ease of calculation
        for inputs, targets in data_iterator:  # Iterate over the data
            partial_outputs = model(inputs.to(torch.float64))  # Get the positive prediction
            complementary_outputs = -partial_outputs[:, 0].unsqueeze(1)  # Complementary negative prediction
            outputs = torch.cat((complementary_outputs, partial_outputs[:, 0].unsqueeze(1)), dim=1)  # Combine
            loss = criterion(outputs, targets.to(torch.int64))  # Calculate loss
            _, predicted = torch.max(outputs, 1)  # Count positive predictions
            correct_predictions += torch.eq(predicted, targets).to(torch.int).sum().item()  # Count correct predictions
            total_examples += targets.size(0)  # Count total data elements seen in this iteration
            total_loss += loss.item() * inputs.size(0)  # Calculate cumulative loss
    average_loss = total_loss / total_examples  # Calculate average loss
    accuracy = correct_predictions / total_examples  # Calculate accuracy
    return average_loss, accuracy  # Return values


def get_predictions_for_data(model, data_iter):
    """
    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    result = torch.empty((0,)).to(get_available_device())
    for inputs, targets in data_iter:
        result = torch.cat((result, model.predict(inputs.to(torch.float64))))
    return result


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    device = get_available_device()  # Get available device
    model = model.to(device)  # Convert model to device
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Initialize ADAM algorithm
    criterion = nn.CrossEntropyLoss().to(device)  # Initialize Cross Entropy Loss and convert to device
    training_data_iterator = data_manager.get_torch_iterator(TRAIN)  # Get training data iterator
    validation_data_iterator = data_manager.get_torch_iterator(VAL)  # Get validation data iterator
    training_losses = []  # Initialize losses and accuracies for training and validation
    training_accuracies = []
    validated_losses = []
    validated_accuracies = []
    initial_training_loss, initial_training_accuracy = evaluate(model, training_data_iterator, criterion)  # Calculate initial losses and accuracies and display
    initial_validated_loss, initial_validated_accuracy = evaluate(model, validation_data_iterator, criterion)
    print("Starting training, initial training loss: ", initial_training_loss, ", training accuracy: ", initial_training_accuracy,
              ", validation loss: ", initial_validated_loss, ", validation accuracy: ", initial_validated_accuracy, "!")
    for i in range(n_epochs):  # Perform each training epoch
        print("Starting training epoch ", i, "...")
        training_loss, training_accuracy = train_epoch(model, training_data_iterator, optimiser, criterion)  # Train the model
        training_losses += [training_loss]  # Add accuracy and loss to list
        training_accuracies += [training_accuracy]
        validated_loss, validated_accuracy = evaluate(model, validation_data_iterator, criterion)  # Validate the model
        validated_losses += [validated_loss]  # Add accuracy and loss to list
        validated_accuracies += [validated_accuracy]
        print("Finished training epoch ", i,  # Print training epoch results
              "; training loss: ", training_loss, ", training accuracy: ", training_accuracy,
              ", validation loss: ", validated_loss, ", validation accuracy: ", validated_accuracy, "!")
    return training_losses, training_accuracies, validated_losses, validated_accuracies  # Return losses and accuracies for training and validation


def train_log_linear_with_one_hot(tree_bank_manager, epoch_count, learning_rate, weight_decay):
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    log_linear_model = LogLinear(len(tree_bank_manager.word_list))  # initialize log-linear model
    train_losses, train_accuracies, valuated_losses, valuated_accuracies = \
        train_model(log_linear_model, tree_bank_manager.one_hot_average_data_manager, epoch_count, learning_rate, weight_decay)
    # train the model, return the model along with losses and accuracies
    return log_linear_model, train_losses, train_accuracies, valuated_losses, valuated_accuracies  # return values and model


def train_log_linear_with_w2v(tree_bank_manager, epoch_count, learning_rate, weight_decay):
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    log_linear_model = LogLinear(W2V_EMBEDDING_DIM)  # initialize log-linear model
    train_losses, train_accuracies, valuated_losses, valuated_accuracies = \
        train_model(log_linear_model, tree_bank_manager.w2v_average_data_manager, epoch_count, learning_rate, weight_decay)
    # train the model, return the model along with losses and accuracies
    return log_linear_model, train_losses, train_accuracies, valuated_losses, valuated_accuracies  # return values and model


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    return


##############################OUR_STUFF#################################


class TreeBankManager:
    def __init__(self, batch_size):  # Initializes the basic structures We'll need
        self.tree_bank = data_loader.SentimentTreeBank()
        self.training_set = self.tree_bank.get_train_set()
        self.validation_set = self.tree_bank.get_validation_set()
        self.test_set = self.tree_bank.get_test_set()
        self.word_count = self.tree_bank.get_word_counts()
        self.word_list = list(self.word_count.keys())
        self.one_hot_average_data_manager = DataManager(data_type=ONEHOT_AVERAGE, batch_size=batch_size)
        self.w2v_average_data_manager = DataManager(data_type=W2V_AVERAGE, batch_size=batch_size)


if __name__ == '__main__':
    tree_bank_managerr = TreeBankManager(64)  # Initialise the TreeBankManager with batch_size of 64
    # train_log_linear_with_one_hot(tree_bank_managerr, 20, 0.01, 0.001)  # Train a log linear model with the specified hyperparameters
    modell, _, _, _, _ = train_log_linear_with_w2v(tree_bank_managerr, 20, 0.01, 0.001)  # Train a log linear model with the specified hyperparameters
    print(get_predictions_for_data(modell, tree_bank_managerr.w2v_average_data_manager.get_torch_iterator(TEST)))
    # train_log_linear_with_w2v()
    # train_lstm_with_w2v()