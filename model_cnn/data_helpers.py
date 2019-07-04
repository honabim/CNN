import numpy as np
import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(greet_data_file, find_data_file,bye_data_file,affirmative_data_file,negative_data_file):
#def load_data_and_labels(greet_data_file,bye_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    find_examples = list(open(find_data_file, "r", encoding='utf-8').readlines())
    find_examples = [s.strip() for s in find_examples]
    # try:
    #     l=list(open(greet_data_file, "r", encoding='utf-8')
    # except IOError:
    #     print('---------------------------------------')
    #     print('l=',len(l))
    #     print('---------------------------------------')
    greet_examples = list(open(greet_data_file, "r", encoding='utf-8').readlines())
    print('greet_examples:',greet_examples)

    greet_examples = [s.strip() for s in greet_examples]
    bye_examples = list(open(bye_data_file, "r", encoding='utf-8').readlines())
    bye_examples = [s.strip() for s in bye_examples]


    affirmative_examples = list(open(affirmative_data_file, "r", encoding='utf-8').readlines())
    affirmative_examples = [s.strip() for s in affirmative_examples]

    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = find_examples + greet_examples+bye_examples+affirmative_examples+negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    find_labels = [[0,0,0,0,1] for _ in find_examples]
    greet_labels = [[0,0,0,1,0] for _ in find_examples]
    bye_labels = [[0,0,1,0,0] for _ in bye_examples]
    affirmative_labels = [[0,1,0,0,0] for _ in affirmative_examples]
    negative_labels=[[1,0,0,0,0] for _ in negative_examples]
    y = np.concatenate([find_labels, greet_labels,bye_labels,affirmative_labels,negative_labels], 0)


    # greet_examples = list(open(greet_data_file, "r", encoding='utf-8').readlines())
    # greet_examples = [s.strip() for s in greet_examples]

    # bye_examples = list(open(bye_data_file, "r", encoding='utf-8').readlines())
    # bye_examples = [s.strip() for s in bye_examples]

    # # Split by words
    # x_text = greet_examples +bye_examples
    # x_text = [clean_str(sent) for sent in x_text]
    # # Generate labels
    # greet_labels = [[0,1] for _ in greet_examples]
    # bye_labels = [[1,0] for _ in bye_examples]
    # y = np.concatenate([greet_labels,bye_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
