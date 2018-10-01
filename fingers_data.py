import pandas as pd
import tensorflow as tf

#TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
#TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
#TRAIN_URL = "C:\Users\PC portatil\.keras\datasets\dataset_training.csv"
#TEST_URL = "C:\Users\PC portatil\.keras\datasets\dataset_testing.csv"


CSV_COLUMN_NAMES = ["menique", "medio","indice",
                    "pulgar", "Classes"]
CLASSES = ["a","b","c","d","e","f","i","k","l","m","n","o","p","q","r","t","u","v","w","x","y","es"]



def maybe_download():
    train_path = 'fingersDataset/DataRed.csv'
    test_path = 'fingersDataset/DataRed.csv'
    
    print (train_path)
    return train_path, test_path

def load_data(y_name='Classes'):
    
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES)
    train_1 = train.sample(frac=1, random_state=99)
    train_x, train_y = train_1, train_1.pop(y_name)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES)
    test_1 = test.sample(frac=1, random_state=99)
    test_x, test_y = test_1, test_1.pop(y_name)

    print(train_x.dtypes)
    return (train_x, train_y), (test_x, test_y)

load_data()

def train_input_fn(features, labels, batch_size):

    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using a the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0],[0.0],[0.0],[0.0],[0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Classes')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset