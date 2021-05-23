from torchtext.legacy.datasets import TranslationDataset, Multi30k
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from nltk.tokenize import WordPunctTokenizer
import pickle


tokenizer_W = WordPunctTokenizer()
def tokenize(x, tokenizer=tokenizer_W):
    return tokenizer.tokenize(x.lower())


def get_dataset(path_do_data: str, transformer: bool) -> TabularDataset:

    SRC = Field(tokenize=tokenize,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=transformer)

    TRG = Field(tokenize=tokenize,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=transformer,
                )

    dataset = TabularDataset(
        path=path_do_data,
        format='tsv',
        fields=[('trg', TRG), ('src', SRC)]
    )
    return SRC, TRG, dataset


def split_data(dataset: TabularDataset, train_size: float, valid_size: float, test_size: float):
    train_data, valid_data, test_data = dataset.split(split_ratio=[train_size, valid_size, test_size])
    return train_data, valid_data, test_data

def _len_sort_key(x):
    return len(x.src)

def save_vocab(vocab, path):
    import pickle
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()