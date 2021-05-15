from torchtext.legacy.datasets import TranslationDataset, Multi30k
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from nltk.tokenize import WordPunctTokenizer


tokenizer_W = WordPunctTokenizer()
def tokenize(x, tokenizer=tokenizer_W):
    return tokenizer.tokenize(x.lower())


def get_dataset(path_do_data: str) -> TabularDataset:

    SRC = Field(tokenize=tokenize,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    TRG = Field(tokenize=tokenize,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

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