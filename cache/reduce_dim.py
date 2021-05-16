import fasttext
import fasttext.util


def reduce_size():
    ft = fasttext.load_model('cc.ru.300.bin')
    fasttext.util.reduce_model(ft, 100)
    ft.save_model('cc.ru.100.bin')


if __name__ == "__main__":
    reduce_size()
