import torch
import tqdm
from utils import get_text
from nltk.translate.bleu_score import corpus_bleu
from loguru import logger


def get_bleu(model, test_iterator, TRG, transformer):
    original_text = []
    generated_text = []
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(test_iterator)):
            src = batch.src
            trg = batch.trg

            if transformer:
                src = src.permute(1, 0)
                trg = trg.permute(1, 0)
                output, _ = model(src, trg)
                output = output.permute(1, 0, 2)
            else:
                output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]
            if transformer:
                output = output.argmax(dim=-1).permute(1, 0)
            else:
                output = output.argmax(dim=-1)

            original_text.extend([get_text(x, TRG.vocab) for x in trg.cpu().numpy()])
            generated_text.extend([get_text(x, TRG.vocab) for x in output[:].detach().cpu().numpy()])

    print(corpus_bleu([[text] for text in original_text], generated_text) * 100)