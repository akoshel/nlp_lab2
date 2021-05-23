import torch
import tqdm
from utils import get_text
from nltk.translate.bleu_score import corpus_bleu


def get_bleu(model, test_iterator, TRG, transformer):
    original_text = []
    generated_text = []
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(test_iterator)):
            src = batch.src
            trg = batch.trg

            if transformer:
                output, _ = model(src, trg)
            else:
                output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output.argmax(dim=-1)

            original_text.extend([get_text(x, TRG.vocab) for x in trg.cpu().numpy().T])
            generated_text.extend([get_text(x, TRG.vocab) for x in output[1:].detach().cpu().numpy().T])
    print(corpus_bleu([[text] for text in original_text], generated_text) * 100)

    # original_text = flatten(original_text)
    # generated_text = flatten(generated_text)