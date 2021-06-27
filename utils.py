def flatten(l):
    return [item for sublist in l for item in sublist]


def remove_tech_tokens(mystr, tokens_to_remove=['<eos>', '<sos>', '<unk>', '<pad>']):
    return [x for x in mystr if x not in tokens_to_remove]


def get_text(x, TRG_vocab):
    text = [TRG_vocab.itos[token] for token in x]
    try:
        end_idx = text.index('<eos>')
    except ValueError:
        if '.' in text:
            end_idx = text.index('.') + 1
        else:
            end_idx = len(text) - 1
    text = text[:end_idx]
    text = (text)
    if len(text) < 1:
        text = []
    return text


def generate_translation(src, trg, model, TRG_vocab, SRC_vocab, transformer=False):
    model.eval()
    if transformer:
        src = src.permute(1, 0)
        trg = trg.permute(1, 0)
    if transformer:
        output, _ = model(src, trg)
        output = output.permute(1, 0, 2)
    else:
        output = model(src, trg, 0)  # turn off teacher forcing
    output = output.argmax(dim=-1).cpu().numpy()
    src_text = get_text(list(src.permute(1, 0)[:, 0].cpu().numpy()), SRC_vocab)
    original = get_text(list(trg.permute(1, 0)[:, 0].cpu().numpy()), TRG_vocab)
    generated = get_text(list(output[:, 0]), TRG_vocab)
    print('Source: {}'.format(' '.join(src_text)))
    print('Original: {}'.format(' '.join(original)))
    print('Generated: {}'.format(' '.join(generated)))
    print()
