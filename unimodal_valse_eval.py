import torch
import numpy as np
import sys
from tqdm import tqdm

from config import DATA
from read_foil_dataset import read_foils

def load_model(which):
    device = 'cuda'
    if which == 'gpt2':
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        model_id = 'gpt2-large'
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        max_length = model.config.n_positions
    elif which == 'gpt1':
        from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
        model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt').to(device)
        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        max_length = model.config.n_positions
    else:
        raise NotImplementedError("Implemented models are gpt2 and gpt1.")
    return model, device, tokenizer, max_length

def compute_ppl(test_sentence):
    ''' Compute the perplexity of a sentence. '''
    encodings = tokenizer(test_sentence, return_tensors='pt')

    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * stride

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / i)
    return ppl

which = sys.argv[1]
print(f"Running experiments with {which} unimodal model.")
model, device, tokenizer, max_length = load_model(which)

for instrument, foils_path in DATA.items():

    foils_data = read_foils(foils_path)

    count, foil_detected = 0, 0
    ppl_correct = []
    ppl_foils = []

    tested_foil_ids = {}

    for foil_id, foil in tqdm(foils_data.items()):
        caption_fits = foil['mturk']['caption']

        if caption_fits >= 2:  # MTURK filtering!

            test_sentences = [foil["caption"], foil["foil"]]
            stride = 3
            max_length = model.config.n_positions

            ppls = []

            for i, test_sentence in enumerate(test_sentences):
                ppl = compute_ppl(test_sentence).to('cpu')
                ppls.append(ppl)

                if i == 0:
                    ppl_correct.append(ppl)
                else:
                    ppl_foils.append(ppl)

            if ppls[0] < ppls[1]:  # ppl smaller is better
                foil_detected += 1

            count += 1

    print(f'{instrument}: {which} could detect {foil_detected / count * 100:.2f}% of foils (pairwise).')
    print(f'The average perplexity for correct sentences is {np.mean(ppl_correct)} and for foils is {np.mean(ppl_foils)}.')
