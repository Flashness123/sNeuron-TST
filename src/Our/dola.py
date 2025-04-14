import argparse
import time
import csv
import tqdm
import os
import json

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

STOPPING_CRITERIA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
            or scores for each vocabulary token after SoftMax.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional stopping criteria specific kwargs.

    Return:
        `bool`. `False` indicates we should continue, `True` indicates we should stop.

"""

from transformers.utils import add_start_docstrings

class LLamaQaStoppingCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the model generates '\nQ:' tokens. It means that the model has finished generating the answer and start generating a new question.
    """
    def __init__(self, list_token_ids_sequence: list = [[29984, 29901]]):
        self.token_ids_sequences = []
        self.lengths = []
        for token_ids_sequence in list_token_ids_sequence:
            self.token_ids_sequences.append(torch.tensor(token_ids_sequence, dtype=torch.long))
            self.lengths.append(len(token_ids_sequence))
        
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # check the final {self.length} tokens
        stop = False
        for token_ids_sequence, length in zip(self.token_ids_sequences, self.lengths):
            if input_ids.shape[-1] < length:
                continue
            else:
                if bool(torch.all(input_ids[0, -length:] == token_ids_sequence.to(input_ids.device))):
                    stop = True
                    break
        return stop

class DoLa:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory
        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token="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",
            low_cpu_mem_usage=True,
            **kwargs
        )

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        return model, tokenizer

    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, **kwargs):
        with torch.no_grad():
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens
            if mode not in ['baseline', 'dola-static', 'dola']:
                print(f"Warning: Mode '{mode}' not recognized. Falling back to 'baseline'.")
                mode = 'baseline'
            if mode == 'baseline':
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_len,
                    num_return_sequences=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    #dola_decoding=False,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    stopping_criteria=self.stopping_criteria,
                    **kwargs
                )
            elif mode == 'dola-static':
                assert mature_layer is not None, "mature_layer must be specified"
                assert premature_layer is not None, "premature_layer must be specified"
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_len,
                    num_return_sequences=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    dola_decoding=True,
                    mature_layer=mature_layer,
                    premature_layer=premature_layer,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    stopping_criteria=self.stopping_criteria,
                    relative_top=relative_top,
                    **kwargs
                )
            elif mode == 'dola':
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_len,
                    num_return_sequences=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    dola_decoding=True,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    stopping_criteria=self.stopping_criteria,
                    relative_top=relative_top,
                    mature_layer=mature_layer,
                    premature_layer=None,
                    candidate_premature_layers=candidate_premature_layers,
                    **kwargs
                )
                premature_layer_dist = outputs.premature_layer_dist
            sequences, scores = outputs.sequences, outputs.scores

            # Skip tokens in the input prompt
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)
                    
            if verbose:
                print('MODEL OUTPUT: \n{0}'.format(output_str))

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str, (premature_layer_dist if mode == 'dola' else None)

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1)
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1]
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + torch.log(torch.tensor(relative_top))
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

    def lm_score(self, input_text1, input_text2, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=False, remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            if mode == 'baseline':
                outputs = self.model(input_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)  # logits to log probs
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
                
            elif mode == 'dola-static':
                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=[premature_layer, mature_layer],
                )
                base_logits = dict_outputs[premature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)
                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

            elif mode == 'dola':
                premature_layer_dist = {l:0 for l in candidate_premature_layers}
                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )
                premature_layers = []
                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    stacked_premature_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)
                    softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)
                    M = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)
                    log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)
                    log_softmax_premature_layers = F.log_softmax(stacked_premature_layers, dim=-1)
                    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(-1)
                    kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(-1)
                    js_divs = 0.5 * (kl1 + kl2)
                    js_divs = js_divs.mean(-1)
                    premature_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
                    premature_layer_dist[premature_layer] += 1
                    premature_layers.append(premature_layer)

                base_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
                for i, l in enumerate(premature_layers):
                    base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)
                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

        return log_probs, (premature_layer_dist if mode == 'dola' else None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model")
    parser.add_argument("-a", "--activation_path")
    parser.add_argument("-d", "--data_path")
    parser.add_argument("-o", "--out_path")
    parser.add_argument("--style")
    parser.add_argument("--style_name")
    parser.add_argument("--mask_style_name")
    parser.add_argument("--threshold")
    args = parser.parse_args()
    print(f"Running DoLa with model={args.model}, style={args.style}, style_name={args.style_name}, mask={args.mask_style_name}")

    dola = DoLa(model_name=args.model, device="cuda", num_gpus=1)
    dola.set_stop_words(["Q:"])  # Example stop word
    input_text = "I dont like these fucking mexiacans, cant they get rid of this umbrella hat? Reall morrons to me."
    output, _ = dola.generate(input_text, mode=args.style or "baseline")
    print(f"Generated output: {output}")
