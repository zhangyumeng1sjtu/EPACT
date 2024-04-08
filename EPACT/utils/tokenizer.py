import itertools
from typing import Sequence, List

import torch


PROTEIN_SEQ_TOKS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                    'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


class IUPACTokenizer(object):

    def __init__(
        self,
        standard_toks: Sequence[str] = PROTEIN_SEQ_TOKS,
        prepend_toks: Sequence[str] = ("<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        
        self.all_toks = self.prepend_toks + self.standard_toks + self.append_toks
        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.sep_idx = self.get_idx("<sep>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_toks = ['<eos>', '<unk>',
                                 '<pad>', '<sep>', '<cls>', '<mask>']
        self.speical_token_idxes = [self.tok_to_idx[tok]
                                    for tok in self.all_special_toks]
        self.standard_token_idxes = [self.tok_to_idx[tok]
                                    for tok in self.standard_toks]
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, idx):
        return self.all_toks[idx]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]

    def mask_token(self, tokens, mask_prob=0.15, beta_token_mask=None, beta_mask_prob=0.25):
        labels = tokens.clone()
        prob_matrix = torch.full(labels.shape, mask_prob)
        
        if beta_token_mask is not None:
            prob_matrix.masked_fill_(beta_token_mask, value=beta_mask_prob)
            
        special_token_mask = torch.isin(
            labels, torch.tensor(self.speical_token_idxes))
        prob_matrix.masked_fill_(special_token_mask, value=0.0)

        masked_idxes = torch.bernoulli(prob_matrix).bool()
        tokens[masked_idxes] = self.mask_idx
        return tokens, labels
    