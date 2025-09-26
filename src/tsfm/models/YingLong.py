import torch
from einops import rearrange
from torch import nn
from transformers import AutoModelForCausalLM

from layers.prune_mask import MaskedLayer
from models.base import PrunableModel
from utils.pruning import merge_weights


class Model(nn.Module, PrunableModel):
    def __init__(self, args):
        super(Model, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            f'qcw2333/YingLong_{args.model_size}',
            trust_remote_code=True,
            torch_dtype='bfloat16',
            # attn_implementation='flash_attention_2',
            device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
        )
        self.pred_len = args.pred_len
        if getattr(args, "input_ensemble", False):
            self.input_ensemble = [l for l in [512, 1024, 2048, 4096] if l <= args.seq_len]
        else:
            self.input_ensemble = []
        # self.num_heads = self.transformers[0].attn.num_heads
        # self.transformer_names = []

    def forward(self, x: torch.FloatTensor = None, labels: torch.FloatTensor = None):
        x = x.squeeze(-1)
        if self.input_ensemble and not self.training:
            logits = 0
            for history in self.input_ensemble:
                x_train = torch.cat((x[:, -history:], -x[:, -history:]), dim=0)

                logits_all = self.model(idx=x_train, future_token=4096)
                logits_all = rearrange(logits_all, '(t b) l c d -> b (l c) d t', t=2)[:, :self.pred_len, [49], :]
                logits += logits_all[..., 0] - logits_all[..., 1].flip(dims=[-1])
            logits = logits / (2 * len(self.input_ensemble))
            return logits.float()
        else:
            predictions = self.model.generate(x, future_token=4096, )
            return predictions[..., [49]].float()

    @property
    def transformers(self):
        return self.model.transformer.h

    def merge_weights_(self):
        dependency_graph = {
            'mlp.swiglu.w2': ({}, {'mlp.swiglu.w1': 1, 'mlp.swiglu.w3': 0}),
            'mlp.swiglu.w1': ({}, {'mlp.swiglu.w2': 1, 'mlp.swiglu.w3': 0}),
            'mlp.swiglu.w3': ({'mlp.swiglu.w1': 1, 'mlp.swiglu.w2': 1}, {}),
        } # 0 is input dimension, 1 is output dimension
        qkvo_names = []

        merge_weights(self.transformers,
                      names=self.transformer_names,
                      qkvo_names=qkvo_names,
                      dependency_graph=dependency_graph,
                      num_heads=self.transformers[0].attn.num_heads,
                      revise_head_num=None, enable_index_add=False,
                      )
        for module in self.model.modules():
            if (module not in self.transformers and isinstance(module, MaskedLayer)
                    and not (hasattr(module, 'merged') and module.merged)):
                merge_weights(module)
