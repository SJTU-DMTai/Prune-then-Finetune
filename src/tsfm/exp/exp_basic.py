import os
import warnings
from collections import OrderedDict

import torch
import typing

from torch import optim, nn

from utils.tools import remove_state_key_prefix


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.flag_load_state_dict = True
        self.ddp_model = False
        if self.args.use_multi_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
            self.model = self._build_model().to(self.device)
        else:
            self.device = self._acquire_device()
            self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    @property
    def _model(self):
        if self.ddp_model:
            return self.model.module
        return self.model

    def state_dict(self, destination: typing.OrderedDict[str, torch.Tensor]=None) -> typing.OrderedDict[str, torch.Tensor]:
        r"""Returns a dictionary containing a whole state of the module and the state of the optimizer.

        Returns:
            dict:
                a dictionary containing a whole state of the module and the state of the optimizer.
        """
        if hasattr(self.args, 'save_opt') and self.args.save_opt:
            if destination is None:
                destination = OrderedDict()
                destination._metadata = OrderedDict()
            destination['model'] = self._model.state_dict()
            if hasattr(self.args, 'freeze') and self.args.freeze:
                for k, v in self._model.named_parameters():
                    if not v.requires_grad:
                        destination['model'].pop(k)
            destination['model_optim'] = self.model_optim.state_dict() if self.model_optim is not None else None
            return destination
        else:
            destination = self._model.state_dict()
            if hasattr(self.args, 'freeze') and self.args.freeze:
                for k, v in self._model.named_parameters():
                    if not v.requires_grad:
                        destination.pop(k)
            return destination

    def load_buffer(self, model: nn.Module, state_dict: typing.OrderedDict[str, torch.Tensor]):
        for name, v in state_dict.items():
            try:
                if model.get_buffer(name) is None:
                    model.get_submodule('.'.join(name.split('.')[:-1])).register_buffer(
                        name.split('.')[-1], v, persistent=True
                    )
            except AttributeError:
                pass

    def load_state_dict(self, state_dict: typing.OrderedDict[str, torch.Tensor], model=None, strict=False):
        if model is None:
            model = self._model
        if 'model_optim' not in state_dict:
            state_dict = remove_state_key_prefix(state_dict, model)
            model.load_state_dict(state_dict, strict=strict)
            self.load_buffer(model, state_dict)
        else:
            for k, v in state_dict.items():
                if k == 'model':
                    v = remove_state_key_prefix(v, model)
                    model.load_state_dict(v, strict=strict)
                    self.load_buffer(model, v)
                elif hasattr(self, k) and getattr(self, k) is not None:
                    if isinstance(getattr(self, k), optim.Optimizer):
                        assert len(getattr(self, k).param_groups) == len(v['param_groups'])
                        try:
                            getattr(self, k).load_state_dict(v)
                        except ValueError:
                            warnings.warn(f'{k} has different state dict from the checkpoint. '
                                          f'Trying to save all states of frozen parameters...')
                            assert k == 'model_optim'
                            self.model_optim = self._select_optimizer(filter_frozen=False, return_self=False)
                            self.model_optim.load_state_dict(v)
                    elif isinstance(getattr(self, k), dict):
                        setattr(self, k, v)
                    else:
                        getattr(self, k).load_state_dict(v, strict=strict)
        return model

    def load_checkpoint(self, load_path=None, model=None, strict=False):
        if self.flag_load_state_dict:
            return self.load_state_dict(torch.load(load_path, map_location=self.device), model, strict=strict)
        else:
            ckpt = torch.load(load_path, map_location=self.device)
            if int(os.environ.get("LOCAL_RANK", "-1")) >= 0:
                self.model.module = ckpt
            else:
                self.model = ckpt

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda')
            print('Use GPU: cuda')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self, setting):
        pass

    def train(self, setting):
        pass

    def test(self, setting, test=0):
        pass
