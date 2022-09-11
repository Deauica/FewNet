import warnings

import torch
from typing import Dict

from concern.config import Configurable, State
import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict
import sys

from .learning_rate import *  # mahy


class FewNetOptimizerScheduler(Configurable):
    """ Optimizer Scheduler for FewNet """
    optimizer = State()
    optimizer_args = State(default={})
    learning_rate = State()
    learning_rate_args = State(default={})
    
    def __init__(self, cmd={}, **kwargs):
        super(FewNetOptimizerScheduler, self).__init__(cmd=cmd, **kwargs)
        if "lr" in cmd:
            warnings.warn("Current FewNet do not support pass lr in cmd")
        
    def create_optimizer(self, named_parameters):
        """ create optimizer and its corresponding learning_rate object,
        return optimizer ultimately
        
        Notes:
            - `params_key` should be the bridge between optimizer and scheduler.
            - named_parameters should be generator.
        """
        # optimizer
        named_parameters = list(named_parameters)  # transform generator to list
        
        optimizer_constructor = getattr(torch.optim, self.optimizer)
        optimizer_kwargs = []  # List[Dict]
        named_flags = OrderedDict(
            [(k, False) for k, _ in named_parameters]  # generator
        )
        # fill optimizer_kwargs
        for i, param_dict in enumerate(self.optimizer_args["params_dict"]):
            # for each specified param
            param_key = param_dict["params_key"]
            params = [v for k, v in named_parameters if param_key in k]
            for k, v in named_parameters:
                if param_key in k:
                    assert not named_flags[k], (
                        "Please check your params_key, since {} is added to "
                        "at least two param groups".format(k)
                    )
                    named_flags[k] = True
            
            optimizer_kwargs.append({
                "params": params, **self.optimizer_args["params_dict"][i]
            })
        # for default optimizer
        default_params = [v for k, v in named_parameters if not named_flags[k]]
        optimizer_kwargs.append({"params": default_params, "params_key": "default"})
        # construct
        optimizer = optimizer_constructor(
            optimizer_kwargs, **self.optimizer_args["constructor_args"]
        )
        
        # scheduler
        scheduler_constructor = getattr(sys.modules[__name__], self.learning_rate)
        learning_rate_args = self.learning_rate_args
        constructor_default_args = learning_rate_args["constructor_args"]  # default kwargs
        
        if "default" not in learning_rate_args:
            learning_rate_args["default"] = {
                "constructor": learning_rate_args["constructor"],
                "constructor_args": learning_rate_args["constructor_args"]
            }
        scheduler_kwargs, scheduler_param_list = [], []
        for item in optimizer_kwargs:
            # if not specified, then default is utilized
            param_key = (
                item["params_key"] if item["params_key"] in learning_rate_args
                else "default"
            )
            
            scheduler_param_list.append(param_key)
            
            _t = constructor_default_args.copy()
            _t.update(learning_rate_args[param_key]["constructor_args"])
            scheduler_kwargs.append({
                "constructor": learning_rate_args[param_key]["constructor"],
                "constructor_args": _t
            })
        self.learning_rate = scheduler_constructor(scheduler_kwargs, scheduler_param_list)
        return optimizer


class FewNetScheduler(object):
    def __init__(self, scheduler_kwargs, param_key_list, *args, **kwargs):
        self.param_key_list = param_key_list  # order is important
        
        self._scheduler_kwargs = scheduler_kwargs
        self.schedulers = []
        
        for _kwargs in self._scheduler_kwargs:
            constructor = getattr(sys.modules[__name__], _kwargs["constructor"])
            constructor_args = _kwargs["constructor_args"]
            self.schedulers.append(
                constructor(**constructor_args)
            )
    
    def get_learning_rate(self, *args, **kwargs):
        return [scheduler.get_learning_rate(*args, **kwargs)
                for scheduler in self.schedulers]


class OptimizerScheduler(Configurable):
    optimizer = State()
    optimizer_args = State(default={})
    learning_rate = State(autoload=False)

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.load('learning_rate', cmd=cmd, **kwargs)
        if 'lr' in cmd:
            self.optimizer_args['lr'] = cmd['lr']

    def create_optimizer(self, parameters):
        parameters = [item[-1] for item in parameters]  #
            
        optimizer = getattr(torch.optim, self.optimizer)(
                parameters, **self.optimizer_args)
        if hasattr(self.learning_rate, 'prepare'):
            self.learning_rate.prepare(optimizer)
        return optimizer
