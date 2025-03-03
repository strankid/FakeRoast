'''
  why new version? Previous version was more of a script with huge technical debt.
  goals
      - modularize the script .. separate the roast stuff from the model parser
      - run by sparsity
      - apply sparisty to only compressed modules
      - add a grad_scaler to roast-array (new logic to be implemeted next)
      - return a summary of what is compressed and how

'''
import torch
import copy
import sys
sys.path.append('Synaptic-Flow')
import Layers

try:
    from .FakeRoast import *
except:
    from FakeRoast import *

# general functions
NONE=0
INFO=1
DEBUG=2

def get_module_params(target_attr):
    ns = 0
    for p in target_attr.parameters():
        if p.requires_grad :
            ns += p.numel()
    return ns

class ModelParser:

    def __init__(self):
        self.verbose = NONE
        pass
  
    def lambda_init(self, state_dict):
        return state_dict

    def lambda_func(self, state_dict):
        return state_dict

    def lambda_next(self, state_dict):
        return state_dict

    def run(self, name, model, state_dict):
        state_dict['model'] = model
        state_dict = self.lambda_init(state_dict)
        for attr in dir(model):
            target_attr = getattr(model, attr)
            state_dict['target_attr'] = target_attr
            state_dict['name'] = attr
            state_dict['model'] = model
            state_dict = self.lambda_func(state_dict)
        for name, immediate_child_module in  model.named_children():
            target_attr = immediate_child_module
            state_dict['target_attr'] = target_attr
            state_dict['name'] = name
            state_dict['model'] = model
            state_dict = self.lambda_func(state_dict)
            state_dict = self.lambda_next(state_dict)
            self.run(name, immediate_child_module, state_dict)
        

class ModelPrinter(ModelParser):
    def __init__(self, model):
        super(ModelPrinter, self).__init__()
        self.model = model

    def lambda_func(self, state_dict):
        print("--->", type(state_dict['target_attr']), isinstance(state_dict['target_attr'], torch.nn.Module))
        return state_dict

    def process(self):
        self.run("model", self.model, {})

class Roastable:
    def __init__(self, module_limit_size=None, verbose=NONE):
        self.LINEAR = [torch.nn.Linear, torch.nn.modules.Linear, Layers.layers.Linear]
        self.CONV2D = [torch.nn.Conv2d, torch.nn.modules.Conv2d, Layers.layers.Conv2d]
        self.EMBEDDING = [torch.nn.Embedding, torch.nn.modules.Embedding]
        self.module_limit_size=module_limit_size
        self.verbose=verbose

    def is_linear(self, attr):
        return type(attr) in self.LINEAR

    def is_conv2d(self, attr):
        return type(attr) in self.CONV2D

    def is_embedding(self, attr):
        return type(attr) in self.EMBEDDING

    def roastable(self, attr):

        #if the module has been marked as do not roast
        do_not_roast = False
        if 'do_not_roast' in dir(attr):
            do_not_roast = attr.do_not_roast

        # checks
        sanity_checks = True
        if self.module_limit_size is not None and isinstance(attr, torch.nn.Module):
            sanity_checks = sanity_checks and (get_module_params(attr) >= self.module_limit_size)
            if self.verbose > DEBUG:
                print("checker", attr, get_module_params(attr), self.module_limit_size)

        # modules
        module_check = (self.is_linear(attr) 
                                  or self.is_conv2d(attr) 
                                  or self.is_embedding(attr))
        return (not do_not_roast) and sanity_checks and  module_check

    def get_parameter(self, attr):
        assert(self.roastable(attr))
        idc = id(attr.weight)
        c = attr.weight.numel()
        return idc, c
        
class ModelRoastableParameters(ModelParser, Roastable):
    def __init__(self, model, module_limit_size=None, verbose=NONE):
        ModelParser.__init__(self)
        Roastable.__init__(self, module_limit_size=module_limit_size, verbose=verbose)
        self.model = model

    def lambda_func(self, state_dict):
        attr = state_dict['target_attr']
        if isinstance(attr, torch.nn.Parameter):
            if attr.requires_grad:
                state_dict['all'][id(attr)] = attr.numel()

        if isinstance(attr, torch.nn.Module):
            is_roastable = self.roastable(attr)
            if is_roastable:
                  idc, c = self.get_parameter(attr)
                  state_dict['compressable'][idc] = c

        return state_dict

    def process(self):
        state_dict = {"compressable" : {}, "all" : {}}
        self.run("model", self.model, state_dict)
        
        total = 0
        roastable = 0
        for _,i in state_dict['compressable'].items():
            roastable += i
        for _,i in state_dict['all'].items():
            total += i
    
        #print("Roastable {} / {}".format(roastable, total))
        state_dict['roastable'] = roastable
        state_dict['all'] = total

        return state_dict


class ModelRoaster(ModelParser, Roastable):

    def __init__(self, model, roast_global, sparsity, module_limit_size=None, verbose=NONE, init_std=0.04, mapper_args=None, partial = False, init_seed = 1):
        ModelParser.__init__(self)
        Roastable.__init__(self, module_limit_size=module_limit_size, verbose=verbose)
      
        self.verbose = verbose

        self.ROAST_INIT = init_std
        self.model = model
        self.is_global = roast_global
        self.compression = sparsity
        self.partial = partial
        self.init_seed = init_seed
        self.layers = []
        self.offsets = [0]

        parameter_finder = ModelRoastableParameters(model, module_limit_size=module_limit_size)
        pf = parameter_finder.process()
        roastable_params, total_params = pf['roastable'], pf['all']

        if self.verbose >= INFO:
            print("Roastable params: {}/{}".format(roastable_params, total_params))

        if self.is_global:
            ''' need to compute the max params: sparsity is applied to roastable parameters '''
            max_params = int(sparsity * roastable_params)
            #self.roast_array = torch.nn.Parameter(torch.FloatTensor(max_params).uniform_(-self.ROAST_INIT, self.ROAST_INIT))
            self.roast_array = torch.nn.Parameter(torch.FloatTensor(max_params).normal_(std=self.ROAST_INIT))
            if self.partial:
                self.roast_array = torch.nn.Parameter(torch.zeros(max_params))
        else:
            self.roast_array = None

        self.original_total_params = total_params
        self.original_roastable_params = roastable_params
        self.global_offset = 0
        self.mapper_args = mapper_args

    def make_roast_module(self, target_attr, seed):
        if not self.roastable(target_attr):
              return None
        new_attr = None

        if self.mapper_args is not None:
            mapper_args = copy.deepcopy(self.mapper_args)
            mapper_args["original_offset"] = self.global_offset
        else:
            mapper_args = None

        if self.is_linear(target_attr):
            new_attr = FakeRoastLinear(target_attr.in_features, 
                            target_attr.out_features,
                            target_attr.bias is not None,
                            self.is_global,
                            self.roast_array,
                            self.ROAST_INIT,
                            self.compression,
                            False,
                            "mapper" if (mapper_args is not None) else "random",
                            seed,
                            req_scale = torch.std(target_attr.weight).item(),
                            mapper_args = mapper_args,
                            partial = self.partial,
                            original_weight = target_attr.weight if self.partial else None,
                            original_bias = target_attr.bias if self.partial else None)
            self.global_offset = self.global_offset + target_attr.weight.numel()
            self.layers.append(new_attr)
            self.offsets.append(self.global_offset)

        if self.is_conv2d(target_attr):
            new_attr = FakeRoastConv2d( target_attr.in_channels,
                            target_attr.out_channels,
                            target_attr.kernel_size,
                            self.is_global,
                            self.roast_array,
                            self.ROAST_INIT,
                            self.compression,
                            target_attr.stride,
                            target_attr.padding,
                            target_attr.dilation,
                            target_attr.groups, 
                            target_attr.bias is not None,
                            target_attr.padding_mode,
                            False,
                            "mapper" if (mapper_args is not None) else "random",
                            seed,
                            req_scale = torch.std(target_attr.weight).item(),
                            mapper_args = mapper_args,
                            partial = self.partial,
                            original_weight = target_attr.weight if self.partial else None,
                            original_bias = target_attr.bias if self.partial else None)
            self.global_offset = self.global_offset + target_attr.weight.numel()
            self.layers.append(new_attr)
            self.offsets.append(self.global_offset)
        if self.is_embedding(target_attr):
            new_attr = FakeRoastEmbedding(target_attr.num_embeddings, 
                            target_attr.embedding_dim,
                            self.is_global, self.roast_array, self.ROAST_INIT, 
                            self.compression, target_attr.padding_idx, 
                            target_attr.max_norm, target_attr.norm_type,
                            target_attr.scale_grad_by_freq, 
                            target_attr.sparse,
                            matrix_mode= "mapper" if (mapper_args is not None) else "random",
                            req_scale = torch.std(target_attr.weight).item(),
                            mapper_args = mapper_args) # missing seed?
            self.global_offset = self.global_offset + target_attr.weight.numel()
            self.layers.append(new_attr)
            self.offsets.append(self.global_offset)
    
        return new_attr

    def lambda_init(self, state_dict):
        state_dict['seed'] = state_dict['init_seed'] * 1024
        return state_dict


    def lambda_next(self, state_dict):
        state_dict['init_seed'] = state_dict['init_seed'] + 1
        return state_dict

    def lambda_func(self, state_dict):
        attr = state_dict['target_attr']
        name = state_dict['name']
        state_dict['seed'] = state_dict['seed'] + 1
        new_attr = self.make_roast_module(attr, state_dict['seed'])
        if self.verbose >= DEBUG:
            print(type(attr), new_attr, flush=True)
        if new_attr is not None:
            setattr(state_dict['model'], name, new_attr)

        return state_dict

    def process(self):
        state_dict = {'init_seed' : self.init_seed}
        self.run("model", self.model, state_dict)
        if self.is_global:
            self.model.roast_array = self.roast_array
        else:
            self.model.roast_array = None
        return self.model
   

class ModelRoasterGradScaler(ModelRoaster):
    def __init__(self, model, roast_global, sparsity, module_limit_size=None, verbose=NONE, init_std=0.04, scaler_mode="v3", mapper_args=None, partial = "pending", k=-1, init_seed = 1):
        super(ModelRoasterGradScaler, self).__init__(model, roast_global, sparsity, module_limit_size=None, verbose=NONE, init_std=init_std,
                                                     mapper_args=mapper_args, partial=partial, init_seed=init_seed)
        assert(roast_global) # this should be defined only for roast_global
        self.scaler_mode = scaler_mode
        self.count = torch.zeros_like(self.roast_array)
        self.update_count = torch.zeros_like(self.roast_array).cuda()
        self.aggregate_scale = torch.zeros_like(self.roast_array)
        self.aggregate_scale_partial = torch.zeros_like(self.roast_array).cuda()
        self.partial = partial
        self.k = k


    ## adapt this function for your case
    def lambda_func(self, state_dict):
        attr = state_dict['target_attr']
        name = state_dict['name']
        state_dict['seed'] = state_dict['seed'] + 1
        new_attr = self.make_roast_module(attr, state_dict['seed'])
        if self.verbose >= DEBUG:
            print(type(attr), new_attr, flush=True)
        if new_attr is not None:
            setattr(state_dict['model'], name, new_attr)
            weight = new_attr.WHelper()
            _, count = new_attr.WHelper.wt_orig_to_comp(weight)
            self.count += count
            # note that this is sum of scale factors without sign G
            self.aggregate_scale += count * new_attr.scale 
        return state_dict

    def compute_roast_grad_scale_v1(self):
      return torch.square(self.aggregate_scale) / (1e-3 + self.count)

    def compute_roast_grad_scale_v2(self):
      return self.aggregate_scale

    def compute_roast_grad_scale_v3(self):
      return (self.aggregate_scale) / (1e-3 + self.count)

    def compute_roast_grad_scale(self):
      if self.scaler_mode == "v1":
          return self.compute_roast_grad_scale_v1()
      if self.scaler_mode == "v2":
          return self.compute_roast_grad_scale_v2()
      if self.scaler_mode == "v3":
          return self.compute_roast_grad_scale_v3()
      raise NotImplementedError

    '''
    def compute_roast_grad_scale(self):
      #return torch.square(self.aggregate_scale) / (1e-3 + self.count)
      return (self.aggregate_scale) / (1e-3 + self.count)
      #return self.aggregate_scale
    '''  
    def process(self):
        super().process()
        self.roast_array._roast_grad_scaler = self.compute_roast_grad_scale()
        return self.model
    
    def update_k(self, step):        
        start_step = self.k+1
        end_step = self.k+step

        if self.k >= self.original_roastable_params-1:
            raise Exception(f"{step} number of params not available to roast. {self.original_roastable_params - self.k - 1} params left")
        elif self.k + step >= self.original_roastable_params:
            end_step = self.original_roastable_params-1
            
        i=-1
        j=-1
        for threshold in self.offsets:
            if start_step>=threshold:
                i += 1
            if end_step >= threshold:
                j += 1

        for p in range(i, j+1):
            layerOffset = self.offsets[p]
            original_weights = self.layers[p].original_weight.data.flatten().cuda()
            roast_weights = self.layers[p].WHelper.weight.data.cuda()
            layer_scale = self.layers[p].scale
            layer_G = self.layers[p].WHelper.G.flatten().cuda()
            layer_index = torch.tensor(range(start_step-layerOffset, min(end_step + 1 - layerOffset, self.offsets[p+1]-layerOffset))).cuda()
            roast_index = self.layers[p].WHelper.IDX.flatten()[layer_index].cuda()

            roast_weights = roast_weights*self.update_count + torch.zeros_like(roast_weights).scatter_add_(0, roast_index, layer_G[layer_index]*original_weights[layer_index]/layer_scale)
            self.update_count.scatter_add_(0, roast_index, torch.ones_like(roast_index, dtype=torch.float32))
            self.aggregate_scale_partial.scatter_add_(0, roast_index, layer_scale*torch.ones_like(roast_index, dtype=torch.float32))
            self.layers[p].WHelper.weight.data = (roast_weights/self.update_count).nan_to_num(0)
            print("number of collisions: ", len(self.update_count[self.update_count > 1]))

            if end_step + 1 < self.offsets[p+1]:
                self.layers[p].mode = "roasting"
                self.layers[p].offset = end_step - self.offsets[p]
            else:
                self.layers[p].mode = "roasted"
                self.layers[p].original_weight = None

            start_step = self.offsets[p+1]
        self.k = end_step

class RoastGradScaler:
    def __init__(self):
        pass

    def scale_step(self, model):
        if not('roast_array' in dir(model)):
            return

        for p in model.parameters(): 
            if (p.requires_grad) and (p.grad is not None) and '_roast_grad_scaler' in dir(p):
                p._roast_grad_scaler = p._roast_grad_scaler.to(p.device)
                # p.grad = p.grad / (1e-6+p._roast_grad_scaler)
                p.grad = p.grad / (p._roast_grad_scaler)

