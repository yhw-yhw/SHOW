import torch


def parse_config_to_weight_dict(config)->dict:
    opt_weights_dict=config.opt_weights_dict

    if config.use_pre_compute_betas:
        opt_weights_dict.update(config.pre_compute_betas_weight)
        
    len_set=[len(val) for _,val in opt_weights_dict.items()]
    # assert(len(set(len_set))==1)
    
    max_weight_len=max(len_set)
    for k,v in opt_weights_dict.items():
        assert isinstance(v,list), "Invalid weights type"
        if len(v)<max_weight_len:
            opt_weights_dict[k].extend([v[-1]]*(max_weight_len-len(v)))
    
    return opt_weights_dict


def parse_weight_dict_to_list(opt_weights_dict,device,dtype)->list:
    keys = opt_weights_dict.keys()
    
    opt_weights = [dict(zip(keys, vals)) for vals in
                zip(*(opt_weights_dict[k] for k in keys))]
    
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(
                weight_list[key], device=device, dtype=dtype)

    return opt_weights


def parse_weight(config,device,dtype)->list:
    ret=parse_config_to_weight_dict(config)
    ret=parse_weight_dict_to_list(ret,device,dtype)
    return ret