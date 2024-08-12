from functools import partial
from transformers import AutoModelForCausalLM,  AutoTokenizer

def replacing_hook(module, 
                  input,
                  output, 
                  name,
                  pos,
                  vec):
    if "attn" in name:    
        output[0][:, pos, :] = vec
    else:
        output[:, pos, :] = vec

    return output


def extracting_hook(module, 
                  input, 
                  output, 
                  name,
                  pos,
                  cache):
    if "attn" in name:    
       cache[name]  = output[0][:, pos, :]
    else:
        cache[name] = output[:, pos, :]
    return output


def apply_hook(module, 
               hook_func):
    """
    Applies a hook to a given layer of a model

    Parameters:
    - model: A transformer model
    - layer (int): Layer index to hook at
    - hook_func: A PyTorch Hook Fn

    Returns:
    A hook
    """
    hook = module.register_forward_hook(hook_func)
    return hook

def create_replace_hook(vec, name, pos):
    """
    Creates replacement hooks 
    
    Parameters:
        - vecs (List of Tensors): Vec to replace residual stream with
        - name (str): Name of input vec
        - pos (int): Position of the token of interest
    
    Returns:
        A hook
    """
    
    # Create hooks for each steering vector
    hook = partial(replacing_hook, vec=vec, pos=pos, name=name) 
    
    return hook


def create_extract_hook(name, pos, cache):
    """
    Creates hooks 
    
    Parameters:
        - name (str): Name of input vec
        - pos (int): Position of the token of interest
        - cache (Dict): Dictionary to store extracted vec
        
    Returns:
        A extraction hook
    """
    
    # Create hooks for each steering vector
    hook = partial(extracting_hook, cache=cache, name=name, pos=pos) 
    
    return hook


def load_model_tok(model_name):

    """
    Loads model and associated tokenizer

    Parameters:
    - model_name (str): Name of the model to load

    Returns:
    A tuple containing the model and tokenizer
    """

    # Load model and tokenizer
  
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda() 
    model.eval()  

    tokenizer = AutoTokenizer.from_pretrained(model_name)
  

    return model, tokenizer