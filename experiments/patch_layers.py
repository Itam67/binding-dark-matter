
import torch
from src.utils import apply_hook, create_extract_hook, create_replace_hook, load_model_tok
import matplotlib.pyplot as plt



def main(config):

    # Load the model and tokenizer
    model, tokenizer = load_model_tok(config["model_name"])

    # Load the data
    source_input = config["source_input"]
    source_inputs = tokenizer(source_input, return_tensors="pt").input_ids.cuda()

    target_input = config["target_input"]
    target_inputs = tokenizer(target_input, return_tensors="pt").input_ids.cuda()

    # Position of the token of interest
    pos = config["pos"]
    target_tk = source_inputs[0, pos-1]
    target_tk_name = tokenizer.decode(target_tk)
    print("Target Token: ", target_tk_name)

    # Create extract hooks
    cache = {}
    components = [(name, module) for name, module in model.named_modules()]
    components = [(name, module) for name, module in components if name.endswith("self_attn") or name.endswith("mlp")]
    
    hook_funcs = [(create_extract_hook(name=name, cache=cache, pos=pos), module) for name, module in components]

    # Hook the model with extract hook
    hooks = []
    for i, (hook, module) in enumerate(hook_funcs):
        hooks.append(apply_hook(module, hook))


    # Run the model to get the hidden states
    print("Getting the hidden states from source...")
    model(source_inputs)

    # Remove the hooks
    for hook in hooks:
        hook.remove()

    # Patch in at every layer
    logits = []
    ranks = []
    
    # The first element is the control
    control = model(target_inputs)
    logits.append(control.logits[0, -1, target_tk])
    ranks.append(control.logits[0, -1].argsort().flip(0).tolist().index(target_tk))



    print("Patching in at from each layer to layer 4...")
    for layer in components:

        print(layer[0])
    
        # Hook the model with replacement hooks
        hook_fn = create_replace_hook(vec=cache[layer[0]], pos=-2, name=layer[0])

        if "attn" in layer[0]:
            hook = apply_hook(components[6][1], hook_fn)
        else:
            hook = apply_hook(components[7][1], hook_fn)

        # Run the model and replace the hidden states
        outputs = model(target_inputs)

        # Get logit of token of interest
        logits.append(outputs.logits[0, -1, target_tk])
        ranks.append(outputs.logits[0, -1].argsort().flip(0).tolist().index(target_tk))

        hook.remove()

    # Save the logits
    exp_name = config["exp_name"]
    torch.save(torch.Tensor(logits), f"experiment_results/{exp_name}_logits.pt")
    torch.save(torch.Tensor(ranks), f"experiment_results/{exp_name}_ranks.pt")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(ranks[1:], label='Patchscopes Ranks')
    plt.axhline(y=ranks[0], color='red', linestyle='--', linewidth=2, label="Control")
    
    # Create custom x-tick labels alternating between ATTN_layer_# and MLP_layer_#
    tick_labels = [''] * len(ranks[1:])
    for i in range(0, len(ranks[1:]), 10):
        tick_labels[i] = f'Attn_{i//2 }'  # Label for ATTN layer
        tick_labels[i+1] = f'MLP_{i//2}'  # Label for MLP layer

    # Set x-ticks and their labels
    plt.xticks(range(len(ranks[1:])), tick_labels, rotation=60, ha="right")
    
    
    plt.title(f"Rank of '{target_tk_name}' token when Patched to Model Layer 4 on Target Prompt")
    plt.xlabel("Layer of Source Prompt Activation Used in Patching")
    plt.ylabel("Token Rank")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"experiment_results/{exp_name}_ranks.png")

    return



config = {
    "exp_name": "red",
    "model_name": "google/gemma-2-9b",
    "source_input": "A red mouse and a blue cat went to the market.",
    "target_input": "banana:yellow, apple:red, grass:green, sky:blue, x:",
    "pos": 3
}

if __name__ == "__main__":
    main(config)