# BERT


#### Some comments extracted from the miniBERT codebase

# type_given = config['model_type'] is not None
# params_given = all([config['n_layer'] is not None, config['n_head'] is not None, config['n_embd'] is not None])
# assert type_given ^ params_given # exactly one of these (XOR)
# if type_given:
#     # translate from model_type to detailed configuration
#     config['merge_from_dict']({
#         # names follow the huggingface naming conventions
#         # GPT-1 yer=12, n_head=12, n_embd=768),  # 117M params
#         # GPT-2 configs
#         'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
#         'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
#         'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
#         'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
#         # Gophers
#         'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
#         # (there are a number more...)
#         # I made these tiny models up
#         'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
#         'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
#         'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
#     }[config['model_type']])


