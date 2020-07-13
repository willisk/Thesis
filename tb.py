from itertools import product

parameters = dict(
    lr=[0.1, 0.01],
    batch_size=[32, 64],
)

param_values = parameters.values()
param_keys = parameters.keys()
comment_fmt = " ".join([k + "={}" for k in param_keys])

for param in product(*param_values):
    comment = comment_fmt.format(*param)
    print(comment)
