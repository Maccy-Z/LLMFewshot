import torch

lin_fn = torch.nn.Linear(3, 4)

def fun(xs):
    return lin_fn(xs)


def batch_forward(xs, fn):
    splits = [len(x) for x in xs]

    xs = torch.cat(xs)
    xs = fn(xs)
    xs = torch.split(xs, splits)

    return xs


ts = [torch.arange(0, i * 3 + 3, 1, dtype=torch.float).view(i + 1, 3) for i in range(4)]

ts = batch_forward(ts, fun)

for t in ts:
    print(t)