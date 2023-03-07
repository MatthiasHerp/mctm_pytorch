#import torch
#import functorch
#def custom_where(x):
#    return torch.greater_equal(x,0)
#
#if __name__ == "__main__":
#
#    a = torch.normal(2, 3, size=(1,4))
#    print(functorch.vmap(custom_where)(a))
#
import torch
from functorch import vmap

def f(x):
    if x > 0:
        return x
    else:
        return 0
    #return torch.greater_equal(torch.cumsum(x, dim=0), .5 * 10)

if __name__ == "__main__":
    x = torch.randn([10,10])
    vmap(f)(x)