# from collections import OrderedDict

# from torch.autograd import Variable
from utils import to_torch
import torch
from torch.autograd import Variable

# def extract_cnn_feature(model, inputs, modules=None):
#     model.eval()
#     inputs = to_torch(inputs)
#     with torch.no_grad():
#         inputs = inputs.cuda()
#         if modules is None:
#             outputs = model(inputs)
#             outputs = outputs.data
#             return outputs

#     # Register forward hook for each module
#     outputs = OrderedDict()
#     handles = []
#     for m in modules:
#         outputs[id(m)] = None
#         def func(m, i, o): outputs[id(m)] = o.data
#         handles.append(m.register_forward_hook(func))
#     model(inputs)
#     for h in handles:
#         h.remove()
#     return list(outputs.values())


def extract_cnn_feature(model, inputs, pool_feature=False):
    model.eval()
    with torch.no_grad():
        inputs = to_torch(inputs)
        inputs = Variable(inputs).cuda()
        if pool_feature is False:
            outputs = model(inputs)
            return outputs
        else:
            # Register forward hook for each module
            outputs = {}


        def func(m, i, o): outputs['pool_feature'] = o.data.view(n, -1)
        hook = model.module._modules.get('features').register_forward_hook(func)
        model(inputs)
        hook.remove()
        # print(outputs['pool_feature'].shape)
        return outputs['pool_feature']

    
