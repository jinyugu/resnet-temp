import torch



# def update_state_dict(state_dict, idx_start=9):

#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k[idx_start:]  # remove 'module.0.' of dataparallel
#         new_state_dict[name]=v

#     return new_state_dict

state_dict = torch.load('/home/jinyu/resnet-temp/state_dict/sd12.pt')
print("Lodaing ...")
for k,v in state_dict.items():
    print(k)