import torch
import os
from collections import OrderedDict

def _convert_keras2pytorch(modelname='model3'):
    path = os.path.join('./converted_model', modelname + '.pth')
    if modelname == 'model3':
        orig_dict = torch.load(path)

        base_dict = torch.load('./tnet_init.pth')
        base_dict = [(name, val) for name, val in base_dict.items() if 'conv' in name]
        base_dict = OrderedDict(base_dict)

        assert len(orig_dict) == len(base_dict), 'cannot assign'

        dest_dict = []
        for (base_name, base_val), (orig_name, orig_val) in zip(base_dict.items(), orig_dict.items()):
            if orig_val.ndim == 4: # convert shape = (h, w, in_c, out_c) to (out_c, in_c, h, w)
                dest_dict += [(base_name, orig_val.permute((3, 2, 0, 1)).contiguous())]
            else:
                dest_dict += [(base_name, orig_val)]
        dest_dict = OrderedDict(dest_dict)

        torch.save(dest_dict, './{}-converted.pth'.format(modelname))
        print('saved to ' + './{}-converted.pth'.format(modelname))

if __name__ == '__main__':
    _convert_keras2pytorch(modelname='model3')