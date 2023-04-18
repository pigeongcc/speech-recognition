import torch

import torch.nn.functional as F

from TextTransform import TextTransform


class GreedyDecoder:

    def __init__(self, text_transform) -> None:
        self.text_transform = text_transform
    
    def decode(self,
               output,
               labels,
               label_lengths,
               blank_label=1,
               collapse_repeated=True):
        
        output = F.log_softmax(output, dim=2)

        arg_maxes = torch.argmax(output, dim=2)
        decodes = []
        targets = []
        
        for i, args in enumerate(arg_maxes):
            decode = []
            targets.append(self.text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
            blank_ctr = 0
            for j, index in enumerate(args):
                if collapse_repeated and index == blank_label:
                    blank_ctr += 1
                else:
                    if blank_ctr > 2:
                        decode.append(blank_label)
                    blank_ctr = 0

                    if collapse_repeated and j != 0 and index == args[j - 1]:
                        continue
                    
                    decode.append(index.item())
            
            decodes.append(self.text_transform.int_to_text(decode))

        return decodes, targets