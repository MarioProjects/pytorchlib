import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from slackclient import SlackClient

def slack_message(message, channel):
    token = 'xoxp-458177397862-456198240913-464006277364-a7557a18c11f5e99ca9ce33deacbefc4'
    sc = SlackClient(token)
    sc.api_call('chat.postMessage', channel=channel,
                text=message, username='My Sweet Bot',
                icon_emoji=':robot_face:')

def topk_classes(input, k):
    """
    Devuelve las k clases mas votadas de cada entrada de mayor a menor probabilidad
    Para uso mas exhaustivo ir a https://pytorch.org/docs/stable/torch.html#torch.topk
    """
    probs_values, class_indxs = torch.topk(input, k)
    return class_indxs

def models_average(outputs, scheme):
    """
    Dada una lista de salidas (outputs) promedia los resultados obtenidos
    teniendo dos posibildiades: 
        - voting: donde la clase de mayor probabilidad vota 1 y el resto 0
        - sum: se suma la probabilidad de todas las clases para decidir
    """
    # Si no utilizamos el numpy() el tensor va por referencia y 
    # modifica la entrada original
    if scheme == "sum":
        result = outputs[0].data.cpu().numpy()
        for output in outputs[1:]:
            result += output.data.cpu().numpy()
        return torch.from_numpy(result)
    elif scheme == "voting":
        # Es necesario tratar los outputs para pasarlos a codificacion 1 y 0s
        # https://discuss.pytorch.org/t/keep-the-max-value-of-the-array-and-0-the-others/14480/7
        view = outputs[0].view(-1, 5)
        result = (view == view.max(dim=1, keepdim=True)[0]).view_as(outputs[0])
        for output in outputs[1:]:
            view = output.view(-1, 5)
            result += (view == view.max(dim=1, keepdim=True)[0]).view_as(output)
        return result
    else: assert False, "Ivalid model average scheme!"


