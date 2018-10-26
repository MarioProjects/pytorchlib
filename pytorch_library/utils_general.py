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



