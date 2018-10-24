def slack_message(message, channel):
    token = 'xoxp-458177397862-456198240913-457863204487-2b70237b1cce3a921fb8ab925555e951'
    sc = SlackClient(token)
    sc.api_call('chat.postMessage', channel=channel,
                text=message, username='My Sweet Bot',
                icon_emoji=':robot_face:')