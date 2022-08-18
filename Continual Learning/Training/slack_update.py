
from slack_sdk import WebClient

 
slack_bot_token = " "
slack_webclient = WebClient(token=slack_bot_token)


def post_message(msg):
    slack_webclient.chat_postMessage(channel="#automated", text=msg)
    
def post_figure(filename):
    slack_webclient.files_upload(channels="#automated", file=filename)