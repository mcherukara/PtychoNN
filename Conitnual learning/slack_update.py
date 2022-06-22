
from slack_sdk import WebClient

 
slack_bot_token = "xoxb-679835710832-2052497567909-oB0WeYpoEChiXF3FL0XYm1tb"
slack_webclient = WebClient(token=slack_bot_token)


def post_message(msg):
    slack_webclient.chat_postMessage(channel="#automated", text=msg)
    
def post_figure(filename):
    slack_webclient.files_upload(channels="#automated", file=filename)