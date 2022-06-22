import paramiko
import os

def agx_update(best_model_path):
    ssh=paramiko.SSHClient()
    ssh.load_host_keys(os.path.expanduser(os.path.join("~",".ssh","known_hosts")))
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("jetson-agx.xray.aps.anl.gov", username="nvidia-agx", password="agx@xsd", look_for_keys=False, allow_agent=False)
    sftp=ssh.open_sftp()

    sftp.put(best_model_path, "/home/nvidia-agx/Inference/trainedModels/best_model_new.pth")

