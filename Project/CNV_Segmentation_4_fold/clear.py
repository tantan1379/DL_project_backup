import os
from config import cnv_single_config

def remove_log_and_checkpoint(net_name):
    checkpoint_path = os.path.join("./checkpoints", net_name)
    for fold in os.listdir(checkpoint_path):
        for checkpoint in os.listdir(os.path.join(checkpoint_path, fold)):
            os.remove(os.path.join(checkpoint_path, fold, checkpoint))
    if os.path.exists(os.path.join('./logs',net_name+'.txt')):
        os.remove(os.path.join('./logs',net_name+'.txt'))
    if os.path.exists(os.path.join('./logs',net_name+'_test_indicator.txt')):
        os.remove(os.path.join('./logs',net_name+'_test_indicator.txt'))

if __name__ == "__main__":
    args = cnv_single_config()
    remove_log_and_checkpoint(args.net_work)