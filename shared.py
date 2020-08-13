# NOTE to self: this shouldn't be necessary; implement in utility.py
# although using interactive notebooks makes this approach useful
# reload shared on new experiment run; don't reload on child modules


from torch.utils.tensorboard import SummaryWriter

tb = None
LOGDIR = None
ENABLED = True


def init_summary_writer(log_dir):
    global LOGDIR
    LOGDIR = log_dir


def get_summary_writer(comment=None):
    if ENABLED:
        global tb
        if comment is not None:
            tb = SummaryWriter(LOGDIR + " " + comment)
        return tb
    else:
        return None


def disable_tb():
    ENABLED = False
