# NOTE to self: this shouldn't be necessary; implement in utility.py
# although using interactive notebooks makes this approach useful
# reload shared on new experiment run; don't reload on child modules


from torch.utils.tensorboard import SummaryWriter

tb = None


def init_summary_writer(log_dir, comment=""):
    global tb
    tb = SummaryWriter(log_dir=log_dir, comment=comment)
    return tb


def get_summary_writer():
    return tb
