# reload shared on new experiment run; don't reload on child modules
import sys

from torch.utils.tensorboard import SummaryWriter


def parse_args_to_hyperparameters():
    return dict(
        method=args.method,
        cc=args.class_conditional,
        mask_bn=args.mask_bn,
        use_bn_stats=args.use_bn_stats,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        factor_reg=args.factor_reg,
        factor_input=args.factor_input,
        factor_layer=args.factor_layer,
        factor_criterion=args.factor_criterion,
        distr_a=args.distr_a,
        distr_b=args.distr_b,
    )


class Dummy(object):
    def __getattr__(self, _):
        return self.nop

    def nop(*args, **kw):
        pass


def init_summary_writer(log_dir):
    global LOGDIR
    LOGDIR = log_dir


def get_summary_writer(comment=None):
    if ENABLED:
        global writer
        if comment is not None:
            print(LOGDIR)
            writer = SummaryWriter(LOGDIR + "/" + comment)
        return writer
    else:
        return Dummy()


def disable_tb():
    ENABLED = False


writer = Dummy()
LOGDIR = None
ENABLED = True
