import argparse
import importlib
import os
import glob


def define_arguments(parser):
    parser.add_argument('--root', type=str, default="", help="")
    parser.add_argument('--src-lang', type=str, default='en', help="")
    parser.add_argument('--tgt-lang', type=int, default='zh', help="")
    parser.add_argument('--adv-training', action='store_true', help="")
    parser.add_argument('--batch-size', type=int, default=2, help="")
    parser.add_argument('--eval-batch-size', type=int, default=32, help="")
    parser.add_argument('--patience', type=int, default=5, help="")
    parser.add_argument('--hidden-dim', type=int, default=1024, help="")
    parser.add_argument('--max-length', type=int, default=128, help="")
    parser.add_argument('--run-task', choices=['sent', 'trigger', 'role'], help="")
    parser.add_argument('--accumulation-steps', type=int, default=4, help="")
    parser.add_argument('--no-gpu', action="store_true", help="don't use gpu")
    parser.add_argument('--gpu', type=str, default='1', help="gpu")
    parser.add_argument('--max-grad-norm', type=float, default=1, help="")
    parser.add_argument('--g-lam', type=float, default=5e-1, help="")
    parser.add_argument('--learning-rate', type=float, default=1e-5, help="")
    parser.add_argument('--decay', type=float, default=1e-2, help="")
    parser.add_argument('--warmup-step', type=float, default=0, help="")
    parser.add_argument('--seed', type=int, default=44739242, help="random seed")
    parser.add_argument('--log-dir', type=str, default="./log/", help="path to save log file")
    parser.add_argument('--save-model', type=str, default="model", help="prefix to save checkpoints")
    parser.add_argument('--model-name', type=str, default="xlm-roberta-large", help="pretrained lm name")
    parser.add_argument('--load-model', type=str, default="", help="path to saved checkpoint")
    parser.add_argument('--train-epoch', type=int, default=50, help='epochs to train')
    parser.add_argument('--train-step', type=int, default=100000, help='steps to train')
    parser.add_argument('--test-only', action="store_true", help='is testing')
    parser.add_argument('--parallel', action="store_true", help='parallel')
    parser.add_argument('--adv', action="store_true", help='adv')
    parser.add_argument('--continue-train', action="store_true", help='continue training')
    parser.add_argument('--clean-log-dir', action="store_true", help='is testing')


def parse_arguments():
    parser = argparse.ArgumentParser()
    cwd = os.getcwd()
    path = "default_options" if cwd.endswith("utils") else "utils.default_options"
    default_options = importlib.util.find_spec(path)
    if default_options:
        define_default_arguments = default_options.loader.load_module().define_default_arguments
        define_default_arguments(parser)
    else:
        define_arguments(parser)
    args = parser.parse_args()
    args.log = os.path.join(args.log_dir, f"logfile.log")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if args.test_only and args.load_model == "":
        args.load_model = glob.glob(os.path.join(args.log_dir, "model*"))[0]
    if False and args.clean_log_dir and (not args.test_only) and (not args.continue_train) and os.path.exists(args.log_dir):
        existing_logs = glob.glob(os.path.join(args.log_dir, "*"))
        for _t in existing_logs:
            os.remove(_t)
    
    return args
