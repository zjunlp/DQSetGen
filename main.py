import argparse
import warnings
warnings.filterwarnings("ignore")
from args import train_argparser, eval_argparser
from config_reader import process_configs
from src import input_reader
from src.ssn_trainer import SSNTrainer


def __train(run_args):
    trainer = SSNTrainer(run_args)
    if run_args.discontinuous:
        input_reader_cls = input_reader.DiscontinuousInputReader
    else:
        input_reader_cls = input_reader.JsonInputReader
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader_cls)


def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __eval(run_args):
    trainer = SSNTrainer(run_args)
    if run_args.discontinuous:
        input_reader_cls = input_reader.DiscontinuousInputReader
    else:
        input_reader_cls = input_reader.JsonInputReader
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader_cls)


def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()

    print(args.mode)

    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    else:
        raise Exception("Mode not in ['train', 'eval'], e.g. 'python ssn.py train ...'")
