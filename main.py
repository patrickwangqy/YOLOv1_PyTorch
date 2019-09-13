import argparse

from train.train import Train


def train_exec(args):
    train = Train(10)
    train.fit(None)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="ssub command")

    # train
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train_exec)

    args = parser.parse_args()
    args.func(train_exec)


if __name__ == "__main__":
    main()
