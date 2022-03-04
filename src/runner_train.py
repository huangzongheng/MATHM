import argparse
from train import main


# to manage multiple experiments
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CSE_ResNet Model for TUBerlin Training')
    # pdb.set_trace()
    parser.add_argument('args', type=str, help='args')
    print(parser.parse_args())
    args = parser.parse_args().args
    args = args.split(' ')
    while True:
        try :
            args.remove('')
        except ValueError as e:
            break
    main(args)
