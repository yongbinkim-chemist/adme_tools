from .parser import parse_args

from .tasks import run_train, run_predict, fine_tunning

def main():

    args = parse_args()
    TASK = args.task

    if TASK == 'train':
        run_train(args)
    elif TASK == 'predict':
        run_predict(args)
    else:
        fine_tunning(args)

if __name__ == "__main__":
    main()
