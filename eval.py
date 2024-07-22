from gan_tools.benchmark_utils import eval_utils
from gan_tools.model import utils

import warnings

warnings.filterwarnings("ignore")


def run_eval(config):
    model = eval_utils.load_model(config)
    fid = eval_utils.get_fid_score(model, config)
    print('The FID score of the current model is {0}'.format(fid))


def main():
    # parse command line and run
    parser = utils.prepare_parser()
    config = vars(parser.parse_args())
    run_eval(config)


if __name__ == '__main__':
    main()
