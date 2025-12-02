import os
import argparse
import platform

from utils.config import cfg, cfg_from_file, cfg_from_list


def cp_some(src, tgt):
    if platform.system().lower() == "windows":
        cmd = "copy {} {}".format(src, tgt)
    else:
        cmd = "cp {} {}".format(src, tgt)
    print(cmd)
    os.system(cmd)


def generate_output_path(model_name):
    output_path = os.path.join("results", model_name)
    model_save_path = os.path.join("results", model_name, "model_save")
    return output_path, model_save_path


def parse_args(
        description,
        extra_args=None,
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--cfg",
        "--config",
        dest="cfg_file",
        action="append",
        help="an optional config file",
        default=None,
        type=str,
    )

    # Apply extra user inputs.
    if extra_args is not None:
        if not isinstance(extra_args, list):
            extra_args = [extra_args]
        for ex_arg in extra_args:
            ex_arg(parser)

    args = parser.parse_args()

    # load cfg from file
    if args.cfg_file is not None:
        for f in args.cfg_file:
            cfg_from_file(f)

    if len(cfg.MODEL_NAME) != 0:
        output_path, model_save_path = generate_output_path(cfg.MODEL_NAME)
        cfg_from_list(
            ["OUTPUT_PATH", output_path, "MODEL_SAVE_PATH", model_save_path]
        )
        if not os.path.exists(cfg.OUTPUT_PATH):
            os.makedirs(cfg.OUTPUT_PATH, exist_ok=True)
        if not os.path.exists(cfg.MODEL_SAVE_PATH):
            os.makedirs(cfg.MODEL_SAVE_PATH, exist_ok=True)

    # Save the config file into the model save path
    for f in args.cfg_file:
        cp_some(f, cfg.OUTPUT_PATH)

    return args