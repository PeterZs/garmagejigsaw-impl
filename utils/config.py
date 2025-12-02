import importlib
from easydict import EasyDict as edict


# === basic ===
__C = edict()
cfg = __C
__C.MODEL_NAME = ""  # this name would be the result file name
__C.MODULE = ""
__C.PROJECT = ""        # wandb project name
__C.LOG_FILE_NAME = ""  # the suffix of log file
__C.MODEL_SAVE_PATH = ""  # auto generated
__C.BATCH_SIZE = 32
__C.NUM_WORKERS = 8


# === Dataset ===
__C.DATASET = ""
# Other dataset specific configs should be imported from dataset_config.py


# === Training options ===
__C.TRAIN = edict()
__C.TRAIN.NUM_EPOCHS = 200      # Total epochs
__C.TRAIN.OPTIMIZER = "SGD"     # Optimizer type
__C.TRAIN.LR = 0.001            # Start learning rate
__C.TRAIN.LR_SCHEDULER = "cosine"   # LR Scheduler
__C.TRAIN.LR_DECAY = 100.0      # Learning rate decay
__C.TRAIN.LR_STEP = [10, 20]    # Learning rate decay step (in epochs)
__C.TRAIN.WARMUP_RATIO = 0.0    # warmup_ratio for Adam Cosine
__C.TRAIN.CLIP_GRAD = None      # clip_grad
__C.TRAIN.beta1 = 0             # beta1, beta2 for Adam Optimizer
__C.TRAIN.beta2 = 0.9
__C.TRAIN.WEIGHT_DECAY = 0.0    # weight decay for Adam or SGD
__C.TRAIN.MOMENTUM = 0.9    # SGD momentum
__C.TRAIN.VAL_EVERY = 1     # Check val every n epoch
__C.TRAIN.LOSS = ""         # Loss function.
__C.TRAIN.FINETUNE = False  # 是否finetune


# === WandB ===
__C.WANDB = edict()
__C.WANDB.ID = ""


# === Callback ===
__C.CALLBACK = edict()
__C.CALLBACK.MATCHING_TASK = ["trans"]
__C.CALLBACK.CHECKPOINT_MONITOR = "val/loss"
__C.CALLBACK.CHECKPOINT_MODE = "min"


# === Loss config ===
__C.LOSS = edict()


# === Evaluation options ===
__C.EVAL = edict()


# === MISC ===
__C.GPUS = [0]      # Parallel GPU indices ([0] for single GPU)
__C.PARALLEL_STRATEGY = "ddp"   # Parallel strategy for multiple gpus
__C.FP16 = False    # Float Precision, 32 for False, 16 for True
__C.CUDNN = False   # CUDNN benchmark
__C.RESUME = False          # Whether to resume training from a specific checkpoint (and continue the WandB training curve)
__C.WEIGHT_FILE = ""        # (Used only for non-split models.)
__C.WEIGHT_FILE_POINTCLASSIFIER = ""     # (Used only for split models) Checkpoint for the point classification module
__C.WEIGHT_FILE_STITCHPREDICTOR = ""     # (Used only for split models) Checkpoint for the point stitching module
__C.OUTPUT_PATH = ""        # Output path (for checkpoints, running logs)
__C.RANDOM_SEED = 42        # random seed used for data loading
__C.STATS = ""      # directory for collecting statistics of results


def _merge_a_into_b(a, b):
    """Merge config dictionary A into config dictionary B, clobbering the
    options in B whenever they are also specified in A.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError("{} is not a valid config key".format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            if type(b[k]) is float and type(v) is int:
                v = float(v)
            # elif type(b[k]) is list:
            #     v = [v]
            else:
                if k not in ["CLASS"]:
                    raise ValueError(
                        "Type mismatch ({} vs. {}) for config key: {}".format(
                            type(b[k]), type(v), k
                        )
                    )

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print("Error under config key: {}".format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml

    with open(filename, "r") as f:
        yaml_cfg = edict(yaml.full_load(f))

    if "MODEL" not in __C:
        model_cfg_module = 'model.garmage_jigsaw.model_config'
        mod = importlib.import_module(model_cfg_module)
        __C["MODEL"] = mod.get_model_cfg()

    if "DATA" not in __C:
        dataset_cfg_module = 'dataset.dataset_config_GarmageJigsaw'
        dat = importlib.import_module(dataset_cfg_module)
        __C["DATA"] = dat.get_dataset_cfg()

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval

    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split(".")
        d = __C
        for sub_key in key_list[:-1]:
            assert sub_key in d.keys()
            d = d[sub_key]
        sub_key = key_list[-1]
        assert sub_key in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(
            d[sub_key]
        ), "type {} does not match original type {}".format(
            type(value), type(d[sub_key])
        )
        d[sub_key] = value
