import os, sys
import warnings

warnings.filterwarnings("ignore")

import argparse
import math, random
import torch
import time

from config import PARAMS_CONFIG
from data import get_acc_data
from models_causal import CausalMoE
from trainer_reasoning import train_iteration, full_eval
import datetime
# import wandb
import os
from utils import (
    get_params,
    set_up_env,
    get_optimizer_and_scheduler,
    load_checkpoint,
    save_checkpoint,
    create_exp_dir,
    freeze_gate_weight,
    Logger,
    set_freq_optimal_search,
)


def launch(
    env_params,
    model_params,
    adapt_span_params,
    optim_params,
    data_params,
    trainer_params,
    wandb_params,
):
    # wandb_flag = wandb_params["wandb_flag"]
    # if wandb_flag:
    #     wandb.init(project=wandb_params["project_name"])
    #     wandb.run.name = wandb_params["job_name"]
    #     wandb.config.update(model_params)
    # global val
    best_val_loss = None
    # ENVIRONMENT (device, distributed, etc.)
    set_up_env(env_params)
    device = env_params["device"]
    distributed = env_params["distributed"]
    resume = trainer_params["resume"]

    if distributed == False or env_params["rank"] == 0:
        print("data_params:\t", data_params)
        print("model_params:\t", model_params)
        print("optim_params:\t", optim_params)
        print("trainer_params:\t", trainer_params)
        print("adapt_span_params:\t", adapt_span_params)
    train_path = os.path.join(data_params["data_path"], "train.json")
    val_path = os.path.join(data_params["data_path"], "valid.json")
    # DATA
    train_data_x,train_data_y= get_acc_data(
        data_params=data_params,
        env_params=env_params,
        path=train_path,
        block_size=model_params["block_size"],
        batch_size=trainer_params["batch_size"],
        device=device,
    )
    val_data_x, val_data_y = get_acc_data(
        data_params=data_params,
        env_params=env_params,
        path=val_path,
        block_size=model_params["block_size"],
        batch_size=trainer_params["batch_size"],
        device=device,
    )

    # MODEL
    model = CausalMoE(
        vocab_size=data_params["vocab_size"],
        **model_params,
        adapt_span_params=adapt_span_params,
    )
    # print(model)
    if distributed:
        local_rank = env_params["local_rank"]
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    else:
        model = torch.nn.DataParallel(model)
        model = model.to(device)

    # OPTIMIZER AND SCHEDULER
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, optim_params=optim_params
    )

    # create logger
    logger = Logger()
    # fold_name = trainer_params["checkpoint_path"].split("/")[-1].split(".")[0]
    fold_name=optim_params["lr"]
    folder_path = "/".join(trainer_params["checkpoint_path"].split("/")[:-1])
    logging = create_exp_dir(f"{folder_path}/experiments/{fold_name}")
    # log paramters
    logging(f"Training Parameters:\n {trainer_params}")
    logging(f"Models Parameters:\n {model_params}")
    # logging time
    current_time = datetime.datetime.now()
    logging(str(current_time))
    # log model
    logging(str(model))
    logging(f"Total of Parameters: {sum(p.numel() for p in model.parameters())}")
    logging(
        f"Total of Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )



    if trainer_params["pretrained_weight"] is not None:
        with open(trainer_params["pretrained_weight"], "rb") as f:
            pretrained_model = torch.load(f)
        # pdb.set_trace()
        pretrained_model_checkpoint = pretrained_model["model"]  # .state_dict()
        filtered_checkpoint = {}
        for key in pretrained_model_checkpoint.keys():
            if not key in model.state_dict():
                logging("Can not load {}".format(key))
            elif (
                    not pretrained_model_checkpoint[key].shape
                        == model.state_dict()[key].shape
            ):
                logging("Can not load {}, shape do not match".format(key))
            else:
                filtered_checkpoint[key] = pretrained_model_checkpoint[key]
        model.load_state_dict(filtered_checkpoint, strict=False)
    iter_init=0
    if not trainer_params["full_eval_mode"] and resume:
        iter_init = load_checkpoint(
            trainer_params["checkpoint_path"],
            model,
            optimizer,
            scheduler,
            logger,
            distributed,
            resume,
        )
    # fix gate
    if model_params["smoe_dropout"]:
        freeze_gate_weight(model)
    # calculate time
    start_time = time.time()
    # eval model
    if trainer_params["full_eval_mode"]:
        base_path = trainer_params["checkpoint_path"]
        folder = os.path.dirname(base_path)
        base_name = os.path.basename(base_path)
        name_no_ext, ext = os.path.splitext(base_name)

        # Find all epoch checkpoints: e.g., model_epoch0.pt, model_epoch1.pt, ...
        import re
        epoch_checkpoints = []
        for f in os.listdir(folder):
            if f.startswith(name_no_ext + "_epoch") and f.endswith(ext):
                match = re.search(rf"{re.escape(name_no_ext)}_epoch(\d+){re.escape(ext)}$", f)
                if match:
                    epoch = int(match.group(1))
                    epoch_checkpoints.append((epoch, os.path.join(folder, f)))

        # Sort by epoch number
        epoch_checkpoints.sort(key=lambda x: x[0])

        if not epoch_checkpoints:
            logging(f"Warning: No epoch checkpoints found matching pattern {name_no_ext}_epoch*{ext} in {folder}")
            # Fallback: try loading the original checkpoint path (e.g., best model)
            epoch_checkpoints = [(None, base_path)]

        # Loop over each checkpoint
        for epoch, ckpt_path in epoch_checkpoints:
            logging(f"\n{'=' * 60}")
            logging(f"Loading checkpoint: {ckpt_path} (Epoch {epoch if epoch is not None else 'N/A'})")
            logging(f"{'=' * 60}")

            # Re-initialize model to avoid state leakage (important!)
            model = CausalMoE(
                vocab_size=data_params["vocab_size"],
                **model_params,
                adapt_span_params=adapt_span_params,
            )
            if distributed:
                local_rank = env_params["local_rank"]
                model = model.to(device)
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True,
                )
            else:
                model = torch.nn.DataParallel(model)
                model = model.to(device)

            # Load this specific checkpoint (ignore optimizer/scheduler for eval)
            try:
                with open(ckpt_path, "rb") as f:
                    pretrained_model = torch.load(f)
                pretrained_model_checkpoint = pretrained_model["model"]  # .state_dict()
                filtered_checkpoint = {}
                for key in pretrained_model_checkpoint.keys():
                    if not key in model.state_dict():
                        logging("Can not load {}".format(key))
                    elif (
                            not pretrained_model_checkpoint[key].shape
                                == model.state_dict()[key].shape
                    ):
                        logging("Can not load {}, shape do not match".format(key))
                    else:
                        filtered_checkpoint[key] = pretrained_model_checkpoint[key]
                model.load_state_dict(filtered_checkpoint, strict=False)
            except Exception as e:
                logging(f"Failed to load {ckpt_path}: {e}")
                continue

            # Fix gate if needed
            if model_params["smoe_dropout"]:
                freeze_gate_weight(model)

            # --- Evaluate on validation set ---
            with torch.no_grad():
                loss_val, counts, radio_val = full_eval(
                    model,
                    optimizer,
                    scheduler,
                    val_data_x,
                    val_data_y,
                    model_params["block_size"],
                    model_params["hidden_size"],
                    trainer_params["batch_split"],
                )

            if distributed and env_params["rank"] != 0:
                continue  # Only rank 0 logs

            if ("enwik8" in data_params["data_path"]) or ("text8" in data_params["data_path"]):
                logging("Val: {:.3f} BPC".format(loss_val / math.log(2)))
                logging("Val Radio: {:.3f} %".format(radio_val * 100))
            else:
                logging("Val: {:.3f} PPL".format(math.exp(loss_val)))
                logging("Val Radio: {:.3f} %".format(radio_val * 100))
            logging("Val Activate_expert: " + str(counts))

            # --- Evaluate on each test file ---
            test_dir = os.path.join(data_params["data_path"], "test")
            if not os.path.exists(test_dir):
                test_files = [os.path.join(data_params["data_path"], "test.json")]
            else:
                test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".json")])

            if not test_files:
                logging("Warning: No .json files found in test directory.")

            for test_file in test_files:
                test_file_path = os.path.join(test_dir, test_file)
                test_name = os.path.splitext(test_file)[0]

                logging(f"\n===== Evaluating on test file: {test_file} =====")
                try:
                    test_data_x, test_data_y = get_acc_data(
                        data_params=data_params,
                        env_params=env_params,
                        path=test_file_path,
                        block_size=model_params["block_size"],
                        batch_size=1,
                        device=device,
                    )
                except Exception as e:
                    logging(f"Failed to load {test_file}: {e}")
                    continue

                with torch.no_grad():
                    loss_test, counts, radio_test = full_eval(
                        model,
                        optimizer,
                        scheduler,
                        test_data_x,
                        test_data_y,
                        model_params["block_size"],
                        model_params["hidden_size"],
                        batch_split=1,
                    )

                if distributed:
                    stats = torch.tensor([loss_test]).to(device)
                    torch.distributed.reduce(stats, 0)
                    loss_test = stats[0].item() / env_params["world_size"]

                if ("enwik8" in data_params["data_path"]) or ("text8" in data_params["data_path"]):
                    logging(f"[{test_name}] Test BPC: {loss_test / math.log(2):.3f}")
                else:
                    logging(f"[{test_name}] Test PPL: {math.exp(loss_test):.3f}")
                logging(f"[{test_name}] Test Radio: {radio_test * 100:.3f} %")
                logging(f"[{test_name}] Activate_expert: {str(counts)}")

        return  # End of full_eval_mode

    # position of current batch
    data_pos = [0] * 2
    # initialize caches for train and valid
    hid_cache = [
        [
            torch.zeros(
                train_data_x.size(0),
                model.module.layers[layer_i].attn.attn.get_cache_size(),
                model_params["hidden_size"],
            ).to(device)
            for layer_i in range(model.module.attn_layer_count)
        ]
        for _ in range(2)
    ]

    nb_batches_per_iter = trainer_params["nb_batches_per_iter"]
    for iter_no in range(iter_init, trainer_params["nb_iter"]):
        # freq type
        if model_params["freq_type"] == "function":
            _threshold = 2.0 / (2.0 + math.sqrt((iter_no + 1)))
            set_freq_optimal_search(model, _threshold)

        # time storing
        t_sta = time.time()
        loss_train, data_pos[0], hid_cache[0],counts,radio_train = train_iteration(
            model,
            model_params["load_balance"],
            optimizer,
            scheduler,
            train_data_x,
            train_data_y,
            nb_batches_per_iter,
            model_params["block_size"],
            False,
            data_pos[0],
            hid_cache[0],
            trainer_params["batch_split"],
            trainer_params["checkpoint_path"],
            trainer_params["causal_loss"],
        )
        elapsed = 1000 * (time.time() - t_sta) / nb_batches_per_iter
        with torch.no_grad():
            loss_val, data_pos[1], hid_cache[1],counts,radio_val = train_iteration(
                model,
                model_params["load_balance"],
                optimizer,
                scheduler,
                val_data_x,
                val_data_y,
                nb_batches_per_iter,
                model_params["block_size"],
                True,
                data_pos[1],
                hid_cache[1],
                trainer_params["batch_split"],
                trainer_params["checkpoint_path"],
            )

        if distributed:
            # collect results into rank0
            stats = torch.tensor([loss_train, loss_val]).to(device)
            torch.distributed.reduce(stats, 0)
            if env_params["rank"] == 0:
                loss_train = stats[0] / env_params["world_size"]
                loss_val = stats[1] / env_params["world_size"]
            else:
                continue
        logging(f"=================== EPOCHS {iter_no} ======================")
        if ("enwik8" in data_params["data_path"]) or (
            "text8" in data_params["data_path"]
        ):
            msg_result = "Epochs: {} | radio_train: {:.3f} % | loss_train: {:.3f} ~ {:.3f} BPC | radio_val: {:.3f} % | loss_val: {:.3f} ~ {:.3f} BPC | elapsed: {:.1f}".format(
                iter_no,
                radio_train * 100,
                loss_train,
                float(loss_train / math.log(2)),
                radio_val * 100,
                loss_val,
                float(loss_val / math.log(2)),
                elapsed,
            )
        else:
            msg_result = "Epochs: {} | radio_train: {:.3f} % | loss_train: {:.3f} ~ {:.3f} PPL | radio_val: {:.3f} % | loss_val: {:.3f} ~ {:.3f} PPL | elapsed: {:.1f}".format(
                iter_no,
                radio_train * 100,
                loss_train,
                float(math.exp(loss_train)),
                radio_val * 100,
                loss_val,
                float(math.exp(loss_val)),
                elapsed,
            )
        logging(msg_result)
        # if wandb_flag:
        #     wandb.log({'train_ppl':float(math.exp(loss_train)),'Epoch':iter_no,'valid_ppl':float(math.exp(loss_val))})
        logger.log_iter(iter_no, nb_batches_per_iter, loss_train, loss_val, elapsed, model)
        # Save the model if the validation loss is the best we've seen so far.
        save_checkpoint(
            trainer_params["checkpoint_path"].replace(".pt", f"_epoch{iter_no}.pt"),
            iter_no,
            model,
            optimizer,
            scheduler,
            logger,
        )
    end_time = time.time()
    logging(f"Training time total: {(end_time - start_time)/3600} h")
    logging("Activate_expert: ")
    logging(str(counts))


if __name__ == "__main__":
    launch(**get_params(params_config=PARAMS_CONFIG))
