import os
import argparse
import time
import ast
import json
from comet_ml import Experiment, ExistingExperiment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.parallel as parallel
import torch.multiprocessing as mp
import torch.distributed as distributed
import torch.utils.data as data
from dataset import Data, collate_fn_padd, collate_fn_padd_order
from utils import GreedyDecoder
from wer2 import cer, wer
import horovod.torch as hvd

from utils import AttrDict
from e2e_model import Transducer
import yaml
import pdb


def setup_comet_ml(args, rank):
    # dummy init of experiment so it can be used without error
    # even if comet is disabled
    experiment = Experiment(api_key='dummy_key', disabled=True)
    if args.comet_api_key:
        # initiating comet
        if args.existing_exp_key:
            if rank == 0:
                print("STARTING FROM AND EXISTING EXPERIMENT")
            experiment = ExistingExperiment(
                api_key=args.comet_api_key, workspace=args.comet_workspace,
                project_name=args.project_name, previous_experiment=args.existing_exp_key,
                auto_output_logging="simple", auto_metric_logging=False, parse_args=False,
                disabled=args.disable_comet or rank != 0)
        else:
            if rank == 0:
                print("STARTING A NEW EXPERIMENT")
            experiment = Experiment(
                api_key=args.comet_api_key, workspace=args.comet_workspace,
                project_name=args.project_name, auto_output_logging="simple", auto_metric_logging=False,
                parse_args=False, disabled=args.disable_comet or rank != 0)

    experiment.log_asset('config.yaml')
    experiment.log_asset('config_prod.yaml')
    experiment.log_asset('config_prod_prime.yaml')

    return experiment


def save_checkpoint(args, model, optimizer, scheduler, epoch, i, total_iter, config):
    CHECKPOINT_PATH = os.path.join(config.dist_train.checkpoint_dir, config.comet_info.exp_name, 'checkpoint-{}.tar'.format(total_iter))
    if CHECKPOINT_PATH != args.resume_from:
        # remove duplicate before saving
        # to prevent corrupt files
        if os.path.isfile(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
        print("saving checkpoint to", CHECKPOINT_PATH)
        torch.save({
            "epoch": epoch,
            "step": i,
            "total_iter": total_iter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
        }, CHECKPOINT_PATH)

    # remove checkpoint more then args.max_checkpoints are saved
    checkpoints = sorted(filter(os.path.isfile, os.listdir('.')), key=os.path.getmtime)
    checkpoints = [c for c in checkpoints if "checkpoint-" in c]
    if len(checkpoints) > args.max_checkpoints:
        rm_path = os.path.join(config.dist_train.checkpoint_dir, config.comet_info.exp_name, checkpoints[0])
        os.remove(rm_path)



def compute_error_rate(preds, world_size, gpu):
    print("CALCULATING CHARACTER & WORD ERROR RATE")
    all_cers = []
    all_wers = []
    for i, (output, labels, label_lengths) in enumerate(preds):
        start = time.time()
        decoded_pred, decoded_targets = GreedyDecoder(output, labels, label_lengths)
        cers = []
        wers = []
        for j in range(len(decoded_pred)):
            _cer = hvd.allreduce(torch.Tensor([cer(decoded_targets[j], decoded_pred[j])]).cuda()).item()
            _wer = hvd.allreduce(torch.Tensor([wer(decoded_targets[j], decoded_pred[j])]).cuda()).item()

            cers.append(_cer)
            wers.append(_wer)
        avg_cer = sum(cers) / len(cers)
        avg_wer = sum(wers) / len(wers)
        all_cers.append(avg_cer)
        all_wers.append(avg_wer)
        if gpu == 0:
            print("Iter", i, "AVG CER:", avg_cer, "time:", time.time() - start)
            print("Iter", i, "AVG WER:", avg_wer, "time:", time.time() - start)

    # # wait for all others to finish calculating wer
    # hvd.allreduce(torch.tensor(0), name='barrier')

    avg_all_cers = sum(all_cers) / len(all_cers)
    avg_all_wers = sum(all_wers) / len(all_wers)
    print("AVG CER of all valid is", avg_all_cers)
    print("AVG WER of all valid is", avg_all_wers)
    return avg_all_cers, avg_all_wers


def evaluate(args, gpu, model, test_loader, tokenizer=None, calc_wer=False):
    print("STARTING VALIDATION")
    model.eval()
    valid_loss = []
    wers = []
    cers = []
    total_step = len(test_loader)

    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            start = time.time()
            spectrograms, labels, input_lengths, label_lengths = _data

            # calculate wer and cer
            if calc_wer:
                for j, xs in enumerate(spectrograms):
                    sample_input_length = int(input_lengths[j])
                    sample_label_length = int(label_lengths[j])
                    sample_spec = torch.unsqueeze(xs[:sample_input_length, :], 0)
                    sample_label = torch.unsqueeze(labels[j][:sample_label_length], 0)
                    #sample_spec.shape = [1, 490, 81] float32
                    #sample_input_length = [490] int

                    y, nll = model.greedy_decode1(sample_spec.cuda(), [sample_input_length])

                    yt = tokenizer.decode(y)

                    ref = sample_label[0].tolist()
                    ref = [int(x) for x in ref]

                    rt = tokenizer.decode(ref)

                    er, _, _ = wer(rt, yt)
                    wers.append(er)
                    cers.append(cer(rt, yt))

            # calculate loss

            input_lengths = torch.tensor(input_lengths, dtype=torch.int32)
            label_lengths = torch.tensor(label_lengths, dtype=torch.int32)
            labels = torch.tensor(labels, dtype=torch.int64)

            spectrograms, input_lengths = spectrograms.cuda(), input_lengths.cuda()
            labels, label_lengths = labels.cuda(), label_lengths.cuda()


            loss = model(spectrograms, labels, input_lengths, label_lengths)

            loss_val = hvd.allreduce(torch.Tensor([loss.item()])).item()
            valid_loss.append(loss_val)

            if gpu == 0:
                print('Step [{}/{}], Loss: {:.4f}, in_shape: {}, time: {:.2f}'.format(
                    i + 1, total_step, loss_val, spectrograms.shape, time.time() - start))

    avg_valid_loss = sum(valid_loss) / len(valid_loss)
    avg_wer = sum(wers) / len(wers) if calc_wer else None
    avg_cer = sum(cers) / len(cers) if calc_wer else None
    print("#" * 10, "loss", avg_valid_loss, "#" * 10)

    return avg_valid_loss, avg_wer, avg_cer


def train(args, gpu, experiment, model, optimizer, scheduler, train_loader, test_loader, train_sampler, config):
    total_step = len(train_loader)
    model.train()
    total_iter = args.total_iter
    try:
        for epoch in range(args.start_epoch, (config.dist_train.epochs + 1)):
            model.zero_grad()
            train_sampler.set_epoch(epoch - 1)
            losses = []
            for i, _data in enumerate(train_loader, start=args.start_step):
                start = time.time()
                spectrograms, labels, input_lengths, label_lengths = _data

                input_lengths = torch.tensor(input_lengths, dtype=torch.int32)
                label_lengths = torch.tensor(label_lengths, dtype=torch.int32)
                labels = torch.tensor(labels, dtype=torch.int64)

                spectrograms, input_lengths = spectrograms.cuda(), input_lengths.cuda()
                labels, label_lengths = labels.cuda(), label_lengths.cuda()                # print(spectrograms.shape)

                # spectrograms.shape  = [2,461, 81]   tensor float32 cuda
                # labels.shape        = [2,16]        tensor int64 cuda
                # input_lengths       = [461, 433]    tensor int32
                # label_lengths       = [16, 12]      tensor int32 cuda

                if i % config.dist_train.grad_acc_steps == 0 and config.model.enc.noisy:
                    model.encoder.add_noise()

                loss = model(spectrograms, labels, input_lengths, label_lengths)
                loss_val = hvd.allreduce(torch.Tensor([loss.item()]).cuda()).item()

                losses.append(loss_val)
                loss = loss / config.dist_train.grad_acc_steps
                loss.backward()
                experiment.log_metric('loss', loss_val, step=total_iter, epoch=epoch + 1)
                experiment.log_metric('learning_rate', scheduler.get_lr(), step=total_iter, epoch=epoch + 1)
                # print('lr', scheduler.get_lr(), '\n')
                if gpu == 0:
                    print('Iter [{}/{}], Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, in_shape: {}, time: {:.2f}'.format(
                        total_iter, int((total_step * config.dist_train.epochs) / config.dist_train.grad_acc_steps), epoch,
                        config.dist_train.epochs, i, total_step, loss_val, spectrograms.shape, time.time() - start
                    ))

                if i % config.dist_train.grad_acc_steps == 0 or i == total_step:
                    optimizer.synchronize()  # https://horovod.readthedocs.io/en/latest/api.html#horovod.torch.DistributedOptimizer
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    with optimizer.skip_synchronize():
                        optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    total_iter += 1

                    if total_iter != 0:
                        # validate
                        if (total_iter % config.dist_train.valid_every) == 0 or (total_iter % config.dist_train.error_rate_every) == 0:
                            if (total_iter % config.dist_train.error_rate_every) == 0:
                                avg_valid_loss, avg_wer, avg_cer = evaluate(
                                    args, gpu, model, test_loader,
                                    tokenizer=test_loader.dataset.tokenizer,
                                    calc_wer=True
                                )
                                experiment.log_metric('cer', avg_cer, step=total_iter, epoch=epoch + 1)
                                experiment.log_metric('wer', avg_wer, step=total_iter, epoch=epoch + 1)
                            else:
                                avg_valid_loss, _, _ = evaluate(args, gpu, model, test_loader)
                            experiment.log_metric('mean_train_loss', sum(losses) / len(losses), step=total_iter,
                                                  epoch=epoch + 1)
                            experiment.log_metric('valid_loss', avg_valid_loss, step=total_iter, epoch=epoch + 1)
                            model.train()
                            losses = []  # empty losses
                        # save cmodel checkpoint
                        if (total_iter % config.dist_train.checkpoint_every) == 0:
                            if args.rank == 0:
                                save_checkpoint(args, model, optimizer, scheduler, epoch, i, total_iter, config)
                            # distributed.barrier()  # wait for model to save befor continuing

                    # restart epoch. needed when restart from checkpoint
                    if i > total_step:
                        args.start_step = 1
                        break


    except UnboundLocalError as e:
        print(str(e))

    # training is complete, save model and evaluate one last time
    if args.rank == 0:
        save_checkpoint(args, model, optimizer, scheduler, epoch, i, total_iter)
    # distributed.barrier()
    avg_valid_loss, avg_wer, avg_cer = evaluate(
        args, gpu, model, test_loader,
        tokenizer=test_loader.dataset.tokenizer,
        calc_wer=True
    )
    experiment.log_metric('cer', avg_cer, step=total_iter, epoch=epoch + 1)
    experiment.log_metric('wer', avg_wer, step=total_iter, epoch=epoch + 1)
    experiment.log_metric('valid_loss', avg_valid_loss, step=total_iter, epoch=epoch + 1)
    experiment.log_metric('mean_train_loss', sum(losses) / len(losses), step=total_iter, epoch=epoch + 1)

    if gpu == 0:
        print("Training is complete")


def startup(gpu, args, config):
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    torch.manual_seed(7)
    args.rank = hvd.rank()
    # args.rank = args.nr * args.gpus + gpu
    # print('rank', args.rank,  'gpu', gpu, 'worldsize', args.world_size)
    # distributed.init_process_group(
    #     backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
    torch.set_num_threads(1)

    experiment = setup_comet_ml(args, args.rank)

    # model
    model = Transducer(config)

    if config.model.random_init:
        for param in model.parameters():
            torch.nn.init.uniform(param, -0.1, 0.1)

    model.preload(config.model.preload_from)
    model.preload_lm(config.model.dec.pretrain_file)

    model.cuda()

    # torch.cuda.set_device(gpu)

    # model = parallel.DistributedDataParallel(model, device_ids=[gpu])

    # data
    d_params = Data.parameters
    d_params['freq_mask'] = config.data.freq_mask
    d_params['time_mask'] = config.data.time_mask

    train_dataset = Data(
        mean=config.data.mean, std=config.data.std,
        json_path=config.data.train_json,
        order_time_feature=True,
        tokenizer=config.model.tokenizer,
        bpe_size=config.model.bpe_size,
        cache_dir=config.model.bpe_cache_dir,
        adaptive_specaug=config.data.adaptive_specaug,
        time_repeats=config.data.time_repeats,
        **d_params
    )

    test_dataset = Data(
        mean=config.data.mean, std=config.data.std,
        json_path=config.data.valid_json,
        order_time_feature=True,
        tokenizer=config.model.tokenizer,
        bpe_size=config.model.bpe_size,
        cache_dir=config.model.bpe_cache_dir,
        adaptive_specaug=config.data.adaptive_specaug,
        time_repeats=config.data.time_repeats,
        **d_params, valid=True
    )

    train_sampler = data.distributed.DistributedSampler(train_dataset,
                                                        num_replicas=hvd.size(),
                                                        rank=hvd.rank())
    test_sampler = data.distributed.DistributedSampler(test_dataset,
                                                       num_replicas=hvd.size(),
                                                       rank=hvd.rank())

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=config.dist_train.batch_size,
                                   shuffle=train_sampler is None,
                                   num_workers=args.data_workers,
                                   pin_memory=True,
                                   collate_fn=collate_fn_padd if config.model.type == "BasicRNNT" else collate_fn_padd_order,
                                   drop_last=True,
                                   sampler=train_sampler)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=config.dist_train.batch_size,
                                  shuffle=False,
                                  num_workers=args.data_workers,
                                  collate_fn=collate_fn_padd if config.model.type == "BasicRNNT" else collate_fn_padd_order,
                                  drop_last=True,
                                  pin_memory=True,
                                  sampler=test_sampler)

    # freeze encoder if specified
    if config.model.enc.freeze:
        for param in model.encoder.parameters():
            param.requires_grad = False


    # define optimizer and loss
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.dist_train.learning_rate,
        weight_decay=config.dist_train.weight_decay
    )

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.dist_train.learning_rate,
                                              steps_per_epoch=int(len(train_loader) / config.dist_train.grad_acc_steps),
                                              epochs=config.dist_train.epochs,
                                              div_factor=config.dist_train.div_factor,
                                              pct_start=config.dist_train.pct_start,
                                              anneal_strategy='linear')

    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        backward_passes_per_step=config.dist_train.grad_acc_steps,
        op=hvd.Adasum if args.use_adasum else hvd.Average)

    if args.load_model_from and args.rank == 0:
        print("LOADING MODEL FROM", args.load_model_from)
        checkpoint = torch.load(args.load_model_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # logging in terminal and comet
    h_params = {}

    h_params.update({
        "batch_size": config.dist_train.batch_size,
        "grad_acc": config.dist_train.grad_acc_steps,
        "virtual_batch_size": config.dist_train.batch_size * config.dist_train.grad_acc_steps,
        "learning_rate": config.dist_train.learning_rate,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
    })
    num_params = sum([param.nelement() for param in model.parameters()])
    if args.rank == 0:
        print(model)
        print(h_params)
        print(d_params)
        print('number of model parameters: ', num_params)
        print("\n train dataset summary \n", train_dataset.describe())
        print("\n test dataset summary \n", test_dataset.describe())
        print("\n data transforms \n", train_dataset.audio_transforms, test_dataset.audio_transforms)
    if args.rank == 0:
        experiment.log_parameters(h_params)
        experiment.log_parameters(d_params)
        experiment.set_name(config.comet_info.exp_name)  # experiment name
        experiment.log_others(vars(args))
        experiment.log_other('train_summary', str(train_dataset.describe()))
        experiment.log_other('test_summary', str(test_dataset.describe()))
        experiment.log_other('train_data_transforms', str(train_dataset.audio_transforms))
        experiment.log_other('valid_data_transforms', str(test_dataset.audio_transforms))
        experiment.log_other('n_model_params', num_params)

    # save args to file
    if args.rank == 0:
        ckpt_dir = os.path.join(config.dist_train.checkpoint_dir, config.comet_info.exp_name)
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        params_file = os.path.join(ckpt_dir, "args.txt")
        pretty_json = json.dumps(vars(args), sort_keys=True, indent=4)
        with open(params_file, 'w+') as f:
            f.write(pretty_json)
        print(pretty_json)

    # # resume from checkpoint
    # distributed.barrier()  # block processes until enter loading
    args.start_epoch = 1
    args.start_step = 1
    args.total_iter = 0
    if args.resume_from:
        if args.rank == 0:
            print("LOADING FROM CHECKPOINT...", args.resume_from)
            checkpoint = torch.load(args.resume_from, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.cuda()
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            args.start_epoch = checkpoint['epoch']
            if args.resume_step > 0:
                args.start_step = args.resume_step
            else:
                args.start_step = checkpoint['step']
            args.total_iter = checkpoint['total_iter']
        # distributed.barrier()  # block process until finish loading
        print('broadcasting model state')
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        print('broadcasting optimizer state..')
        # new_optimizer_state = hvd.broadcast_object(optimizer.state_dict(), root_rank=0)
        # optimizer.load_state_dict(new_optimizer_state)
        print('broadcasting scheduler state..')
        new_scheduler_state = hvd.broadcast_object(scheduler.state_dict(), root_rank=0)
        scheduler.load_state_dict(new_scheduler_state)
        print('broadcasting other args')
        args.start_epoch = hvd.broadcast_object(args.start_epoch, root_rank=0)
        args.start_step = hvd.broadcast_object(args.start_step, root_rank=0)
        args.total_iter = hvd.broadcast_object(args.total_iter, root_rank=0)

    train(args, args.rank, experiment, model, optimizer, scheduler, train_loader, test_loader, train_sampler, config)


def main():
    parser = argparse.ArgumentParser("Hulk Training Module")

    # distributed training setup
    parser.add_argument('-n', '--nodes', default=1, type=int, help='number of data loading workers')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('-w', '--data_workers', default=0, type=int,
                        help='n data loading workers, default 0 = main process only')
    parser.add_argument('-ma', '--master_adress', default='127.0.0.1', type=str,
                        help='ip address of master machine')
    parser.add_argument('-mp', '--master_port', default='29500', type=str,
                        help='port to access of master machine')

    # train and valid
    parser.add_argument('--max_checkpoints', default=50, required=False, type=int,
                        help='save the last N checkpoints')
    parser.add_argument('--resume_from', default=None, required=False, type=str,
                        help='checkpoint to resume from')
    parser.add_argument('--load_model_from', default=None, required=False, type=str,
                        help='checkpoint to load model. This start training from scratch with pretrained model')

    # general
    parser.add_argument('--eval_batch', default=-1, type=int,
                        help='size of eval batch. if not set it will be double the train batch size')
    parser.add_argument("--hparams_override", default="{}", type=str, required=False,
                        help='override the hyper parameters, should be in form of dict. ie. {"attention_layers": 16 }')
    parser.add_argument("--dparams_override", default="{}", type=str, required=False,
                        help='override the data parameters, should be in form of dict. ie. {"sample_rate": 8000 }')
    parser.add_argument('--model', default='AssemblySelfAttentionCTC', required=False, type=str,
                        help='model to run')
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')
    parser.add_argument('--div_factor', default=100, type=int, help='factor to divide learningrate by')
    parser.add_argument('--resume_step', default=-1, type=int)

    # comet_ml
    parser.add_argument("--comet_api_key", default=None, type=str, required=False,
                        help='comet_ml api key')
    parser.add_argument("--comet_workspace", default="assemblyai", type=str, required=False,
                        help='comet_ml workspace to upload to')
    parser.add_argument("--project_name", default=None, type=str, required=False,
                        help='name of project of this exp. Use for organization in comet_ml')
    parser.add_argument("--existing_exp_key", default=None, type=str, required=False,
                        help='key to continue from existing experiment')
    parser.add_argument("--disable_comet", action='store_true', required=False,
                        help='disable uploading to comet ml')

    parser.add_argument('--config', required=True, type=str)

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = args.master_adress
    os.environ['MASTER_PORT'] = args.master_port

    args.hparams_override = ast.literal_eval(args.hparams_override)
    args.dparams_override = ast.literal_eval(args.dparams_override)


    configfile = open(args.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))


    startup(args.gpus, args, config)


if __name__ == '__main__':
    main()