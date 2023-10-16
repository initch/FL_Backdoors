import argparse
import shutil
from datetime import datetime

import yaml
from prompt_toolkit import prompt
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from dataset.pipa import Annotations  # legacy to correctly load dataset.
from helper import Helper
from attack import Attack
from utils.utils import *
from neurotoxin import grad_mask_cv, apply_grad_mask

logger = logging.getLogger('logger')


def train(hlpr: Helper, local_attack: Attack, epoch, model, optimizer, train_loader, attack=False, neurotoxin=False):
    criterion = hlpr.task.criterion
    model.train()

    if attack and neurotoxin:
        mask_grad_list = grad_mask_cv(model, train_loader, criterion)

    for i, data in tqdm(enumerate(train_loader)):
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        loss = local_attack.compute_blind_loss(model, criterion, batch, attack)
        loss.backward()

        if attack and neurotoxin:
            apply_grad_mask(model, mask_grad_list)
            
        optimizer.step()

        hlpr.report_training_losses_scales(i, epoch)
        if i == hlpr.params.max_batch_id:
            break

    return loss.item()


def test(hlpr: Helper, epoch, backdoor=False):
    model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()

    if backdoor:
        with torch.no_grad():
            for i, data in tqdm(enumerate(hlpr.task.poisoned_test_loader)):
                batch = hlpr.task.get_batch(i, data)
                batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,test=True,attack=True)
                outputs = model(batch.inputs)
                hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    else:
        with torch.no_grad():
            for i, data in tqdm(enumerate(hlpr.task.test_loader)):
                batch = hlpr.task.get_batch(i, data)
                outputs = model(batch.inputs)
                hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    metric = hlpr.task.report_metrics(epoch,
                             prefix=f'Backdoor {str(backdoor):5s}. Epoch: ',
                             tb_writer=hlpr.tb_writer,
                             tb_prefix=f'Test_backdoor_{str(backdoor):5s}')

    return metric

def local_test(hlpr: Helper, local_attack: Attack, epoch, model, backdoor=False):
    model.eval()
    hlpr.task.reset_metrics()

    with torch.no_grad():
        for i, data in tqdm(enumerate(hlpr.task.test_loader)):
            batch = hlpr.task.get_batch(i, data)
            if backdoor:
                batch = local_attack.synthesizer.make_backdoor_batch(batch,
                                                                    test=True,
                                                                    attack=True)

            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    metric = hlpr.task.report_metrics(epoch,
                             prefix=f'Backdoor {str(backdoor):5s}. Epoch: ',
                             tb_writer=hlpr.tb_writer,
                             tb_prefix=f'Test_backdoor_{str(backdoor):5s}')

    return metric


def run(hlpr):
    acc = test(hlpr, 0, backdoor=False)
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        loss = train(hlpr, hlpr.attack, epoch, hlpr.task.model, hlpr.task.optimizer,
              hlpr.task.train_loader)
        acc = test(hlpr, epoch, backdoor=False)
        test(hlpr, epoch, backdoor=True)
        hlpr.save_model(hlpr.task.model, epoch, acc)
        if hlpr.task.scheduler is not None:
            if hlpr.params.scheduler == 'ReduceLROnPlateau':
                hlpr.task.scheduler.step(loss)
                logger.warning(f"Current LR: {hlpr.task.optimizer.param_groups[0]['lr']}")
            else:
                hlpr.task.scheduler.step(epoch)
        

def fl_run(hlpr: Helper):
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        run_fl_round(hlpr, epoch)
        metric = test(hlpr, epoch, backdoor=False)
        test(hlpr, epoch, backdoor=True)

        hlpr.save_model(hlpr.task.model, epoch, metric)


def run_fl_round(hlpr, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model

    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()

    i_mal = 0
    n_mal = hlpr.params.fl_number_of_adversaries
    for user in tqdm(round_participants):
        hlpr.task.copy_params(global_model, local_model)
        if user.compromised and epoch > hlpr.params.start_poison_epoch:
            i_mal += 1
            logger.warning(f'Attacker (client {user.user_id}) is poisoning on epoch {epoch}.')
            local_epochs = hlpr.params.fl_adv_local_epochs
            optimizer = hlpr.task.make_poison_optimizer(local_model)
        else:
            local_epochs = hlpr.params.fl_local_epochs
            optimizer = hlpr.task.make_optimizer(local_model)
        for local_epoch in range(local_epochs):
            if user.compromised and epoch > hlpr.params.start_poison_epoch:
                if hlpr.params.synthesizer.lower() == 'dba':
                    local_synthesizer = hlpr.make_synthesizer((i_mal, n_mal))
                    local_attack = Attack(hlpr.params, local_synthesizer)
                    train(hlpr, local_attack, local_epoch, local_model, optimizer,
                        user.train_loader, attack=True, neurotoxin=hlpr.params.neurotoxin)
                else:
                    train(hlpr, hlpr.attack, local_epoch, local_model, optimizer,
                        user.train_loader, attack=True, neurotoxin=hlpr.params.neurotoxin)
            else:
                train(hlpr, hlpr.attack, local_epoch, local_model, optimizer,
                        user.train_loader, attack=False)
        if user.compromised  and epoch > hlpr.params.start_poison_epoch:
            if hlpr.params.synthesizer.lower() == 'dba':
                local_test(hlpr, local_attack, epoch, local_model, backdoor=True)
            else:
                local_test(hlpr, hlpr.attack, epoch, local_model, backdoor=True)
        local_update = hlpr.task.get_fl_update(local_model, global_model)
        if user.compromised and epoch > hlpr.params.start_poison_epoch:
            hlpr.attack.fl_scale_update(local_update)
            # hlpr.attack.fl_scale_update(local_model, global_model)
        hlpr.task.accumulate_weights(weight_accumulator, local_update)
    
    logger.warning(f"Current LR: {optimizer.param_groups[0]['lr']}")

    hlpr.task.update_global_model(weight_accumulator, global_model)


if __name__ == '__main__':

    # torch.backends.cudnn.enabled = False
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', default='configs/cifar_fed.yaml')
    parser.add_argument('--name', dest='name', required=True)
    parser.add_argument('--commit', dest='commit', default=get_current_git_hash())

    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['commit'] = args.commit
    params['name'] = args.name

    helper = Helper(params)
    logger.warning(create_table(params))

    try:
        if helper.params.fl:
            fl_run(helper)
        else:
            run(helper)
    except (KeyboardInterrupt):
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                shutil.rmtree(helper.params.folder_path)
                if helper.params.tb:
                    shutil.rmtree(f'runs/{args.name}')
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}. "
                             f"TB graph: {args.name}")
        else:
            logger.error(f"Aborted training. No output generated.")
