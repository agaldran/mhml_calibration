import os, json, sys, time, random, os.path as osp
import numpy as np
import torch
from tqdm import trange
import medmnist
from utils.data_handling import get_medmnist_loaders
from utils.data_handling import get_class_loaders
from utils.evaluation import evaluate_cls
from utils.get_model_v2 import get_arch

def set_seed(seed_value, use_cuda):
    if seed_value is not None:
        np.random.seed(seed_value)  # cpu vars
        torch.manual_seed(seed_value)  # cpu  vars
        random.seed(seed_value)  # Python
        torch.use_deterministic_algorithms(True)

        if use_cuda:
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True

def save_model(path, model):
    os.makedirs(path, exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()},
                 osp.join(path, 'model_checkpoint.pth'))

def add_weight_decay(model, weight_decay, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_args_parser():
    import argparse

    def str2bool(v):
        # as seen here: https://stackoverflow.com/a/43357954/3208255
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', 'yes'):
            return True
        elif v.lower() in ('false', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError('boolean value expected.')

    medmnist_choices = ['pneumoniamnist', 'breastmnist', 'pathmnist', 'dermamnist', 'octmnist', 'retinamnist', 'bloodmnist',
                        'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist']

    parser = argparse.ArgumentParser(description='Training for Biomedical Image Classification')
    parser.add_argument('--medmnist_subdataset', type=str, default=None, choices = medmnist_choices, help='medmnist dataset')
    parser.add_argument('--csv_train', type=str, default='data/train_kather.csv', help='path to training data csv')
    parser.add_argument('--data_path', type=str, default='data/kather/', help='path to training images')
    parser.add_argument('--model', type=str, default='resnet18', help='architecture')
    parser.add_argument('--n_heads', type=int, default=1, help='if greater than 1, use NHeads Ensemble Learning')
    parser.add_argument('--balanced_mh', type=str2bool, nargs='?', const=True, default=False, help='Balance loss on heads')

    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=True, help='use imagenet weights')
    parser.add_argument('--cycle_lens', type=str, default='10/5', help='cycling config (nr cycles/cycle len)')

    parser.add_argument('--opt', default='sgd', type=str, choices=('sgd', 'adamw'), help='optimizer to use (sgd | adamW)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')  # adam -> 1e-4, 3e-4
    parser.add_argument('--momentum', default=0., type=float, help='sgd momentum')
    parser.add_argument('--epsilon', default=1e-8, type=float, help='adamW epsilon for numerical stability')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='label smoothing (default: 0.0)')

    parser.add_argument('--metric', type=str, default='auc', help='which metric to use for monitoring progress (auc)')
    parser.add_argument('--im_size', help='delimited list input, could be 500, or 600,400', type=str, default='224')
    parser.add_argument('--save_path', type=str, default=None, help='path to save model (defaults to None => debug mode)')
    parser.add_argument('--num_workers', default=6, type=int, help='number of data loading workers (default: 6)')
    parser.add_argument('--device', default='cuda', type=str, help='device (cuda or cpu, default: cuda)')
    parser.add_argument('--seed', type=int, default=None, help='fixes random seed (slower!)')

    # parser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=True, help='avoid saving anything')

    args = parser.parse_args()

    return args

def run_one_epoch(model, optimizer, ce_weights, loader, scheduler=None, assess=False):

    device='cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here

    if train: model.train()
    else: model.eval()
    ce = torch.nn.functional.cross_entropy
    probs_class_all, preds_class_all, labels_all = [], [], []
    run_loss_class = 0
    weights = [torch.tensor(w, dtype=torch.float32).to(device) for w in ce_weights]
    # w0 = torch.tensor(ce_weights[0], dtype=torch.float32).to(device)
    # w1 = torch.tensor(ce_weights[1], dtype=torch.float32).to(device)
    with trange(len(loader)) as t:
        n_elems, running_loss_class = 0, 0
        for i_batch, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.squeeze().to(device)
            logits = model(inputs)
            if model.n_heads == 1:
                loss_class = ce(logits, labels)
            elif model.n_heads == 2:
#                 if train:
#                     loss_class = ce(logits[:, 0, :], labels, weights[0]) + ce(logits[:, 1, :], labels, weights[1])
#                     loss_class/=2
#                 else:
#                     loss_class = ce(logits, labels)
                if train:
                    loss_class = ce(logits[:, 0, :], labels, weights[0]) + ce(logits[:, 1, :], labels, weights[1])
                    # loss_class/=2
                    overall_loss = ce(logits.mean(dim=1), labels)
                    loss_class = (loss_class+overall_loss)/3
                else:
                    loss_class = ce(logits, labels)
                    
#             elif model.n_heads == 4:
#                 if train:
#                     loss_class = ce(logits[:, 0, :], labels, weights[0]) + ce(logits[:, 1, :], labels, weights[1]) + \
#                                  ce(logits[:, 2, :], labels, weights[2]) + ce(logits[:, 3, :], labels, weights[3])
#                     loss_class/=4
#                 else:
#                     loss_class = ce(logits, labels)

            elif model.n_heads == 4:
                if train:
                    loss_class = ce(logits[:, 0, :], labels, weights[0]) + ce(logits[:, 1, :], labels, weights[1]) + \
                                 ce(logits[:, 2, :], labels, weights[2]) + ce(logits[:, 3, :], labels, weights[3])
                    overall_loss = ce(logits.mean(dim=1), labels)
                    loss_class = (loss_class+overall_loss)/5
                else:
                    loss_class = ce(logits, labels)
                    
            
            else: sys.exit('need to generalize to more heads but no time for everything')
            if train:  # only in training mode
                loss_class.backward()
                optimizer.step()
                lr = get_lr(optimizer)
                scheduler.step()
                optimizer.zero_grad()
            if assess:
                probs_class = logits.softmax(dim=1)
                # if model.n_heads == 1:
                #     probs_class = logits.softmax(dim=1)
                # else:
                #     probs_class = logits.mean(dim=-2).softmax(dim=1)  # take average over heads
                preds_class = np.argmax(probs_class.detach().cpu().numpy(), axis=1)
                probs_class_all.extend(probs_class.detach().cpu().numpy())
                preds_class_all.extend(preds_class)
                labels_all.extend(labels.cpu().numpy())
            # Compute running loss
            running_loss_class += loss_class.detach().item() * inputs.size(0)
            n_elems += inputs.size(0)
            run_loss_class = running_loss_class / n_elems
            if train: t.set_postfix(loss_lr="{:.4f}/{:.6f}".format(run_loss_class, lr))
            else: t.set_postfix(vl_loss="{:.4f}".format(float(run_loss_class)))
            t.update()
    if train: print('Class Loss= {:.4f} -- LR = {:.7f}'.format(loss_class, lr))
    if assess: return np.stack(preds_class_all), np.stack(probs_class_all), np.stack(labels_all), run_loss_class
    return None, None, None, None

def train_one_cycle(model, optimizer, ce_weights, train_loader, scheduler, cycle=0):

    model.train()
    optimizer.zero_grad()
    cycle_len = scheduler.cycle_lens[cycle]

    for epoch in range(cycle_len):
        print('Cycle {:d} | Epoch {:d}/{:d}'.format(cycle+1, epoch+1, cycle_len))
        if epoch == cycle_len-1: assess=True # only get probs/preds/labels on last cycle
        else: assess = False
        tr_preds, tr_probs, tr_labels, tr_loss = run_one_epoch(model, optimizer, ce_weights, train_loader, scheduler, assess)

    return tr_preds, tr_probs, tr_labels, tr_loss

def train_model(model, optimizer, ce_weights, train_loader, val_loader, scheduler, metric, save_path):

    n_cycles = len(scheduler.cycle_lens)
    best_loss, best_auc, best_acc, best_cycle = 10, 0, 0, 0
    all_tr_aucs, all_vl_aucs, all_tr_accs, all_vl_accs = [], [], [], []
    all_tr_losses, all_vl_losses = [], []

    n_classes = len(train_loader.dataset.classes)
    class_names = ['C{}'.format(i) for i in range(n_classes)]
    print_conf, text_file_train, text_file_val = False, None, None

    for cycle in range(n_cycles):
        print('\nCycle {:d}/{:d}'.format(cycle+1, n_cycles))
        # train one cycle
        _, _, _, _ = train_one_cycle(model, optimizer, ce_weights, train_loader, scheduler, cycle=cycle)

        with torch.inference_mode():
            tr_preds, tr_probs, tr_labels, tr_loss = run_one_epoch(model, None, np.ones_like(ce_weights), train_loader, assess=True)
            vl_preds, vl_probs, vl_labels, vl_loss = run_one_epoch(model, None, np.ones_like(ce_weights), val_loader, assess=True)

        if save_path is not None:
            print_conf = True
            text_file_train = osp.join(save_path,'performance_cycle_{}.txt'.format(str(cycle+1).zfill(2)))
            text_file_val = osp.join(save_path, 'performance_cycle_{}.txt'.format(str(cycle+1).zfill(2)))

        tr_auc, tr_f1, tr_acc, tr_auc_all, tr_f1_all = evaluate_cls(tr_labels, tr_preds, tr_probs, print_conf=print_conf,
                                                              class_names=class_names, text_file=text_file_train, loss=tr_loss)
        vl_auc, vl_f1, vl_acc, vl_auc_all, vl_f1_all = evaluate_cls(vl_labels, vl_preds, vl_probs, print_conf=print_conf,
                                                              class_names=class_names, text_file=text_file_val, loss=vl_loss)

        print('Train||Val Loss: {:.4f}||{:.4f} - AUC: {:.2f}||{:.2f} - ACC: {:.2f}||{:.2f}'.format(tr_loss, vl_loss,
                                                                                                   100 * tr_auc, 100 * vl_auc,
                                                                                                   100 * tr_acc, 100 * vl_acc))
        all_tr_aucs.append(tr_auc)
        all_vl_aucs.append(vl_auc)
        all_tr_accs.append(tr_acc)
        all_vl_accs.append(vl_acc)
        all_tr_losses.append(tr_loss)
        all_vl_losses.append(vl_loss)

        # check if performance was better than anyone before and checkpoint if so
        if vl_auc > best_auc:
            print('-------- Best {} attained. {:.2f} --> {:.2f} --------'.format(metric, 100*best_auc, 100*vl_auc))
            best_loss, best_auc, best_cycle = vl_loss, vl_auc, cycle+1
            best_auc = vl_auc
            best_acc = vl_acc
            if save_path is not None: save_model(save_path, model)
        else:
            print('-------- Best AUC so far {:.2f} at cycle {:d} --------'.format(100 * best_auc, best_cycle))


    del model
    torch.cuda.empty_cache()
    return best_auc, best_acc, best_loss, all_tr_aucs, all_vl_aucs, all_tr_accs, all_vl_accs, all_tr_losses, all_vl_losses, best_cycle

def main(args):
    use_cuda = args.device=='cuda' and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # reproducibility
    set_seed(args.seed, use_cuda)

    save_path = args.save_path
    if save_path is not None:
        save_path=osp.join('experiments', save_path)
        args.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        config_file_path = osp.join(save_path,'config.cfg')
        with open(config_file_path, 'w') as f:
            json.dump(vars(args), f, indent=2)

    # Prepare training data
    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    csv_train = args.csv_train
    csv_val = csv_train.replace('train', 'val')
    if args.medmnist_subdataset is not None:
        train_loader, val_loader = get_medmnist_loaders(args.medmnist_subdataset, args.batch_size, args.num_workers, tg_size=tg_size)
        num_classes = len(medmnist.INFO[args.medmnist_subdataset]['label'])
    else:
        train_loader, val_loader = get_class_loaders(csv_train, csv_val, args.data_path, tg_size, args.batch_size, args.num_workers)
        num_classes = len(train_loader.dataset.classes)
    n_heads = args.n_heads

    # Prepare model for training
    model = get_arch(args.model, num_classes, n_heads)
    model.to(device)

    # Prepare optimizer and scheduler
    weight_decay = 0
    if weight_decay > 0:
        # it's okay to use weight decay, but do not apply to it normalization layers
        # https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3
        parameters = add_weight_decay(model, weight_decay)
        weight_decay = 0.
    else:
        parameters = model.parameters()


    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay)
    else:
        raise RuntimeError('Invalid optimizer {}. Only SGD and AdamW are supported.'.format(args.opt))


    cycle_lens, metric = args.cycle_lens.split('/'), args.metric
    cycle_lens = list(map(int, cycle_lens))
    if len(cycle_lens) > 2:
        sys.exit('cycles should be specified as a pair n_cycles/cycle_len')
    cycle_lens = cycle_lens[0] * [cycle_lens[1]]

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cycle_lens[0]*len(train_loader), eta_min=0)

    setattr(optimizer, 'max_lr', args.lr)  # store maximum lr inside the optimizer for accessing to it later
    setattr(scheduler, 'cycle_lens', cycle_lens)

    # Start training
    start = time.time()
    classes = np.arange(num_classes)
    random.shuffle(classes)
    more_weighted_classes_per_head = np.array_split(classes, n_heads)
    ce_weights = []
    for c in more_weighted_classes_per_head:
        # weights for this head are 2 for classes in c and 1/2 for classes that are not in c
        w = [2 if i in c else 1/2 for i in np.arange(num_classes)]
        w = [n_heads if i in c else 1/n_heads for i in np.arange(num_classes)]
        if args.balanced_mh:
            # do not use weighted-CE, so this is an Unperturbed Bycephal
            w = num_classes*[1]
        ce_weights.append(w)

    print('Using class weights: ', ce_weights)
    best_auc, best_acc, best_loss, all_tr_aucs, all_vl_aucs, all_tr_accs, all_vl_accs, all_tr_losses, all_vl_losses, best_cycle = \
        train_model(model, optimizer, ce_weights, train_loader, val_loader, scheduler, metric, save_path)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('done')

    if save_path is not None:
        with open(osp.join(save_path, 'log.txt'), 'w') as f:
            print('Best AUC = {:.2f}\nBest ACC = {:.2f}\nBest loss = {:.2f}\nBest cycle = {}\n'.format(100*best_auc, 100*best_acc, best_loss, best_cycle), file=f)

            for j in range(len(all_tr_aucs)):
                print('Cycle = {} -> AUC={:.2f}/{:.2f}, ACC={:.2f}/{:.2f}, Loss={:.4f}/{:.4f}'.format(j+1,
                            100*np.mean(all_tr_aucs[j]), 100*np.mean(all_vl_aucs[j]),
                            100*np.mean(all_tr_accs[j]), 100*np.mean(all_vl_accs[j]),
                            all_tr_losses[j], all_vl_losses[j]), file=f)

            print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)

if __name__ == "__main__":
    args = get_args_parser()
    main(args)
