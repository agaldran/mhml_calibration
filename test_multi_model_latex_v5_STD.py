#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import log_loss
import random, os.path as osp
import numpy as np
import torch
import medmnist
from utils.data_handling import get_medmnist_test_loader
from utils.data_handling import get_class_loaders
from utils.evaluation import evaluate_cls
from utils.get_model_v2 import get_arch
from torchmetrics.classification import MulticlassCalibrationError
from utils.data_handling import get_class_test_loader
from utils.data_handling import get_medmnist_loaders
import warnings
warnings.warn = warn
# In[ ]:



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


# In[ ]:






# In[ ]:


def test_one_epoch(model, loader, device):
    model.to(device)
    model.eval()
    probs_all, labels_all = [], []

    for i_batch, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        logits = model(inputs)  # bs x n_classes
        probs = logits.softmax(dim=1).detach().cpu().numpy()
        labels = labels.numpy()
        if labels.ndim == 0:  # for 1-element batches labels degenerates to a scalar
            labels = np.expand_dims(labels, 0)
        probs_all.extend(probs)
        labels_all.extend(list(labels))

    return np.stack(probs_all), np.array(labels_all).squeeze()

def test_cls(model, test_loader, device):

    with torch.inference_mode():
        probs, labels = test_one_epoch(model, test_loader, device)

    del model
    torch.cuda.empty_cache()

    return probs, labels


# In[ ]:


def test_one_epoch_multihead(model, loader, device):
    ## IMPORTANT: Note that multi-head models in test time return softmax-activated tensors and not logits
    ## which is shitty because I cannot apply TS directly
    model.to(device)
    model.eval()

    probs_all, labels_all = [], []

    for i_batch, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        probs = model(inputs).detach().cpu().numpy()

        labels = labels.numpy()
        if labels.ndim == 0:  # for 1-element batches labels degenerates to a scalar
            labels = np.expand_dims(labels, 0)
        probs_all.extend(probs)
        labels_all.extend(list(labels))

    return np.stack(probs_all), np.array(labels_all).squeeze()

def test_cls_multihead(model, test_loader, device):
    with torch.inference_mode():
        probs, labels = test_one_epoch_multihead(model, test_loader, device)

    del model
    torch.cuda.empty_cache()

    return probs, labels


# In[ ]:


def warn(*args, **kwargs):
    pass



# In[ ]:


from sklearn.metrics import accuracy_score as acc


# In[ ]:


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')


# In[ ]:


def print_results_multi_fold(dataset=None, model_name='convnext', n_bins=15, method='sl1h', with_ens=False):
    assert dataset is not None
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    seed=0
    set_seed(seed, use_cuda)
    if model_name in ['convnext', 'swin'] :
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = True

    assert dataset in ['chaoyang', 'kvasir', 'pathmnist']

    if dataset=='pathmnist':
        tg_size = 28 if model_name !='convnext' else 32
        test_loader = get_medmnist_test_loader(dataset, batch_size=128, num_workers=6, tg_size=tg_size)
        class_names = list(medmnist.INFO[dataset]['label'].keys())
        num_classes = len(class_names)

            
    else:
        data_path = osp.join('data', dataset)
        csv_test = osp.join('data', 'test_'+dataset+'.csv')
        tg_size = 224,224
        test_loader  = get_class_test_loader(csv_test, data_path, tg_size, batch_size=64, num_workers=6)
        num_classes = len(test_loader.dataset.classes)
        class_names = ['C{}_probs'.format(i) for i in range(num_classes)]

        
    load_path = osp.join('experiments', dataset, method)
    
    ########################
    # results for individual models
    ########################    
    if model_name=='convnext':
        load_path_this = osp.join(load_path, 'cnx_f')        
    elif model_name=='resnet50':
        load_path_this = osp.join(load_path, 'r50_f')
    elif model_name=='swin':
        load_path_this = osp.join(load_path, 'swt_f')
    model = get_arch(model_name, num_classes)
    checkpoint_list = [osp.join(load_path_this + str(i), 'model_checkpoint.pth') for i in [0, 1, 2, 3, 4]]
    states = [torch.load(c,  map_location=device) for c in checkpoint_list]
    
    all_probs = []
    # Do inference for each model
    with torch.inference_mode():
        for i in range(len(states)):
            state=states[i]
            model.load_state_dict(state['model_state_dict'])
            probs, labels = test_cls(model, test_loader, device)
            all_probs.append(probs)
    
    ece = MulticlassCalibrationError(num_classes=num_classes, n_bins=n_bins, norm='l1')
    

    accs, eces, ces= [],[],[]
    for probs in all_probs:
        preds = np.argmax(probs, axis=1)
        test_auc, test_f1, test_acc, test_auc_all, test_f1_all = evaluate_cls(labels, preds, probs, 
                                                                              print_conf=False)
        e = ece(torch.from_numpy(probs), torch.from_numpy(labels)).item()        
        ce = log_loss(labels,probs)
        
        accs.append(100*test_acc)
        eces.append(100*e)
        ces.append(100*ce)
 
    ########################
    # print average results
    ######################## 
    print('ACC={:.2f}+/-{:.2f}, ECE={:.2f}+/-{:.2f}, NLL={:.2f}+/-{:.2f}'.format(
        np.mean(accs), np.std(accs), np.mean(eces), np.std(eces), np.mean(ces), np.std(ces)))

    if with_ens:
        ########################
        # results for ensemble
        ########################  
        ens_probs = np.mean(all_probs, axis=0)
        ens_preds = np.argmax(ens_probs, axis=1)
        test_auc, test_f1, test_acc, test_auc_all, test_f1_all = evaluate_cls(labels, ens_preds, ens_probs, 
                                                                              print_conf=False)
        e = ece(torch.from_numpy(ens_probs), torch.from_numpy(labels)).item()
        ce = log_loss(labels,ens_probs)

        print(30*'-')
        print('DEEP ENSEMBLES:')
        print('AUC={:.2f}, ACC={:.2f}, ECE={:.2f}, NLL={:.2f}'.format(100*test_acc,100*e,100*ce))
        
    mtrcs = [np.mean(accs), np.mean(eces), np.mean(ces), 
             np.std(accs),  np.std(eces),  np.std(ces)]

    if with_ens:
        return mtrcs, [100*test_acc, 100*e, 100*ce]
    return mtrcs


# In[ ]:


def print_results_multi_head(dataset=None, model_name='convnext', method='4lmh', n_bins=15, nh=2):
    assert dataset is not None
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # reproducibility
    seed=0
    set_seed(seed, use_cuda)
    if model_name in ['convnext', 'swin'] :
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = True

    assert dataset in ['chaoyang', 'kvasir', 'pathmnist']
    

    if dataset=='pathmnist':
        tg_size = 28 if model_name !='convnext' else 32
        test_loader = get_medmnist_test_loader(dataset, batch_size=128, num_workers=6, tg_size=tg_size)
        class_names = list(medmnist.INFO[dataset]['label'].keys())
        num_classes = len(class_names)

    else:
        data_path = osp.join('data', dataset)
        csv_test = osp.join('data', 'test_'+dataset+'.csv')
        tg_size = 224,224   
        test_loader  = get_class_test_loader(csv_test, data_path, tg_size, batch_size=64, num_workers=6)
        num_classes = len(test_loader.dataset.classes)
        class_names = ['C{}_probs'.format(i) for i in range(num_classes)]

    
    load_path = osp.join('experiments', dataset, method)
    if '2h' in method: nh = 2
    elif '4h' in method: nh = 4
    else: return 'need multi-head model here'
    
    if model_name=='convnext':
        load_path_this = osp.join(load_path, 'cnx_f')
    elif model_name=='resnet50':
        load_path_this = osp.join(load_path, 'r50_f')
    elif model_name=='swin':
        load_path_this = osp.join(load_path, 'swt_f')

    ########################
    # results for multi-head
    ########################
    model = get_arch(model_name, num_classes, n_heads=nh)
    checkpoint_list = [osp.join(load_path_this + str(i), 'model_checkpoint.pth') for i in [0, 1, 2, 3, 4]]
    states = [torch.load(c,  map_location=device) for c in checkpoint_list]
    all_probs = []
    # Do inference on val data
    with torch.inference_mode():
        for i in range(len(states)):
            state=states[i]        
            model.load_state_dict(state['model_state_dict'])
            probs, labels = test_cls_multihead(model, test_loader, device)
            all_probs.append(probs)
    
    ece = MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='l1')

    
    accs, eces, ces= [],[],[]
    for probs in all_probs:
        preds = np.argmax(probs, axis=1)
        test_auc, test_f1, test_acc, test_auc_all, test_f1_all = evaluate_cls(labels, preds, probs, 
                                                                              print_conf=False)
        e = ece(torch.from_numpy(probs), torch.from_numpy(labels)).item()
        ce = log_loss(labels,probs)                
        accs.append(100*test_acc)
        eces.append(100*e)
        ces.append(100*ce)
        
    ########################
    # print average results
    ######################## 
    print('ACC={:.2f}+/-{:.2f}, ECE={:.2f}+/-{:.2f}, NLL={:.2f}+/-{:.2f}'.format(
    np.mean(accs), np.std(accs), np.mean(eces), np.std(eces), np.mean(ces), np.std(ces)))
    
    mtrcs = [np.mean(accs), np.mean(eces), np.mean(ces), 
             np.std(accs),  np.std(eces),  np.std(ces)]
    return mtrcs


# In[ ]:


def print_all(dataset, model_name):
    metrcs = dict()
    print(60*'=')
    print('Single Head (avg) and Ensembles')
    print(60*'=')

    metrcs['sl1h'], metrcs['ens'] =     print_results_multi_fold(dataset, model_name, with_ens=True)

    # LS
    print('\n')
    print(60*'=')
    print('LS gamma=0.05: Single Head (avg) and Ensembles')
    print(60*'=')
    metrcs['ls_005'] =     print_results_multi_fold(dataset, model_name, method='ls_005', with_ens=False)

    # MbLS
    print('\n')
    print(60*'=')
    print('MbLS m=6: Single Head (avg) and Ensembles')
    print(60*'=')
    metrcs['mbls_6'] =     print_results_multi_fold(dataset, model_name, method='mbls_6', with_ens=False)
    
    # MIXUP
    print('\n')   
    print(60*'=')
    print('MixUp gamma=0.2: Single Head (avg) and Ensembles')
    print(60*'=')
    metrcs['mxp_02'] =     print_results_multi_fold(dataset, model_name, method='mxp_02', with_ens=False)

    # DCA
    print('\n')   
    print(60*'=')
    print('DCA: Single Head (avg) and Ensembles')
    print(60*'=')
    metrcs['dca'] =     print_results_multi_fold(dataset, model_name, method='dca', with_ens=False)

    # Two-Headed
    print('\n') 
    print(60*'=')
    print('Unperturbed ByCephal')
    print(60*'=')
    metrcs['2hsl'] =     print_results_multi_head(dataset, model_name, method='2hsl')

#     # Two-Headed
    print(60*'*')
    print('Perturbed ByCephal')
    print(60*'*')
    metrcs['2hml'] =     print_results_multi_head(dataset, model_name, method='2hml')

#     # 4-Headed    
    print(60*'*')
    print('Perturbed 4Cephal')
    print(60*'*')
    metrcs['4hml'] =     print_results_multi_head(dataset, model_name, method='4hml')
    
    return metrcs


# In[ ]:


def fill_rows(col_model_dict, col1, col2, col3, col4, s1, s2, s3, strings):
    sl1h_str, DE_str, LS_str, MbLS_str, Mxp_str, DCA_str, h2sl_str, h2ml_str, h4ml_str = strings
    a, b, c = 0, 1, 2
    x, y, z = 3, 4, 5
    sl1h_str=sl1h_str.replace(col1, '{:.2f}'.format(col_model_dict['sl1h'][a]))
    sl1h_str=sl1h_str.replace(col2, '{:.2f}'.format(col_model_dict['sl1h'][b]))
    sl1h_str=sl1h_str.replace(col3,  '{:.2f}'.format(col_model_dict['sl1h'][c]))
    sl1h_str=sl1h_str.replace(col4,  '{:.1f}'.format(col_model_dict['sl1h'][-1]))
    sl1h_str=sl1h_str.replace(s1, '{:.2f}'.format(col_model_dict['sl1h'][x]))
    sl1h_str=sl1h_str.replace(s2, '{:.2f}'.format(col_model_dict['sl1h'][y]))
    sl1h_str=sl1h_str.replace(s3,  '{:.2f}'.format(col_model_dict['sl1h'][z]))
    
    DE_str=DE_str.replace(col1, '{:.2f}'.format(col_model_dict['ens'][a]))
    DE_str=DE_str.replace(col2, '{:.2f}'.format(col_model_dict['ens'][b]))
    DE_str=DE_str.replace(col3,  '{:.2f}'.format(col_model_dict['ens'][c]))
    DE_str=DE_str.replace(col4,  '{:.1f}'.format(col_model_dict['ens'][-1]))

    LS_str=LS_str.replace(col1, '{:.2f}'.format(col_model_dict['ls_005'][a]))
    LS_str=LS_str.replace(col2, '{:.2f}'.format(col_model_dict['ls_005'][b]))
    LS_str=LS_str.replace(col3,  '{:.2f}'.format(col_model_dict['ls_005'][c]))
    LS_str=LS_str.replace(col4,  '{:.1f}'.format(col_model_dict['ls_005'][-1]))
    LS_str=LS_str.replace(s1, '{:.2f}'.format(col_model_dict['ls_005'][x]))
    LS_str=LS_str.replace(s2, '{:.2f}'.format(col_model_dict['ls_005'][y]))
    LS_str=LS_str.replace(s3,  '{:.2f}'.format(col_model_dict['ls_005'][z]))    

    MbLS_str=MbLS_str.replace(col1, '{:.2f}'.format(col_model_dict['mbls_6'][a]))
    MbLS_str=MbLS_str.replace(col2, '{:.2f}'.format(col_model_dict['mbls_6'][b]))
    MbLS_str=MbLS_str.replace(col3,  '{:.2f}'.format(col_model_dict['mbls_6'][c]))
    MbLS_str=MbLS_str.replace(col4,  '{:.1f}'.format(col_model_dict['mbls_6'][-1]))
    MbLS_str=MbLS_str.replace(s1, '{:.2f}'.format(col_model_dict['mbls_6'][x]))
    MbLS_str=MbLS_str.replace(s2, '{:.2f}'.format(col_model_dict['mbls_6'][y]))
    MbLS_str=MbLS_str.replace(s3,  '{:.2f}'.format(col_model_dict['mbls_6'][z]))   
    
    Mxp_str=Mxp_str.replace(col1, '{:.2f}'.format(col_model_dict['mxp_02'][a]))
    Mxp_str=Mxp_str.replace(col2, '{:.2f}'.format(col_model_dict['mxp_02'][b]))
    Mxp_str=Mxp_str.replace(col3,  '{:.2f}'.format(col_model_dict['mxp_02'][c]))
    Mxp_str=Mxp_str.replace(col4,  '{:.1f}'.format(col_model_dict['mxp_02'][-1]))
    Mxp_str=Mxp_str.replace(s1, '{:.2f}'.format(col_model_dict['mxp_02'][x]))
    Mxp_str=Mxp_str.replace(s2, '{:.2f}'.format(col_model_dict['mxp_02'][y]))
    Mxp_str=Mxp_str.replace(s3,  '{:.2f}'.format(col_model_dict['mxp_02'][z]))   
    
    DCA_str=DCA_str.replace(col1, '{:.2f}'.format(col_model_dict['dca'][a]))
    DCA_str=DCA_str.replace(col2, '{:.2f}'.format(col_model_dict['dca'][b]))
    DCA_str=DCA_str.replace(col3,  '{:.2f}'.format(col_model_dict['dca'][c]))
    DCA_str=DCA_str.replace(col4,  '{:.1f}'.format(col_model_dict['dca'][-1]))
    DCA_str=DCA_str.replace(s1, '{:.2f}'.format(col_model_dict['dca'][x]))
    DCA_str=DCA_str.replace(s2, '{:.2f}'.format(col_model_dict['dca'][y]))
    DCA_str=DCA_str.replace(s3,  '{:.2f}'.format(col_model_dict['dca'][z]))   
    
    h2sl_str=h2sl_str.replace(col1, '{:.2f}'.format(col_model_dict['h2sl'][a]))
    h2sl_str=h2sl_str.replace(col2, '{:.2f}'.format(col_model_dict['h2sl'][b]))
    h2sl_str=h2sl_str.replace(col3,  '{:.2f}'.format(col_model_dict['h2sl'][c]))
    h2sl_str=h2sl_str.replace(col4,  '{:.1f}'.format(col_model_dict['h2sl'][-1]))
    h2sl_str=h2sl_str.replace(s1, '{:.2f}'.format(col_model_dict['h2sl'][x]))
    h2sl_str=h2sl_str.replace(s2, '{:.2f}'.format(col_model_dict['h2sl'][y]))
    h2sl_str=h2sl_str.replace(s3,  '{:.2f}'.format(col_model_dict['h2sl'][z]))
    
    h2ml_str=h2ml_str.replace(col1, '{:.2f}'.format(col_model_dict['h2ml'][a]))
    h2ml_str=h2ml_str.replace(col2, '{:.2f}'.format(col_model_dict['h2ml'][b]))
    h2ml_str=h2ml_str.replace(col3,  '{:.2f}'.format(col_model_dict['h2ml'][c]))
    h2ml_str=h2ml_str.replace(col4,  '{:.1f}'.format(col_model_dict['h2ml'][-1]))
    h2ml_str=h2ml_str.replace(s1, '{:.2f}'.format(col_model_dict['h2ml'][x]))
    h2ml_str=h2ml_str.replace(s2, '{:.2f}'.format(col_model_dict['h2ml'][y]))
    h2ml_str=h2ml_str.replace(s3,  '{:.2f}'.format(col_model_dict['h2ml'][z]))
    
    h4ml_str=h4ml_str.replace(col1, '{:.2f}'.format(col_model_dict['h4ml'][a]))
    h4ml_str=h4ml_str.replace(col2, '{:.2f}'.format(col_model_dict['h4ml'][b]))
    h4ml_str=h4ml_str.replace(col3,  '{:.2f}'.format(col_model_dict['h4ml'][c]))
    h4ml_str=h4ml_str.replace(col4,  '{:.1f}'.format(col_model_dict['h4ml'][-1]))
    h4ml_str=h4ml_str.replace(s1, '{:.2f}'.format(col_model_dict['h4ml'][x]))
    h4ml_str=h4ml_str.replace(s2, '{:.2f}'.format(col_model_dict['h4ml'][y]))
    h4ml_str=h4ml_str.replace(s3,  '{:.2f}'.format(col_model_dict['h4ml'][z]))
    
    return sl1h_str, DE_str, LS_str, MbLS_str, Mxp_str, DCA_str, h2sl_str, h2ml_str, h4ml_str


# In[ ]:


def add_acc_ece_nll_combined_ranks(dic):
    dic_acc = {key:value[0] for (key,value) in dic.items()} #acc
    # False for lower is better, True for higher is better
    dic_rank_acc = {key: rank for rank, key in enumerate(sorted(dic_acc, key=dic_acc.get, reverse=True), 1)}
       
    dic_ece = {key:value[1] for (key,value) in dic.items()}
    dic_rank_ece = {key: rank for rank, key in enumerate(sorted(dic_ece, key=dic_ece.get, reverse=False), 1)}

    dic_nll = {key:value[2] for (key,value) in dic.items()}
    # False for lower is better, True for higher is better
    dic_rank_nll = {key: rank for rank, key in enumerate(sorted(dic_nll, key=dic_nll.get, reverse=False), 1)}

    
    for key in dic:
        dic[key].extend([dic_rank_acc[key], dic_rank_ece[key], dic_rank_nll[key],
                         np.mean([dic_rank_acc[key], dic_rank_ece[key], dic_rank_nll[key]])])    
    return dic


# In[ ]:


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# # CHAOYANG

# In[ ]:


dataset='chaoyang'


# In[ ]:


m_r50 = print_all(dataset, 'resnet50')


# In[ ]:


m_cvx = print_all(dataset, 'convnext')


# In[ ]:


m_swt = print_all(dataset, 'swin')


# In[ ]:


# m_r18_with_ranks = add_acc_ece_combined_ranks(m_r18)
m_r50_with_ranks = add_acc_ece_nll_combined_ranks(m_r50)
m_cvx_with_ranks = add_acc_ece_nll_combined_ranks(m_cvx)
m_swt_with_ranks = add_acc_ece_nll_combined_ranks(m_swt)


# In[ ]:


sl1h_str  ='\\textbf{SL1H}     & xx1$\pm$ss1 & xx2$\pm$ss2 & xx3$\pm$ss3 & xx4 & xx5$\pm$ss5 & xx6$\pm$ss6 & xx7$\pm$ss7 & xx8 & xx9$\pm$ss9 & x10$\pm$s10 & x11$\pm$s11 & x12 \\\\'
DE_str    ='\\textbf{D-Ens}    & xx1 & xx2 & xx3 & xx4 & xx5 & xx6 & xx7 & xx8 & xx9 & x10 & x11 & x12 \\\\'
LS_str    ='\\textbf{LS}       & xx1$\pm$ss1 & xx2$\pm$ss2 & xx3$\pm$ss3 & xx4 & xx5$\pm$ss5 & xx6$\pm$ss6 & xx7$\pm$ss7 & xx8 & xx9$\pm$ss9 & x10$\pm$s10 & x11$\pm$s11 & x12 \\\\'
MbLS_str  ='\\textbf{MbLS}     & xx1$\pm$ss1 & xx2$\pm$ss2 & xx3$\pm$ss3 & xx4 & xx5$\pm$ss5 & xx6$\pm$ss6 & xx7$\pm$ss7 & xx8 & xx9$\pm$ss9 & x10$\pm$s10 & x11$\pm$s11 & x12 \\\\'
Mxp_str   ='\\textbf{MixUp}    & xx1$\pm$ss1 & xx2$\pm$ss2 & xx3$\pm$ss3 & xx4 & xx5$\pm$ss5 & xx6$\pm$ss6 & xx7$\pm$ss7 & xx8 & xx9$\pm$ss9 & x10$\pm$s10 & x11$\pm$s11 & x12 \\\\'
DCA_str   ='\\textbf{DCA}      & xx1$\pm$ss1 & xx2$\pm$ss2 & xx3$\pm$ss3 & xx4 & xx5$\pm$ss5 & xx6$\pm$ss6 & xx7$\pm$ss7 & xx8 & xx9$\pm$ss9 & x10$\pm$s10 & x11$\pm$s11 & x12 \\\\'
h2sl_str  ='\\textbf{2HSL}     & xx1$\pm$ss1 & xx2$\pm$ss2 & xx3$\pm$ss3 & xx4 & xx5$\pm$ss5 & xx6$\pm$ss6 & xx7$\pm$ss7 & xx8 & xx9$\pm$ss9 & x10$\pm$s10 & x11$\pm$s11 & x12 \\\\'
h2ml_str   ='\\textbf{2HML}      & xx1$\pm$ss1 & xx2$\pm$ss2 & xx3$\pm$ss3 & xx4 & xx5$\pm$ss5 & xx6$\pm$ss6 & xx7$\pm$ss7 & xx8 & xx9$\pm$ss9 & x10$\pm$s10 & x11$\pm$s11 & x12 \\\\'
h4ml_str   ='\\textbf{4HML}      & xx1$\pm$ss1 & xx2$\pm$ss2 & xx3$\pm$ss3 & xx4 & xx5$\pm$ss5 & xx6$\pm$ss6 & xx7$\pm$ss7 & xx8 & xx9$\pm$ss9 & x10$\pm$s10 & x11$\pm$s11 & x12 \\\\'

# In[ ]:


strings = [sl1h_str, DE_str, LS_str, MbLS_str, Mxp_str, DCA_str, h2sl_str, h2ml_str, h4ml_str]

strings = fill_rows(m_r50, 'xx1', 'xx2', 'xx3', 'xx4', 'ss1', 'ss2', 'ss3', strings)
strings = fill_rows(m_cvx, 'xx5', 'xx6', 'xx7', 'xx8', 'ss5', 'ss6', 'ss7', strings)
strings = fill_rows(m_swt, 'xx9', 'x10', 'x11', 'x12', 'ss9', 's10', 's11', strings)

sl1h_str, DE_str, LS_str, MbLS_str, Mxp_str, DCA_str, h2sl_str, h2ml_str, h4ml_str = strings


# In[ ]:


caption = r'\caption{Results on the \textbf{Chaoyang dataset} , with standard deviation for 5 training runs.For each model, \unl{\textbf{best}} and \textbf{second best} ranks are marked.}\label{chaoyang_dispersion}'


# In[ ]:


print('\\begin{sidewaystable}')
print('\\renewcommand{\\arraystretch}{1.03}')
print('\\setlength\\tabcolsep{1.00pt}')
print('\\centering')
print(caption)
print('\\smallskip')
print('\\begin{tabular}{c cccc cccc cccc}')
print('& \\multicolumn{4}{c}{\\textbf{ResNet50}} & \\multicolumn{4}{c}{\\textbf{ConvNeXt}} & \\multicolumn{4}{c}{\\textbf{Swin-Transformer}} \\\\')
print('\\cmidrule(lr){2-5} \\cmidrule(lr){6-9} \\cmidrule(lr){10-13} &  ACC$^\\uparrow$  &  ECE$_\\downarrow$  &  NLL$_\\downarrow$    &  Rank$_\\downarrow$  &  ACC$^\\uparrow$  &  ECE$_\\downarrow$  &  NLL$_\\downarrow$    &  Rank$_\\downarrow$    &  ACC$^\\uparrow$ &  ECE$_\\downarrow$  &  NLL$_\\downarrow$    &  Rank$_\\downarrow$\\\\')
print('\\midrule')
print(sl1h_str)
print('\midrule')
print(LS_str)
print('\midrule')
print(MbLS_str)
print('\midrule')
print(Mxp_str)
print('\midrule')
print(DCA_str)
print('\midrule')
print('\midrule')
print(DE_str)
print('\midrule')
print('\midrule')
print(h2sl_str)
print('\midrule')
print(h2ml_str)
print('\midrule')
print(h4ml_str)
print('\\bottomrule')
print('\\\[-0.25cm]')
print('\\end{tabular}')


# # Kvasir

# In[ ]:


dataset='kvasir'


# In[ ]:


m_cvx = print_all(dataset, 'convnext')


# In[ ]:


m_r50 = print_all(dataset, 'resnet50')


# In[ ]:


m_swt = print_all(dataset, 'swin')


# In[ ]:


m_r50_with_ranks = add_acc_ece_nll_combined_ranks(m_r50)
m_cvx_with_ranks = add_acc_ece_nll_combined_ranks(m_cvx)
m_swt_with_ranks = add_acc_ece_nll_combined_ranks(m_swt)


# In[ ]:


sl1h_str  ='\\textbf{SL1H}     & xx1$\pm$ss1 & xx2$\pm$ss2 & xx3$\pm$ss3 & xx4 & xx5$\pm$ss5 & xx6$\pm$ss6 & xx7$\pm$ss7 & xx8 & xx9$\pm$ss9 & x10$\pm$s10 & x11$\pm$s11 & x12 \\\\'
DE_str    ='\\textbf{D-Ens}    & xx1 & xx2 & xx3 & xx4 & xx5 & xx6 & xx7 & xx8 & xx9 & x10 & x11 & x12 \\\\'
LS_str    ='\\textbf{LS}       & xx1$\pm$ss1 & xx2$\pm$ss2 & xx3$\pm$ss3 & xx4 & xx5$\pm$ss5 & xx6$\pm$ss6 & xx7$\pm$ss7 & xx8 & xx9$\pm$ss9 & x10$\pm$s10 & x11$\pm$s11 & x12 \\\\'
MbLS_str  ='\\textbf{MbLS}     & xx1$\pm$ss1 & xx2$\pm$ss2 & xx3$\pm$ss3 & xx4 & xx5$\pm$ss5 & xx6$\pm$ss6 & xx7$\pm$ss7 & xx8 & xx9$\pm$ss9 & x10$\pm$s10 & x11$\pm$s11 & x12 \\\\'
Mxp_str   ='\\textbf{MixUp}    & xx1$\pm$ss1 & xx2$\pm$ss2 & xx3$\pm$ss3 & xx4 & xx5$\pm$ss5 & xx6$\pm$ss6 & xx7$\pm$ss7 & xx8 & xx9$\pm$ss9 & x10$\pm$s10 & x11$\pm$s11 & x12 \\\\'
DCA_str   ='\\textbf{DCA}      & xx1$\pm$ss1 & xx2$\pm$ss2 & xx3$\pm$ss3 & xx4 & xx5$\pm$ss5 & xx6$\pm$ss6 & xx7$\pm$ss7 & xx8 & xx9$\pm$ss9 & x10$\pm$s10 & x11$\pm$s11 & x12 \\\\'
h2sl_str  ='\\textbf{2HSL}     & xx1$\pm$ss1 & xx2$\pm$ss2 & xx3$\pm$ss3 & xx4 & xx5$\pm$ss5 & xx6$\pm$ss6 & xx7$\pm$ss7 & xx8 & xx9$\pm$ss9 & x10$\pm$s10 & x11$\pm$s11 & x12 \\\\'
h2ml_str   ='\\textbf{2HML}      & xx1$\pm$ss1 & xx2$\pm$ss2 & xx3$\pm$ss3 & xx4 & xx5$\pm$ss5 & xx6$\pm$ss6 & xx7$\pm$ss7 & xx8 & xx9$\pm$ss9 & x10$\pm$s10 & x11$\pm$s11 & x12 \\\\'
h4ml_str   ='\\textbf{4HML}      & xx1$\pm$ss1 & xx2$\pm$ss2 & xx3$\pm$ss3 & xx4 & xx5$\pm$ss5 & xx6$\pm$ss6 & xx7$\pm$ss7 & xx8 & xx9$\pm$ss9 & x10$\pm$s10 & x11$\pm$s11 & x12 \\\\'


# In[ ]:


strings = [sl1h_str, DE_str, LS_str, MbLS_str, Mxp_str, DCA_str, h2sl_str, h2ml_str, h4ml_str]

strings = fill_rows(m_r50, 'xx1', 'xx2', 'xx3', 'xx4', 'ss1', 'ss2', 'ss3', strings)
strings = fill_rows(m_cvx, 'xx5', 'xx6', 'xx7', 'xx8', 'ss5', 'ss6', 'ss7', strings)
strings = fill_rows(m_swt, 'xx9', 'x10', 'x11', 'x12', 'ss9', 's10', 's11', strings)

sl1h_str, DE_str, LS_str, MbLS_str, Mxp_str, DCA_str, h2sl_str, h2ml_str, h4ml_str = strings


# In[ ]:


caption = r'\caption{Results on the \textbf{Kvasir dataset} , with standard deviation for 5 training runs.For each model, \unl{\textbf{best}} and \textbf{second best} ranks are marked.}\label{kvasir_dispersion}'


# In[ ]:


print('\\bigskip\\bigskip  % provide some separation between the two tables')
print(caption)
print('\\smallskip')
print('\\smallskip')
print('\\begin{tabular}{c cccc cccc cccc}')
print('& \\multicolumn{4}{c}{\\textbf{ResNet50}} & \\multicolumn{4}{c}{\\textbf{ConvNeXt}} & \\multicolumn{4}{c}{\\textbf{Swin-Transformer}} \\\\')
print('\\cmidrule(lr){2-5} \\cmidrule(lr){6-9} \\cmidrule(lr){10-13} &  ACC$^\\uparrow$  &  ECE$_\\downarrow$  &  NLL$_\\downarrow$    &  Rank$_\\downarrow$  &  ACC$^\\uparrow$  &  ECE$_\\downarrow$  &  NLL$_\\downarrow$    &  Rank$_\\downarrow$    &  ACC$^\\uparrow$ &  ECE$_\\downarrow$  &  NLL$_\\downarrow$    &  Rank$_\\downarrow$\\\\')
print('\\midrule')
print(oneh_str)
print('\midrule')
print(LS_str)
print('\midrule')
print(MbLS_str)
print('\midrule')
print(Mxp_str)
print('\midrule')
print(DCA_str)
print('\midrule')
print('\midrule')
print(DE_str)
print('\midrule')
print('\midrule')
print(mh_str)
print('\midrule')
print(p2h_str)
print('\midrule')
print(p4h_str)
print('\\bottomrule')
print('\\\[-0.25cm]')
print('\\end{tabular}')
print('\\end{sidewaystable}')

