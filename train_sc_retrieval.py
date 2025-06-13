import scanpy as sc, anndata as ad
import os
import argparse
import numpy as np
import json
from PIL import Image
import glob
from utils.scl_dataset import HGDatasetRandom, SCDataset
from model.MVTLM import GeneEncoderClass
from model.sclocate import SCL
from utils.data_util import set_th_config,pad_data, read_folder_sc
import torch
import pandas as pd
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from scipy.stats import pearsonr
import pickle
from scipy.spatial import distance
Image.MAX_IMAGE_PIXELS = 1253830860
parser = argparse.ArgumentParser()
# data parameters
parser.add_argument("--gpu_idx", type=int, default=1, help="gpu")
parser.add_argument("--epochs", type=int, default=2, help="batch size")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--model_name", type=str, default="pretrain_parameter.pkl", help="model name")
#parser.add_argument("--task", type=str, default="68kpbmc", help="saved path")
#parser.add_argument("--out_name", type=str, default='test_ood_out_2.json', help="model name")
parser.add_argument("--lr", type=float, default=4e-3, help="learning rate")
# setting
parser.add_argument("--r", type=int, default=224 // 2, help="spot radius")
parser.add_argument("--save_path", type=str, default="./cont_breast/", help="saved path")
parser.add_argument("--llm_path", type=str, default="./pretrain_30/", help="saved path")
parser.add_argument("--ckpt_name", type=str, default="e1_kd_model.pkl", help="saved path")
parser.add_argument("--t_sample", type=str, default="CID4290", help="saved path")
parser.add_argument("--seed", type=int, default=123456, help="random seed")
parser.add_argument("--trunc", type=int, default=5, help="random seed")
# model setting
parser.add_argument("--c_dim", type=int, default=128, help="layer norm eps")
parser.add_argument("--layer_norm_eps", type=float, default=1e-12, help="layer norm eps")
parser.add_argument("--c_drop_rate", type=float, default=0.0, help="layer norm eps")
configs = parser.parse_args()
set_th_config(configs.seed)
cuda_str = 'cuda' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx)
device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')
configs.device = str(device)
print(configs)

if os.path.exists(configs.save_path):
    pass
else:
    os.mkdir(configs.save_path) 

with open(os.path.join(configs.llm_path, "configs.json"),'r',encoding='utf-8') as f:
    llm_configs = json.load(f)
    llm_configs['device'] = configs.device
    llm_configs['batch_size'] = configs.batch_size

parser.set_defaults(**llm_configs)
llm_configs = parser.parse_args()
with open(os.path.join(configs.llm_path, "tokenizer.json"),'r',encoding='utf-8') as f:
    tok_dict = json.load(f) 
dataset_dict,test_dataset_dict = {}, {}
us_dict,us_train = {},{}
select_gene = list(pd.read_csv('gene_list.csv')['gene'])
with open(r'./1000hvg_common.pkl', 'rb') as f:
    us_gene = pickle.load(f).to_list()
with open(r'./3000hvg_union_sc.pkl', 'rb') as f:
    sc_gene = pickle.load(f).to_list()

new_us = []
for ele in us_gene:
    if ele in select_gene:
        continue
    new_us.append(ele)
us_gene = new_us
def read_folder(folder_path):
    adata = sc.read_mtx(os.path.join(folder_path, 'matrix.mtx'))
    adata_bc=pd.read_csv(os.path.join(folder_path, 'barcodes.tsv'),header=None)
    adata_features=pd.read_csv(os.path.join(folder_path, 'features.tsv'),header=None)
    adata= adata.T
    adata.obs_names= adata_bc[0].tolist()
    adata.var['gene_name']= adata_features[0].tolist()
    adata.var.index= adata.var['gene_name']
    return adata
print(len(select_gene), len(us_gene))
sample_name = ['CID4290', 'CID4465', 'CID4535', 'CID44971', '1142243F', '1160920F']
#sample_name = ['CID4290', 'CID4465', 'CID4535', 'CID44971']
for sample in sample_name:
    st_adata = read_folder("./ST/filtered_count_matrices/%s_filtered_count_matrix" % (sample))
    pos = pd.read_csv("./ST/%s_spatial/tissue_positions_list.csv" % (sample), sep=",", header = None) 
    sc.pp.normalize_total(st_adata, target_sum=1e4)
    sc.pp.log1p(st_adata, base=2)
    mer_adata = st_adata[:, select_gene]
    out_adata = st_adata[:, us_gene]
    print(sample, st_adata.X.shape)
    
    img = Image.open(glob.glob(os.path.join('./ST/%s_spatial' % (sample), 'image.tif'))[0])
    if sample not in [configs.t_sample]:
       dataset_dict[sample] = {'st':mer_adata, 'img': img, 'sample_id': sample, 'pos':pos}
       us_train[sample] = {'st':out_adata, 'img': img, 'sample_id': sample, 'pos':pos}
    else:
        test_dataset_dict[sample] = {'st':mer_adata, 'img': img, 'sample_id': sample, 'pos':pos}
        us_dict[sample] = {'st':out_adata, 'img': img, 'sample_id': sample, 'pos':pos}
sc_sample_name = configs.t_sample
if sc_sample_name == 'CID4290':
    sc_sample_name = 'CID4290A'
test_sc_adata = read_folder_sc('SC/%s' % (sc_sample_name))
sc.pp.normalize_total(test_sc_adata, target_sum=1e4)
sc.pp.log1p(test_sc_adata, base=2)
test_sc_adata = test_sc_adata[:, sc_gene]
with open(os.path.join(configs.save_path, "configs.json"),'w',encoding='utf-8') as f:
    json.dump(vars(configs),f) 

st_deconv = pd.read_csv(os.path.join('cell_type_deconv', '%s_ct_deconv.csv' % (configs.t_sample)),header=0, index_col=0)

#select_samples = pd.read_csv('select_spat_margin20.txt', delimiter='\t')
#select_samples = list(select_samples.itertuples(index=False, name=None))
#select_gene = pd.read_csv(os.path.join("gene_list.csv"))
def tokenize(gene_name):
    if gene_name in tok_dict:
        return tok_dict[gene_name]
    else:
        return tok_dict['C10orf54']
merfish_gene_idx = [tokenize(x) for x in select_gene]
us_gene_idx = [tokenize(x) for x in us_gene]
sc_gene_idx = [tokenize(x) for x in sc_gene]
uni_model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True).to(device)
transform = create_transform(**resolve_data_config(uni_model.pretrained_cfg, model=uni_model))
def train_collate_fn_multi(data):
    sample, bc, patch_img, pos, exp,in_idx = zip(*data)
    patch_img = [transform(x) for x in patch_img]
    in_idx = pad_data(in_idx, 0)
    exp = pad_data(exp, 0.0)
    st_data = torch.tensor(np.stack(exp,axis=0), dtype=torch.float32)
    #pos = torch.tensor(np.stack(pos,axis=0), dtype=torch.float32)
    img_data = torch.stack(patch_img,dim=0)
    idx_data = torch.tensor(np.stack(in_idx,axis=0), dtype=torch.int64)
    return sample, bc, img_data, st_data, idx_data

def get_train_loader(configs,  dataset_dict, task='sw'):
    dataset = HGDatasetRandom(dataset_dict, configs.r, merfish_gene_idx, configs.trunc, 'train')
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=configs.batch_size, shuffle=True,
                                               collate_fn=train_collate_fn_multi)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=configs.batch_size, shuffle=True,
                                               collate_fn=train_collate_fn_multi)
    return train_loader, val_loader
def get_test_loader(configs,  dataset_dict, gene_idx, task='sw'):
    train_set = HGDatasetRandom(dataset_dict, configs.r, gene_idx, configs.trunc, 'test')
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=configs.batch_size, shuffle=False,
                                               collate_fn=train_collate_fn_multi)
    return train_loader    
def get_test_loader_sc(configs,  sc_adata, gene_idx):
    train_set = SCDataset(sc_adata, gene_idx)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=configs.batch_size, shuffle=False)
    return train_loader    
train_loader,val_loader = get_train_loader(configs,  dataset_dict)
us_loader = get_test_loader(configs,  us_dict, us_gene_idx)
sc_loader = get_test_loader_sc(configs, test_sc_adata, sc_gene_idx)
model = SCL(configs)
model.to(device)
g_model = GeneEncoderClass(llm_configs, len(tok_dict))
state_dict = torch.load(os.path.join(configs.llm_path, configs.ckpt_name), map_location=configs.device)
#new_state_dict = OrderedDict()
#for k, v in state_dict.items():
#    new_state_dict[k.replace("module.", "")] = v
g_model.load_state_dict(state_dict)
g_model.to(device)
optimizer = torch.optim.Adam(
            model.parameters(), lr=configs.lr
        )

def compute_metrics(lst):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(lst == 0)) / len(lst)
    metrics["R5"] = 100 * float(np.sum(lst < 5)) / len(lst)
    metrics["R10"] = 100 * float(np.sum(lst < 10)) / len(lst)
    metrics["R50"] = 100 * float(np.sum(lst < 50)) / len(lst)
    metrics["R100"] = 100 * float(np.sum(lst < 100)) / len(lst)
    metrics["MedR"] = np.median(lst) + 1
    metrics["MeanR"] = np.mean(lst) + 1
    #stats = [metrics[x] for x in ("R1", "R5", "R10")]
    #metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    return metrics

def get_sc_embed(data_load, model):
    out_data = []
    #loss_sum = 0
    for iter, data in enumerate(data_load):
        model.eval()
        g_model.eval()
        with torch.no_grad():
            bc, sc_data, in_idx, cell_t = data
            gene_feat,_ = g_model.encode_gene(sc_data.to(device), in_idx.to(device))
            gene_feat = model.gene_projection(gene_feat[:,0,:])
            for co,g,ct in zip(bc, gene_feat.detach().cpu().numpy(), cell_t):
                out_data.append({'barcode':co, 'gene':g, 'cell_type':ct})
    return out_data
def get_st_feature(data_load, model):
    out_data = []
    #loss_sum = 0
    for iter, data in enumerate(data_load):
        model.eval()
        g_model.eval()
        uni_model.eval()
        with torch.no_grad():
            _,bc, img_data, _, in_idx = data
            img_fea = uni_model(img_data.to(device))
            img_feat = model.image_projection(img_fea)
            #gene_feat = gene_feat[:,0,:]
            for co,img in zip(bc, img_feat.detach().cpu().numpy()):
                out_data.append({'barcode':co, 'img':img})
    return out_data

def score_data(sc_out, st_out, deconv):
    for i,row in enumerate(sc_out):
        out = []
        for j,curr_st in enumerate(st_out):
            curr_dis = distance.cosine(row['gene'], curr_st['img'])
            out.append((curr_dis, row['cell_type'], curr_st['barcode']))
        out = sorted(out, key=lambda x: x[0])
        print(deconv.loc[out[0][2]])
        print(out[0][1])
        exit()

def test_unseen(data_load, model, us=True):
    out_data = []
    #loss_sum = 0
    for iter, data in enumerate(data_load):
        model.eval()
        g_model.eval()
        uni_model.eval()
        with torch.no_grad():
            _,bc, img_data, st_data, in_idx = data
            img_fea = uni_model(img_data.to(device))
            mer_exp_out = st_data
            gene_feat,_ = g_model.encode_gene(mer_exp_out.to(device), in_idx.to(device))
            img_feat = model.image_projection(img_fea)
            gene_feat = model.gene_projection(gene_feat[:,0,:])
            #gene_feat = gene_feat[:,0,:]
            for co,g,img in zip(bc, gene_feat.detach().cpu().numpy(), img_feat.detach().cpu().numpy()):
                out_data.append({'barcode':co, 'gene':g, 'image':img})
    all_image = [x['image'] for x in out_data]
    label_idx = {}
    for i,row in enumerate(out_data):
        out = []
        for j in range(len(all_image)):
            curr_dis = distance.cosine(row['gene'], all_image[j])
            if i == j:
                out.append((True, curr_dis))
            else:
                out.append((False, curr_dis))
        out = sorted(out, key=lambda x: x[1])
        for idx,val in enumerate(out):
            if val[0]:
                label_idx[row['barcode']] = idx
                break
    data_idx = np.array([v for k,v in label_idx.items()])
    return compute_metrics(data_idx)
#
# us_out = test_unseen(us_loader, model)


max_val = 0.0
max_s = 0.0
best_model = model
for i in range(configs.epochs):
    loss_sum = 0
    for iter, data in enumerate(train_loader):
        model.train()
        g_model.eval()
        uni_model.eval()
        _,bc, img_data, st_data, in_idx = data
        img_fea = uni_model(img_data.to(device)).detach()
        mer_exp_out = st_data
        gene_fea,_ = g_model.encode_gene(mer_exp_out.to(device), in_idx.to(device))
        cos_smi = model(img_fea, gene_fea[:,0,:].detach())
        label = torch.eye(cos_smi.shape[0], cos_smi.shape[1]).to(device)
        gene_loss = torch.nn.functional.cross_entropy(cos_smi, label)
        images_loss = torch.nn.functional.cross_entropy(cos_smi.T, label.T)
        output_loss = (gene_loss + images_loss) / 2
        optimizer.zero_grad()
        output_loss.backward()
        optimizer.step()
        loss_sum += float(output_loss)
        if iter % 500 == 0:
            print('Epoch %i, Curr iter %i loss %f' % (i+1, iter, float(output_loss)))
    print('Epoch %i, Sum loss %f' % (i+1, loss_sum))
    print('Val')
    seen_out = test_unseen(val_loader, model, False)
    print(seen_out)    
    if seen_out['R100'] > max_s:
        max_s = seen_out['R100']
        print('Save')
        torch.save(model.state_dict(), os.path.join(configs.save_path, 'contr_model.pkl' % (i+1)))
        best_model = model

sc_feature = get_sc_embed(sc_loader, best_model)
img_st_feature = get_st_feature(us_loader, best_model)
#score_data(sc_feature, img_st_feature, st_deconv)
with open('res_%s_e%i.pkl' % (configs.t_sample, configs.epochs), 'wb') as handle:
    pickle.dump({'sc':sc_feature, 'st':img_st_feature}, handle, protocol=pickle.HIGHEST_PROTOCOL)

