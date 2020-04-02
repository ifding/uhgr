import matplotlib
#import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorboardX
from tensorboardX import SummaryWriter

import argparse
import os
import pickle
import random
import shutil
import time
import scipy.sparse as sp

import encoders
import gen.feat as featgen
import gen.data as datagen

import process
import util

from modules import Classifier, LogReg


def evaluate(dataset, model, classifier, args, name='Validation', max_num_examples=None):
    classifier.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float(), requires_grad=False).cuda()
        shuf_fts = Variable(data['shuf_fts'].float(), requires_grad=False).cuda()
        h0 = Variable(data['feats'].float()).cuda()
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

        _, embeds  = model.embed(h0, adj, shuf_fts, batch_num_nodes, assign_x=assign_input)  
        ypred = classifier(embeds)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx+1)*args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    
    result = {'prec': metrics.precision_score(labels, preds, average='macro'),
              'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds),
              'F1': metrics.f1_score(labels, preds, average="micro")}
    print(name, " accuracy:", result['acc'])
    return result


def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
        
    name += '_' + args.task + '-task'   
    name += '_' + args.encoder
    name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
    name += '_ar' + str(int(args.assign_ratio*100))
    if args.linkpred:
        name += '_lp'

    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    return name


def log_assignment(assign_tensor, writer, epoch, batch_idx):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8,6), dpi=300)

    # has to be smaller than args.batch_size
    for i in range(len(batch_idx)):
        plt.subplot(2, 2, i+1)
        plt.imshow(assign_tensor.cpu().data.numpy()[batch_idx[i]], cmap=plt.get_cmap('BuPu'))
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")
    plt.tight_layout()
    fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('assignment', data, epoch)

def log_graph(adj, batch_num_nodes, writer, epoch, batch_idx, assign_tensor=None):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(8,6), dpi=300)

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i+1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='#336699',
                edge_color='grey', width=0.5, node_size=300,
                alpha=0.7)
        ax.xaxis.set_visible(False)

    plt.tight_layout()
    fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs', data, epoch)

    # colored according to assignment
    assignment = assign_tensor.cpu().data.numpy()
    fig = plt.figure(figsize=(8,6), dpi=300)

    num_clusters = assignment.shape[2]
    all_colors = np.array(range(num_clusters))

    for i in range(len(batch_idx)):
        ax = plt.subplot(2, 2, i+1)
        num_nodes = batch_num_nodes[batch_idx[i]]
        adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()

        label = np.argmax(assignment[batch_idx[i]], axis=1).astype(int)
        label = label[: batch_num_nodes[batch_idx[i]]]
        node_colors = all_colors[label]

        G = nx.from_numpy_matrix(adj_matrix)
        nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color=node_colors,
                edge_color='grey', width=0.4, node_size=50, cmap=plt.get_cmap('Set1'),
                vmin=0, vmax=num_clusters-1,
                alpha=0.8)

    plt.tight_layout()
    fig.canvas.draw()

    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    data = tensorboardX.utils.figure_to_image(fig)
    writer.add_image('graphs_colored', data, epoch)



def train_node_task(adj, features, model, args, labels, dataset_idx, writer=None, mask_nodes = True):
    
    writer_batch_idx = [0, 3, 6, 9]
    
    # training params
    batch_size = 1
    nb_epochs = 10000
    patience = 40
    lr = 0.001
    l2_coef = 0.0
    drop_prob = 0.0
    hid_units = 512
    sparse = True
    nonlinearity = 'prelu' # special name to separate parameters 
    best = 1e9
    
    
    nb_classes = labels.shape[1]
    
    idx_train, idx_val, idx_test = dataset_idx[0], dataset_idx[1], dataset_idx[2]
    labels = torch.FloatTensor(labels[np.newaxis]).cuda()
    idx_train = torch.LongTensor(idx_train).cuda()
    idx_val = torch.LongTensor(idx_val).cuda()
    idx_test = torch.LongTensor(idx_test).cuda()      
    
    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001)
    
    nb_nodes = adj.shape[0]  
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))           
    adj = (adj + sp.eye(adj.shape[0])).todense()
    adj = torch.FloatTensor(adj[np.newaxis])
    features = torch.FloatTensor(features[np.newaxis])
    idx = np.random.permutation(nb_nodes)
    shuf_fts = features[:, idx, :]

    if torch.cuda.is_available():
        print('Using CUDA')
        adj = adj.cuda()
        features = features.cuda()
        shuf_fts = shuf_fts.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    batch_num_nodes = np.array([nb_nodes])    
    

    cnt_wait = 0
    best_t = 0    
    for epoch in range(args.num_epochs):
        
        print('Epoch: ', epoch)        
        avg_loss = 0.0
        begin_time = time.time()
        
        model.train()     
        optimizer.zero_grad()
        loss = model(features, adj, shuf_fts, batch_num_nodes)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        avg_loss = loss

        elapsed = time.time() - begin_time
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)
        print('Loss: ', avg_loss, '; epoch time: ', elapsed)
        
        if avg_loss < best:
            best = avg_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break
           
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_dgi.pkl')) 


    log = LogReg(args.hidden_dim*3, nb_classes)
    xent = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    log.cuda() 
    

    best_val_result = {
            'epoch': 0,
            'acc': 0}
    test_result = {
            'epoch': 0,
            'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []   
    
    
    for epoch in range(args.num_epochs):
        
        print('Epoch: ', epoch)
        
        begin_time = time.time()

        node_embeds, _  = model.embed(features, adj, shuf_fts, batch_num_nodes)

        train_embs = node_embeds[0, idx_train]
        val_embs = node_embeds[0, idx_val]
        test_embs = node_embeds[0, idx_test]

        train_lbls = torch.argmax(labels[0, idx_train], dim=1)
        val_lbls = torch.argmax(labels[0, idx_val], dim=1)
        test_lbls = torch.argmax(labels[0, idx_test], dim=1)

        for _ in range(100):
            log.train()
            opt.zero_grad()
            logits = log(train_embs)

            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step() 

        logits = log(val_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
        elapsed = time.time() - begin_time
        print('Val acc: ', acc.item(), '; epoch time: ', elapsed)

        if acc > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = acc.item()
            best_val_result['epoch'] = epoch
            
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            
            test_result['acc'] = acc.item()
            test_result['epoch'] = epoch
    
    return test_result      


def train_graph_task(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
        mask_nodes = True):    
    writer_batch_idx = [0, 3, 6, 9]
    
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001)
    iter = 0
    
    patience = 40
    cnt_wait = 0
    best = 1e9
    best_t = 0
    
    
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            model.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            shuf_fts = Variable(data['shuf_fts'].float(), requires_grad=False).cuda()
            label = Variable(data['label'].long()).cuda()
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()
            
            loss = model(h0, adj, shuf_fts, batch_num_nodes, assign_x=assign_input)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss

            # log once per XX epochs
            if epoch % 10 == 0 and batch_idx == len(dataset) // 2 and writer is not None:
                log_assignment(model.assign_tensor, writer, epoch, writer_batch_idx)
                log_graph(adj, batch_num_nodes, writer, epoch, writer_batch_idx, model.assign_tensor)
        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)
        print('Avg loss: ', avg_loss, '; epoch time: ', elapsed)
        

        if avg_loss < best:
            best = avg_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break
                
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_dgi.pkl')) 
    
       
    classifier = Classifier(args.hidden_dim*6, [], args.num_classes)
    xent = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
    classifier.cuda() 
    
    best_val_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    test_result = {
            'epoch': 0,
            'loss': 0,
            'acc': 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []         
    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        classifier.train()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            classifier.zero_grad()
            adj = Variable(data['adj'].float(), requires_grad=False).cuda()
            shuf_fts = Variable(data['shuf_fts'].float(), requires_grad=False).cuda()
            h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
            label = Variable(data['label'].long()).cuda()
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()
            
            _, embeds  = model.embed(h0, adj, shuf_fts, batch_num_nodes, assign_x=assign_input)    
            logits = classifier(embeds)

            loss = xent(logits, label)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            opt.step() 
            
            avg_loss += loss            
            
        avg_loss /= batch_idx + 1 
        elapsed = time.time() - begin_time
        print('Avg loss: ', avg_loss, '; epoch time: ', elapsed)
        result = evaluate(dataset, model, classifier, args, name='Train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)            
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, classifier, args, name='Validation')
            val_accs.append(val_result['acc'])
        if val_result['acc'] > best_val_result['acc'] - 1e-7:
            best_val_result['acc'] = val_result['acc']
            best_val_result['epoch'] = epoch
            best_val_result['loss'] = avg_loss
        if writer is not None:
            writer.add_scalar('acc/train_acc', result['acc'], epoch)
            writer.add_scalar('acc/val_acc', val_result['acc'], epoch)
            writer.add_scalar('loss/best_val_loss', best_val_result['loss'], epoch)

        print('Best val result: ', best_val_result)
        best_val_epochs.append(best_val_result['epoch'])
        best_val_accs.append(best_val_result['acc'])
                

    matplotlib.style.use('seaborn')
    plt.switch_backend('agg')
    plt.figure()
    plt.plot(train_epochs, util.exp_moving_avg(train_accs, 0.85), '-', lw=1)
    
    plt.plot(best_val_epochs, best_val_accs, 'bo')
    plt.legend(['train', 'val'])
    
    train_plt_name = 'results/' + gen_prefix(args) + '.png'
    plt.savefig(train_plt_name, dpi=600)
    plt.close()
    matplotlib.style.use('default')

    return val_accs

def node_benchmark_task(args, writer=None):
    all_vals = []

    adj, features, labels, dataset_idx = process.read_graph(args.bmname)
    max_num_nodes = features.shape[0]   
    input_dim = features.shape[1]

    model = encoders.SoftPoolingGcnEncoder(
            args.encoder,               
            max_num_nodes, 
            input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
            args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
            bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
            assign_input_dim=input_dim).cuda()

    for i in range(2):
        val_accs = train_node_task(adj, features, model, args, labels, dataset_idx, writer=writer)
        all_vals.append(val_accs['acc'])
    print(all_vals)
    print(np.mean(all_vals))
    print(np.std(all_vals))

def benchmark_task_val(args, writer=None, feat='node-label'):
    all_vals = []
    graphs = process.read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)
    #graphs = load_data.read_moleculefile(args.datadir, args.bmname, max_nodes=args.max_nodes)

    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in graphs[0].node[0]:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                G.node[u]['feat'] = np.array(G.node[u]['label'])
    else:
        print('Using constant labels')
        featgen_const = featgen.ConstFeatureGen(np.ones(args.input_dim, dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)
         
    for i in range(10):
        train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = \
                process.prepare_graph_task(graphs, args, i, max_nodes=args.max_nodes)

        model = encoders.SoftPoolingGcnEncoder(
                args.encoder,               
                max_num_nodes, 
                input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
                args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
                bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
                assign_input_dim=assign_input_dim).cuda()
   
        val_accs = train_graph_task(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
            writer=writer)
        all_vals.append(np.array(val_accs))
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    print(all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))
    
    
def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--task', dest='task',
            help='Task. Possible values: node, graph')    
    benchmark_parser.add_argument('--bmname', dest='bmname',
            help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname',
            help='Name of the pkl data file')

    model_parser = parser.add_argument_group()
    model_parser.add_argument('--encoder', dest='encoder',
            help='Encoder. Possible values: GCN, GAT')     
    model_parser.add_argument('--assign-ratio', dest='assign_ratio', type=float,
            help='ratio of number of nodes in consecutive layers')
    model_parser.add_argument('--num-pool', dest='num_pool', type=int,
            help='number of pooling layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
            const=True, default=False,
            help='Whether link prediction side objective is used')

    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
            help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
            help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')

    parser.add_argument('--name-suffix', dest='name_suffix',
            help='suffix added to the output filename')

    parser.set_defaults(datadir='data',
                        logdir='log',
                        task='node',
                        dataset='DD',
                        max_nodes=1000,
                        cuda='1',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=20,
                        num_epochs=1000,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=128,
                        output_dim=128,
                        num_classes=2,
                        num_gc_layers=3,
                        dropout=0.0,
                        name_suffix='',
                        encoder='GAT', #GCN
                        assign_ratio=0.1,
                        num_pool=1
                       )
    return parser.parse_args()

def main():
    prog_args = arg_parse()

    # export scalar data to JSON for external processing
    path = os.path.join(prog_args.logdir, gen_prefix(prog_args))
    if os.path.isdir(path):
        print('Remove existing log dir: ', path)
        shutil.rmtree(path)
    writer = SummaryWriter(path)
    #writer = None

    os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
    print('CUDA', prog_args.cuda)
    
    if prog_args.task == 'graph':
        benchmark_task_val(prog_args, writer=writer)
        writer.close()
    elif prog_args.task == 'node':
        node_benchmark_task(prog_args, writer=writer)
        writer.close()
    else:
        print('Unrecognized task argument.')
        writer.close()
        exit()

if __name__ == "__main__":
    main()

