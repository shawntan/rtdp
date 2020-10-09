import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from seq_dataloader import data_loader
from scan_model import SCANModel
PADDING_TOKEN = '<pad>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'
UNK_TOKEN = '<unk>'


def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, optimizer], f)


def model_load(fn):
    global model, optimizer
    with open(fn, 'rb') as f:
        device = torch.device('cpu' if not args.cuda else 'cuda')
        model, optimizer = torch.load(f, map_location=device)


###############################################################################
# Training code
###############################################################################
def idxs2string(idxs, vocab):
    string = ""
    for idx in idxs.cpu().numpy():
        if idx == -1:
            continue
        word = "UNK" if idx > len(vocab) else vocab[idx]
        if word == "<pad>":
            continue
        elif word == "<end>":
            string += " " + word
            break
        else:
            string += " " + word
    return string


class ReprWrapper(object):
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        if isinstance(self.val, list):
            return "[" + repr(self.val[0]) + " " + repr(self.val[1]) + "]"
        else:
            if self.val is not None:
                return self.val
            else:
                return "NONE!"


def list2tree_inorder(depth):
    if depth == 0:
        return [ReprWrapper(None)]
    else:
        left_inorder = list2tree_inorder(depth - 1)
        right_inorder = list2tree_inorder(depth - 1)
        midpoint = len(left_inorder) // 2
        new_node = [left_inorder[midpoint], right_inorder[midpoint]]
        return left_inorder + [ReprWrapper(new_node)] + right_inorder


def idxpos2tree(idxs, pos, vocab):
    inorder = list2tree_inorder(args.nslot)
    idxs = idxs[1:len(pos) + 1]

    for i, idx in enumerate(idxs):
        inorder[pos[i]].val = vocab[idx]
    root = inorder[len(inorder) // 2]
    return repr(root)


def example_str(inp, trg, preds, positions):
    input_string = idxs2string(inp, src_id2w)
    targs_string = idxs2string(trg, trg_id2w)
    preds_string = idxpos2tree(preds, positions, trg_id2w)
    string = ("Input     : " + input_string + "\n"
              "Target    : " + targs_string + "\n"
              "Predicted : " + preds_string)
    return string


def evaluate(data_iter,
             print_examples=None,
             every=2000,
             score='accuracy',
             print_examples_pos=None):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    with torch.no_grad():
        sens_same = 0
        sens_count = 0
        for batch, data in enumerate(data_iter):
            inp, inp_len, trg, trg_len = data
            inp = inp.to('cuda' if args.cuda else 'cpu')
            trg = trg.to('cuda' if args.cuda else 'cpu')

            if score == 'accuracy':
                # Accuracy measure
                preds, positions = model(inp, trg)
                trg = trg.permute(1, 0)
                preds = preds[:trg.size(0)]
                mask = trg != trg_w2id[PADDING_TOKEN]
                sens_full_match = (((preds == trg) & mask) == mask).all(dim=0)
                sens_same += sens_full_match.sum()

            elif score == 'first_word':
                preds, positions = model(inp, trg)
                trg = trg.permute(1, 0)
                mask = trg != trg_w2id[PADDING_TOKEN]
                sens_full_match = preds[1, :] == trg[1, :]
                sens_same += sens_full_match.sum()

            elif score == 'll':
                nll = model(inp, trg, eval_loss=True)
                positions = None
                mask = trg != trg_w2id[PADDING_TOKEN]
                sens_same += -nll

            sens_count += mask.shape[1]

            if print_examples is not None:
                if (batch % every) == 0:
                    for i in range(mask.shape[1]):
                        string = example_str(inp[i], trg[:, i], preds[:, i],
                                             positions[i])
                        if not sens_full_match[i]:
                            print(string, file=print_examples)
                            print(file=print_examples)
                        elif print_examples_pos is not None:
                            print(string, file=print_examples_pos)
                            print(file=print_examples_pos)

        sens_acc = float(sens_same) / sens_count
    return sens_acc


def train(eval_every=-1, eval_fun=None):
    # Turn on training mode which enables dropout.
    model.train()

    total_loss = 0
    start_time = time.time()
    batch = 0
    for update, data in enumerate(training_data_iter):
        # print(data)
        # batch_data = get_batch(next(training_data_iter))
        inp, inp_len, trg, trg_len = data
        inp = inp.to('cuda' if args.cuda else 'cpu')
        trg = trg.to('cuda' if args.cuda else 'cpu')
        batch_size = trg.size(0)
        chunk_size = batch_size // args.chunks_per_batch
        src_lengths = (inp != src_w2id[PADDING_TOKEN]).sum(1)
        trg_lengths = (trg != trg_w2id[PADDING_TOKEN]).sum(1)
        for i in range(args.chunks_per_batch):
            src_chunk_lengths = src_lengths[i * chunk_size: (i+1) * chunk_size]
            trg_chunk_lengths = trg_lengths[i * chunk_size: (i+1) * chunk_size]
            if trg_chunk_lengths.size(0) == 0:
                break
            src_max_length = src_chunk_lengths.max()
            trg_max_length = trg_chunk_lengths.max()
            # print(idxs2string(inp[i * chunk_size, :src_max_length], src_id2w))
            # print(idxs2string(trg[i * chunk_size, :trg_max_length], trg_id2w))
            loss = model(
                inp[i * chunk_size: (i+1) * chunk_size, :src_max_length],
                trg[i * chunk_size: (i+1) * chunk_size, :trg_max_length]
            )
            batch += 1
            loss.backward()
            # print(loss)
            total_loss += loss.detach().data

            if batch % args.log_interval == 0 and batch > 0:
                elapsed = time.time() - start_time
                print(
                    '| epoch {:3d} '
                    '| {:5d} / {:5d} batches '
                    '| lr {:05.5f} | ms/batch {:5.2f} '
                    '| loss {:5.5f}'.format(
                        epoch,
                        batch,
                        len(training_data_iter) * args.chunks_per_batch,
                        optimizer.param_groups[0]['lr'],
                        elapsed * 1000 / args.log_interval,
                        total_loss.item() / batch))
                # total_loss = 0
                start_time = time.time()

        if args.clip:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        optimizer.zero_grad()
        if eval_every > 0 and (update + 1) % eval_every == 0:
            eval_fun()
            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--src_path_train', type=str,
                        default='data/simple/train_wo_valid_random.src')
    parser.add_argument('--trg_path_train', type=str,
                        default='data/simple/train_wo_valid_random.trg')
    parser.add_argument('--src_path_valid', type=str,
                        default='data/simple/valid.random.src')
    parser.add_argument('--trg_path_valid', type=str,
                        default='data/simple/valid.random.trg')
    parser.add_argument('--src_path_test', type=str,
                        default='data/simple/test.src')
    parser.add_argument('--trg_path_test', type=str,
                        default='data/simple/test.trg')

    parser.add_argument('--model_file', type=str, default='model.pt')

    parser.add_argument('--prod-class', type=str, default='Cell',
                        help='model class for generative function')
    parser.add_argument('--bidirection', action='store_true',
                        help='use bidirection model')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='max sequence length')
    parser.add_argument('--seq_len_test', type=int, default=1000,
                        help='max sequence length')
    parser.add_argument('--emsize', type=int, default=128,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=128,
                        help='number of hidden units per layer')
    parser.add_argument('--nslot', type=int, default=8,
                        help='number of memory slots')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=1,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                        help='batch size')
    parser.add_argument('--chunks-per-batch', type=int, default=1)
    parser.add_argument('--batch_size_test', type=int, default=64, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropoutm', type=float, default=0.1,
                        help='dropout applied to memory (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.1,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropouto', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')

    parser.add_argument('--dec-leaf-dropout', type=float, default=0.1,
                        help='Decoder leaf transform dropout')
    parser.add_argument('--dec-out-dropout', type=float, default=0.1,
                        help='Decoder output transform dropout')
    parser.add_argument('--dec-int-dropout', type=float, default=0.1,
                        help='Decoder attention integration dropout')
    parser.add_argument('--dec-attn-dropout', type=float, default=0.6,
                        help='Decoder attention dropout')

    parser.add_argument('--dec-no-node-attn', action='store_false')
    parser.add_argument('--dec-no-leaf-attn', action='store_false')

    parser.add_argument('--encoder-type', type=str, default='OM')
    parser.add_argument('--paren-open', type=str, default='[')
    parser.add_argument('--paren-close', type=str, default=']')
    parser.add_argument('--valid-score', type=str, default='accuracy')
    parser.add_argument('--test-score', type=str, default='accuracy')

    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='report interval')
    parser.add_argument('--test-only', action='store_true',
                        help='Test only')
    parser.add_argument('--logdir', type=str, default='./models/',
                        help='path to save outputs')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--name', type=str, default=randomhash,
                        help='exp name')
    parser.add_argument('--wdecay', type=float, default=0.,
                        help='weight decay applied to all weights')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.logdir, args.name)):
        os.makedirs(os.path.join(args.logdir, args.name))

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    # Compute vocabuary
    src_vocab = set(w.lower() for l in open(args.src_path_train)
                    for w in l.strip().split())
    trg_vocab = set(w.lower() for l in open(args.trg_path_train)
                    for w in l.strip().split())
    src_vocab.update(['<pad>', '<start>', '<end>', '<unk>'])
    src_id2w = list(src_vocab)
    src_id2w.sort()
    src_w2id = {w: i for i, w in enumerate(src_id2w)}
    trg_vocab.update(['<pad>', '<start>', '<end>', '<unk>'])
    trg_id2w = list(trg_vocab)
    trg_id2w.sort()
    trg_w2id = {w: i for i, w in enumerate(trg_id2w)}

    training_data_iter = data_loader.get_loader(
        args.src_path_train, args.trg_path_train,
        src_w2id, trg_w2id,
        batch_size=args.batch_size
    )

    valid_data_iter = data_loader.get_loader(
        args.src_path_valid, args.trg_path_valid,
        src_w2id, trg_w2id,
        batch_size=args.batch_size_test
    )

    test_dataloader = data_loader.get_loader(
        args.src_path_test, args.trg_path_test,
        src_w2id, trg_w2id,
        batch_size=args.batch_size_test
    )

    args.__dict__.update({
        'trg_ntoken': len(trg_id2w),
        'src_ntoken': len(src_id2w),
        'ninp': args.emsize,
        'start_idx': src_w2id[START_TOKEN],
        'end_idx': src_w2id[END_TOKEN],
        'unk_idx': src_w2id[UNK_TOKEN],
        'trg_padding_idx': trg_w2id[PADDING_TOKEN],
        'src_padding_idx': src_w2id[PADDING_TOKEN],
        'paren_open': src_w2id.get(args.paren_open, -1),
        'paren_close': src_w2id.get(args.paren_close, -1),
    })
    model = SCANModel(args)
    print(model)
    criterion = nn.NLLLoss(ignore_index=trg_w2id[PADDING_TOKEN])
    softmax = nn.LogSoftmax(dim=2)

    if args.cuda:
        model = model.cuda()

    params = list(model.parameters())
    total_params = sum(np.prod(x.size()) for x in model.parameters())
    # assert total_params == total_params_sanity
    print("TOTAL PARAMS: %d" % sum(np.prod(x.size())
                                   for x in model.parameters()))
    print('Args:', args)
    print('Model total parameters:', total_params)

    if not args.test_only:

        print("start training")
        # Loop over epochs.
        lr = args.lr
        stored_acc = float("-inf")
        # At any point you can hit Ctrl + C to break out of training early.

        def eval_test():
            global stored_acc
            print("Evaluating")
            valid_sens_acc = evaluate(valid_data_iter,
                                      score=args.valid_score)
            test_sens_acc = evaluate(test_dataloader,
                                     print_examples=sys.stdout,
                                     score=args.test_score,
                                     every=2000)
            valid_acc = valid_sens_acc

            print('-' * 89)
            print(
                '| epoch {:3d} '
                '| time: {:5.2f}s '
                '| ↑ valid score: {:.6f} '
                '| ↑ test acc: {:.4f} '
                ''.format(
                    epoch,
                    (time.time() - epoch_start_time),
                    valid_sens_acc,
                    test_sens_acc,
                )
            )
            if valid_sens_acc >= stored_acc:
                model_save(args.model_file)
                print('Saving model (new best validation)')
                stored_acc = valid_sens_acc
            print('-' * 89)
            return valid_acc

        try:
            optimizer = None
            # Ensure the optimizer is optimizing params,
            # which includes both the model's weights
            # as well as the criterion's weight (i.e. Adaptive Softmax)
            optimizer = torch.optim.Adam(params,
                                         lr=args.lr,
                                         betas=(0, 0.999),
                                         eps=1e-9,
                                         weight_decay=args.wdecay)

            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'max', 0.5,
                patience=1, threshold=0,
            )

            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                train()
                valid_acc = eval_test()
                scheduler.step(valid_acc)
                if optimizer.param_groups[0]['lr'] < 1e-5:
                    break

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    model_load(args.model_file)
    model.decoder.set_depth(args.nslot)
    if args.cuda:
        model = model.cuda()
    if args.test_only:
        print(model)

    sens_acc = evaluate(test_dataloader,
                        score=args.test_score,
                        print_examples=open(args.model_file + '.neg', 'w'),
                        print_examples_pos=open(args.model_file + '.pos', 'w'),
                        every=1)
    data = {'args': args.__dict__,
            'parameters': total_params,
            'test_acc': sens_acc}
    print('-' * 89)
    print('| sent acc: {:.4f} ''|\n'.format(sens_acc))
