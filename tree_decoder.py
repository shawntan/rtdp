import torch
import torch.nn as nn
from torch.nn import functional as F
import ctreec
import numpy as np


def update_logbreak(prev_log_prob, log_prob):
    return 0.5 * prev_log_prob[:, :, None, 1:] + log_prob


def breadth2inorder(depth):
    factor = np.array([-0.5, 0.5])
    start = np.array([0.])
    ranks = [start]
    for d in range(depth):
        level_ranks = ranks[-1][:, None] + factor
        factor = factor / 2
        ranks.append(level_ranks.reshape(-1))
    ranks = np.concatenate(ranks)
    return np.argsort(ranks)


class Attention(nn.Module):
    def __init__(self, hidden_size=0, dropout=0.0):
        super(Attention, self).__init__()
        self.query_t = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size)
        )
        self.key_t = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size)
        )

        self.register_buffer("factor",
                             torch.sqrt(torch.tensor(hidden_size,
                                                     dtype=torch.float)))

    def forward(self, query, key, value,
                query_mask=None, key_mask=None,
                log_eps=torch.tensor(-64, dtype=torch.long),
                eps=torch.tensor(1e-6, dtype=torch.long)):
        query_count, batch_size, _ = query.size()
        key_count, batch_size, _ = value.size()
        query = self.query_t(query)
        key = self.key_t(key)
        q_ = query.permute(1, 0, 2)
        k_ = key.permute(1, 2, 0)
        v_ = value.permute(1, 0, 2)
        scores = torch.matmul(q_, k_) / self.factor
        # batch_size, query_count, key_count

        if key_mask is not None:
            km_ = key_mask.permute(1, 0)[:, None, :]
            if query_mask is not None:
                qm_ = query_mask.permute(1, 0)[:, :, None]
                mask = ~(qm_ & km_)
            else:
                mask = ~km_
        # Masked softmax
        k = scores\
            .masked_fill(mask, scores.min())\
            .max(dim=-1, keepdim=True)[0]
        scores = scores - k
        # logsumexp = ctreec.masked_logsumexp(scores, mask, log_eps, eps)
        # attn = ctreec.exp_safe(scores - logsumexp, log_eps, eps, mask)
        exp_scores = torch.exp(scores - k).masked_fill(mask, 0.)
        attn = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + 1e-3)
        context = torch.matmul(attn, v_).permute(1, 0, 2)
        return context


class CopyCell(nn.Module):
    def __init__(self, hidden_size, activation,
                 dropout=0.0, branch_factor=2):
        super(CopyCell, self).__init__()
        self.hidden_size = hidden_size
        self.cell_hidden_size = hidden_size

        self.output_t = nn.Sequential(
            nn.Linear(2 * hidden_size, self.cell_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.cell_hidden_size,
                      ((branch_factor + 1) * hidden_size))
        )
        # self.output_t = nn.Sequential(
        #     nn.Linear(hidden_size, branch_factor * 2 * hidden_size),
        # )

        self.activation = activation
        self.branch_factor = branch_factor

    def forward(self, x, context):
        length, batch_size, hidden_size = x.size()
        output = self.output_t(torch.cat([x, context], dim=-1))
        copy_gate_, cells = output\
            .split((hidden_size, self.branch_factor * hidden_size), dim=-1)
        copy_gate = torch.sigmoid(copy_gate_)\
            .view(length, batch_size, 1, hidden_size)
        cells = cells.view(length, batch_size,
                           self.branch_factor,
                           hidden_size)
        return (copy_gate * x[:, :, None, :] +
                (1 - copy_gate) * self.activation(cells))


class Cell(nn.Module):
    def __init__(self, hidden_size, activation,
                 dropout=0.0, branch_factor=2):
        super(Cell, self).__init__()
        self.hidden_size = hidden_size
        self.cell_hidden_size = 2 * hidden_size

        in_linear = nn.Linear(2 * hidden_size, self.cell_hidden_size)
        torch.nn.init.xavier_uniform_(in_linear.weight)
        torch.nn.init.zeros_(in_linear.bias)

        out_linear = nn.Linear(self.cell_hidden_size,
                               2 * branch_factor * hidden_size)
        torch.nn.init.xavier_uniform_(out_linear.weight)
        torch.nn.init.zeros_(out_linear.bias)

        self.output_t = nn.Sequential(
            in_linear,
            nn.ReLU(),
            nn.Dropout(dropout),
            out_linear
        )

        self.activation = activation
        self.branch_factor = branch_factor

    def forward(self, x, context):
        length, batch_size, hidden_size = x.size()
        output_size = (length, batch_size,
                       self.branch_factor,
                       hidden_size)
        output = self.output_t(torch.cat([x, context], dim=-1))
        gates_, cells = output.split((
            self.branch_factor * hidden_size,
            self.branch_factor * hidden_size
        ), dim=-1)
        gates = torch.sigmoid(
            gates_.view(length, batch_size,
                        self.branch_factor, hidden_size))
        cells = cells.view(output_size)

        branches = (gates * self.activation(cells) +
                    (1 - gates) * x[:, :, None, :])
        return branches


class Operator(nn.Module):
    def __init__(self, cell, output_classes, activation,
                 leaf_dropout, output_dropout, integrate_dropout,
                 attn_dropout,
                 node_attention, output_attention):
        super(Operator, self).__init__()
        self.cell = cell
        self.output_classes = output_classes
        self.hidden_size = self.cell.hidden_size
        self.activation = activation

        self.in_transform = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            self.activation
        )

        self.leaf_transform = nn.Sequential(
            nn.Dropout(leaf_dropout),
            nn.Linear(2 * self.hidden_size, 2, bias=True),
        )
        self.register_buffer('pos_neg', torch.tensor([1., -1.],
                                                     dtype=torch.float))

        # torch.nn.init.xavier_uniform_(self.leaf_transform[1].weight)
        # torch.nn.init.zeros_(self.leaf_transform[1].bias)
        torch.nn.init.zeros_(self.leaf_transform[-1].weight)
        # torch.nn.init.zeros_(self.leaf_transform[-1].bias)

        self._output_dropout = nn.Dropout(output_dropout)
        self._output_transform = nn.Linear(self.hidden_size,
                                           output_classes, bias=False)
        torch.nn.init.zeros_(self._output_transform.weight)

        if node_attention or output_attention:
            self._attn = Attention(self.hidden_size, dropout=attn_dropout)

        self.int_dropout = nn.Dropout(integrate_dropout)
        if node_attention:
            self.attn = self._attn
            # self.attn = Attention(self.hidden_size)
            # self._integrate_attn = nn.Linear(2 * self.hidden_size,
            #                                 2 * self.hidden_size)
        else:
            self.attn = None

        if output_attention:
            #self.attn_lex = Attention(self.hidden_size,
            #                          dropout=attn_dropout)
            self.attn_lex = self._attn
        else:
            self.attn_lex = None

        self.out_norm = nn.Sequential(
            nn.LayerNorm(self.hidden_size,
                         elementwise_affine=False),
            nn.Tanh()
        )
        # self.out_norm = nn.Tanh()
        self.log_eps = -64.
        # self.inv_temp = nn.Parameter(torch.tensor(1.))

    # def out_norm(self, x):
    #     return self.inv_temp * (
    #         x / (1e-6 + torch.norm(x, p=2, dim=-1, keepdim=True)))

    def output_transform(self, hidden, global_cond):
        if self.attn_lex is not None:
            emb = self.attn_lex(
                 query=hidden,
                 key=global_cond[2],
                 value=self.out_norm(global_cond[3]),
                 key_mask=global_cond[4]
            )
            emb = self._output_dropout(emb)
            # emb = self.out_norm(hidden)
            out_emb = self.out_norm(self._output_transform.weight)
        else:
            emb = self._output_dropout(hidden)
            out_emb = self._output_transform.weight
        logits = F.linear(emb, out_emb)
        output = torch.log_softmax(logits, dim=-1)
        return output

    def log_leaves(self, hidden, context):
        logits = self.leaf_transform(
            torch.cat((hidden, context), dim=-1))
        # logits = logits.expand(-1, -1, -1, 2) * self.pos_neg
        log_leaf = torch.log_softmax(logits, dim=-1)
        return log_leaf

    def init_parent(self, parent):
        prev_context = parent[None, :, :]
        prev_hidden = self.in_transform(parent)[None, :, :]
        # prev_hidden = torch.zeros_like(prev_context)
        prev_log_leaf = self.log_leaves(
            prev_hidden[:, :, None, :],
            prev_context[:, :, None, :]
        )[:, :, 0, :]
        return prev_hidden, prev_context, prev_log_leaf

    def forward(self, prev_hidden, prev_context, prev_log_leaf,
                global_cond):
        branches_struct = self.cell(self.int_dropout(prev_hidden),
                                    prev_context)
        # time_steps, batch_size, branch_factor, hidden_size = branches.size()
        branches_ = branches_struct.permute(0, 2, 1, 3)
        branches_size = branches_.size()
        flat_branches = branches_.flatten(0, 1)

        if self.attn is not None:
            context = self.attn(flat_branches,
                                global_cond[0], global_cond[0],
                                key_mask=global_cond[1])
        else:
            hiddens = flat_branches
            hiddens = self.int_dropout(hiddens)
            context = torch.zeros_like(flat_branches)

        context_struct = \
            context.view(branches_size).permute(0, 2, 1, 3)
        log_leaf_ = self.log_leaves(branches_struct, context_struct)
        log_leaf_struct = update_logbreak(prev_log_leaf, log_leaf_)
        log_leaves = log_leaf_struct.permute(0, 2, 1, 3).flatten(0, 1)

        return (flat_branches, context, log_leaves)


class CTreeDecoder(nn.Module):
    def __init__(self, ntoken, slot_size, producer_class,
                 leaf_dropout, output_dropout, integrate_dropout,
                 attn_dropout,
                 node_attention, output_attention,
                 padding_idx=10, max_depth=8):
        super(CTreeDecoder, self).__init__()

        self.branch_factor = 2
        # self.activation = nn.Tanh()
        # self.activation = nn.LayerNorm(slot_size)
        self.activation = nn.Sequential(
            nn.LayerNorm(slot_size),
            nn.Tanh(),
        )
        cell = eval(producer_class)
        self.expand = Operator(
            cell(slot_size, self.activation),
            ntoken, self.activation,
            leaf_dropout,
            output_dropout,
            integrate_dropout,
            attn_dropout,
            node_attention, output_attention
        )
        self.slot_size = slot_size
        self.input_size = slot_size
        self.emp_idx = ntoken
        self.padding_idx = padding_idx
        self.set_depth(max_depth)

    def set_depth(self, depth):
        self.max_depth = depth
        self.register_buffer('order',
                             torch.tensor(breadth2inorder(depth),
                                          dtype=torch.long))

        self.loss = ctreec.Loss(depth)

    def forward(self, encoded, context, X=None, max_depth=None):
        if max_depth is None:
            max_depth = self.max_depth
        prev_hidden, prev_context, prev_log_leaf = \
            self.expand.init_parent(encoded)
        prev_node_rank = torch.zeros_like(prev_log_leaf[:, :, :1])
        hidden_levels = [prev_hidden]
        context_levels = [prev_context]
        leaves_levels = [prev_log_leaf]
        node_rank_levels = [prev_node_rank]
        depth_levels = [prev_node_rank]

        for depth in range(max_depth):

            prev_hidden, prev_context, prev_log_leaf = \
                self.expand(prev_hidden, prev_context, prev_log_leaf,
                            global_cond=context)

            depths = torch.full_like(prev_node_rank, depth + 1)

            hidden_levels.append(prev_hidden)
            context_levels.append(prev_context)
            leaves_levels.append(prev_log_leaf)
            node_rank_levels.append(prev_node_rank)
            depth_levels.append(depths)

        flattened_log_leaves = \
            torch.cat(leaves_levels, dim=0)[self.order]
        flattened_hiddens = torch.cat(hidden_levels, dim=0)[self.order]
        flattened_context = torch.cat(context_levels, dim=0)[self.order]

        flattened_log_words = \
            (self.expand.output_transform(flattened_hiddens, context) +
             flattened_log_leaves[:, :, :1])
        return flattened_log_words

    def compute_loss(self, encoded, context, X):
        X = X.clone()
        # Remove start
        X = X[1:]
        # Remove end
        target_lengths = torch.sum(X != self.padding_idx, dim=0) - 1
        X[target_lengths, torch.arange(X.size(1), dtype=torch.long)] = \
            self.padding_idx
        X = X[:-1]

        log_tokens = self.forward(encoded, context, X)
        losses = self.loss(log_tokens, X, target_lengths)
        word_losses = losses / target_lengths.float()
        return word_losses.mean()


    def decode(self, encoded, context):
        return self.loss.decode(self.forward(encoded, context))

    def max_prob(self, encoded, context, X):
        start_idx = X[:1]
        target_lengths = torch.sum(X != self.padding_idx, dim=0) - 1
        end_idx = X[target_lengths,
                    torch.arange(X.size(1), dtype=torch.long)]
        log_tokens = self.forward(encoded, context)
        decoded_list, positions = self.loss.decode(log_tokens)
        max_length = max(max(len(d) for d in decoded_list) + 2, X.size(0))
        result = torch.full((max_length, len(decoded_list)),
                            self.padding_idx,
                            dtype=torch.long,
                            device=X.device)
        result[0, :] = start_idx
        for i, seq in enumerate(decoded_list):
            seq_length = seq.size(0)
            result[1:seq_length + 1, i] = seq
            result[seq_length + 1, i] = end_idx[0]
        return result, positions
