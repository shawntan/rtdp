import torch
import torch.nn as nn
# import torch.nn.functional as F
DEBUG = False


class RNNOp(nn.Module):
    def __init__(self, nhid, dropout=0.):
        super(RNNOp, self).__init__()
        self.op = nn.Sequential(
            nn.Linear(2 * nhid, nhid),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

    def forward(self, left, right):
        return self.op(torch.cat([left, right], dim=-1))


class LSTMOp(nn.Module):
    def __init__(self, nhid, dropout=0.):
        super(LSTMOp, self).__init__()
        self.transform = nn.Linear(2 * nhid, 5 * nhid)

    def forward(self, left, right):
        if isinstance(left, tuple):
            h_left, c_left = left
        else:
            h_left, c_left = left, torch.zeros_like(left)

        if isinstance(right, tuple):
            h_right, c_right = right
        else:
            h_right, c_right = right, torch.zeros_like(right)

        h = torch.cat([h_left, h_right], dim=-1)
        i_, f1_, f2_, o_, u_ = self.transform(h).chunk(5, dim=-1)
        i = torch.sigmoid(i_)
        f1 = torch.sigmoid(f1_)
        f2 = torch.sigmoid(f2_)
        o = torch.sigmoid(o_)
        u = torch.tanh(u_)
        c = i * u + f1 * c_left + f2 * c_right
        h = o * torch.tanh(c)

        return h, c


class TreeRNN(nn.Module):
    def __init__(self, ntoken, nhid, padding_idx, parens_id=(0, 1), dropout=0.0, op=LSTMOp):
        super(TreeRNN, self).__init__()
        self.op = op(nhid)
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(ntoken, nhid)
        self.embedding_aux = nn.Embedding(ntoken, nhid)

        self.paren_open, self.paren_close = parens_id

    def forward(self, input):
        lens = (input != self.padding_idx).sum(0)
        # batch_idxs = torch.arange(lens.size(0), device=lens.device)
        # mode = input[lens - 2, batch_idxs]
        batched_root = []
        batched_internal_states = []
        batched_leaves = []
        batched_leaves_aux = []
        batched_lengths = []
        for i in range(lens.size(0)):
            root, internal, leaves, leaves_aux = \
                self.parse(input[1:lens[i] - 1, i])
            batched_root.append(root)
            batched_internal_states.append(internal)
            batched_leaves.append(leaves)
            batched_leaves_aux.append(leaves_aux)
            batched_lengths.append(leaves.size(0))

        batched_root = torch.stack(batched_root)
        batched_lengths = torch.tensor(batched_lengths, device=input.device)
        # batched_root = self.op(batched_root, self.embedding(mode))[0]
        batched_internal_states = nn.utils.rnn.pad_sequence(batched_internal_states)
        batched_leaves = nn.utils.rnn.pad_sequence(batched_leaves)
        batched_leaves_aux = nn.utils.rnn.pad_sequence(batched_leaves_aux)

        leaves_mask = (torch.arange(batched_leaves.size(0),
                                    device=batched_root.device)[:, None] <
                       batched_lengths[None, :])

        internal_mask = (torch.arange(batched_internal_states.size(0),
                                      device=batched_root.device)[:, None] <
                         (2 * batched_lengths - 1)[None, :])
        return (batched_root,
                batched_internal_states, internal_mask,
                batched_leaves, batched_leaves_aux, leaves_mask)

    def parse(self, sent):
        leaves_emb = []
        leaves_emb_aux = []
        internal_states = []
        stack = []
        if DEBUG:
            disp_stack = []
            print(sent)
        for idx in sent:
            if idx == self.paren_close:
                right = stack.pop()
                left = stack.pop()
                combined = self.op(left, right)
                stack.append(combined)
                internal_states.append(combined[0])

                if DEBUG:
                    right = disp_stack.pop()
                    left = disp_stack.pop()
                    disp_stack.append((left, right))
            else:
                if idx != self.paren_open:
                    emb = self.embedding.weight[idx]
                    stack.append(emb)
                    internal_states.append(emb)
                    leaves_emb.append(emb)
                    leaves_emb_aux.append(self.embedding_aux.weight[idx])
                    if DEBUG:
                        disp_stack.append(idx.item())
        if DEBUG:
            print(disp_stack[0])

        result = stack[0]
        internal_states = torch.stack(internal_states)
        leaves = torch.stack(leaves_emb)
        leaves_aux = torch.stack(leaves_emb_aux)
        if isinstance(result, tuple):
            return result[0], internal_states, leaves, leaves_aux
        else:
            return result, internal_states, leaves, leaves_aux


if __name__ == "__main__":
    tree = TreeRNN(5, 50)
    print(tree(torch.Tensor([[0, 0, 2, 3, 1, 0, 2, 2, 1, 1],
                             [2, 4, 4, 4, 4, 4, 4, 4, 4, 4]]).long().t()))
    print(tree(torch.Tensor([[2]]).long().t()))
