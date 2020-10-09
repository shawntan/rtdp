import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Distribution(nn.Module):
    def __init__(self, nslot, hidden_size, dropout):
        super(Distribution, self).__init__()

        self.query = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.key = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.beta = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.hidden_size = hidden_size

    def init_p(self, bsz, nslot):
        return None

    @staticmethod
    def process_softmax(beta, prev_p):
        if prev_p is None:
            return (torch.zeros_like(beta),
                    torch.ones_like(beta),
                    torch.zeros_like(beta))

        beta_normalized = beta - beta.max(dim=-1)[0][:, None]
        x = torch.exp(beta_normalized)

        prev_cp = torch.cumsum(prev_p, dim=1)
        mask = prev_cp[:, 1:]
        mask = mask.masked_fill(mask < 1e-5, 0.)
        mask = F.pad(mask, (0, 1), value=1)

        x_masked = x * mask

        p = F.normalize(x_masked, p=1)
        cp = torch.cumsum(p, dim=1)
        rcp = torch.cumsum(p.flip([1]), dim=1).flip([1])
        return cp, rcp, p

    def forward(self, in_val, prev_out_M, prev_p):
        query = self.query(in_val)
        key = self.key(prev_out_M)
        beta = self.beta(query[:, None, :] + key).squeeze(dim=2)
        beta = beta / math.sqrt(self.hidden_size)
        cp, rcp, p = self.process_softmax(beta, prev_p)
        return cp, rcp, p

class Cell(nn.Module):
    def __init__(self, hidden_size, dropout, activation=None):
        super(Cell, self).__init__()
        self.hidden_size = hidden_size
        self.cell_hidden_size = 4 * hidden_size
        self.input_t = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, self.cell_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.cell_hidden_size, hidden_size * 4),
        )
        # self.gates = nn.Sequential(
        #     nn.Sigmoid(),
        # )
        assert activation is not None
        self.activation = activation
        self.drop = nn.Dropout(dropout)
    def forward(self, vi, hi):
        bsz, hsz = vi.size()
        input = torch.cat([vi, hi], dim=-1)
        g_input, cell = self.input_t(input).split(
            (self.hidden_size * 3, self.hidden_size),
            dim=-1
        )
        # gates = self.gates(g_input)
        # vg, hg, cg = gates.chunk(3, dim=1)
        # output = self.activation(vg * vi + hg * hi + cg * cell)
        g_input = g_input.view(bsz, hsz, 3)
        gates = torch.softmax(g_input, dim=-1)
        vg, hg, cg = gates.unbind(dim=-1)
        output = vg * vi + hg * hi + cg * self.activation(cell)
        return output


class OrderedMemoryRecurrent(nn.Module):
    def __init__(self, input_size, slot_size, nslot,
                 dropout=0.2, dropoutm=0.2):
        super(OrderedMemoryRecurrent, self).__init__()

        self.activation = nn.Sequential(
            nn.LayerNorm(slot_size),
            nn.Tanh()
        )
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, slot_size),
            self.activation
        )

        self.distribution = Distribution(nslot, slot_size, dropoutm)

        self.cell = Cell(slot_size, dropout, activation=self.activation)

        self.nslot = nslot
        self.slot_size = slot_size
        self.input_size = input_size

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        zeros = weight.new(bsz, self.nslot, self.slot_size).zero_()
        p = self.distribution.init_p(bsz, self.nslot)
        return (zeros, zeros, p)

    def omr_step(self, in_val, prev_M, prev_c_M, prev_p):
        batch_size, nslot, slot_size = prev_M.size()
        _batch_size, slot_size = in_val.size()

        assert self.slot_size == slot_size
        assert self.nslot == nslot
        assert batch_size == _batch_size

        # Use current input to perform attention on candidate stack
        cp, rcp, p = self.distribution(in_val, prev_c_M, prev_p)

        # copy over top of the stack from candidate and the rest from previous
        M = prev_M * (1 - rcp)[:, :, None] + prev_c_M * rcp[:, :, None]

        M_list = []

        # Init with first value
        h = in_val
        for i in range(nslot):
            if i == nslot - 1 or (cp[:, i + 1] > 0).any():
                h = self.cell(h, M[:, i, :])
                h = in_val * (1 - cp)[:, i, None] + h * cp[:, i, None]
            M_list.append(h)
        c_M = torch.stack(M_list, dim=1)
        return M, c_M, p

    def forward(self, X, hidden, mask=None):
        prev_M, prev_memory_output, prev_p = hidden
        M_list = []
        p_list = []
        X_projected = self.input_projection(X)
        if mask is not None:
            padded = ~mask
        for t in range(X_projected.size(0)):
            prev_M, prev_memory_output, prev_p = \
                self.omr_step(X_projected[t],
                              prev_M, prev_memory_output, prev_p)
            if mask is not None:
                padded_1 = padded[t, :, None]
                padded_2 = padded[t, :, None, None]
                prev_p = prev_p.masked_fill(padded_1, 0.)
                prev_M = prev_M.masked_fill(padded_2, 0.)
                prev_memory_output = \
                    prev_memory_output.masked_fill(padded_2, 0.)
            M_list.append(prev_memory_output)
            p_list.append(prev_p)

        memories = torch.stack(M_list)
        probs = torch.stack(p_list)
        return X_projected, memories, probs


class OrderedMemory(nn.Module):
    def __init__(self, input_size, slot_size, nslot,
                 ntokens, padding_idx, dropout=0.2, dropoutm=0.1):

        super(OrderedMemory, self).__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(ntokens, 2 * input_size,
                                      padding_idx=self.padding_idx)
        self.rnn_encode = RNNEncoder(input_size)
        self.OM_forward = OrderedMemoryRecurrent(input_size, slot_size, nslot,
                                                 dropout=dropout, dropoutm=dropoutm)
        self.nslot = nslot

    def init_hidden(self, bsz):
        return self.OM_forward.init_hidden(bsz)

    def forward(self, input):
        batch_size = input.size(1)
        input = input[1:]
        mask = (input != self.padding_idx)
        lengths = mask.sum(0) - 2
        input[lengths + 1, torch.arange(batch_size)] = self.padding_idx
        input = input[:-1]
        mask = mask[:-1]

        X = self.embedding(input)

        init_hidden = self.init_hidden(batch_size)
        X, X_aux = X.chunk(2, dim=-1)

        rnn_out, _ = self.rnn_encode(X, mask)
        om_input, memories, probs = self.OM_forward(rnn_out, init_hidden, mask)

        expanded_mask = mask[:, :, None].expand(-1, -1, self.nslot)
        final_state = memories[lengths - 1,
                               torch.arange(batch_size, dtype=torch.long), -1]
        flattened_internal = memories.permute(0, 2, 1, 3).flatten(0, 1)
        flattened_internal_mask = expanded_mask.permute(0, 2, 1).flatten(0, 1)
        return (final_state,
                flattened_internal, flattened_internal_mask,
                om_input, X_aux, mask)

class RNNContextEncoder(nn.Module):
    def __init__(self, input_size, slot_size, nslot,
                 ntokens, padding_idx, dropout=0.2, dropoutm=0.1):

        super(RNNContextEncoder, self).__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(ntokens, 2 * input_size,
                                      padding_idx=self.padding_idx)
        self.rnn_encode = RNNEncoder(input_size, num_layers=nslot,
                                     dropout=dropoutm)
        self.last_transform = nn.Sequential(
            nn.Linear(nslot * slot_size, slot_size),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        batch_size = input.size(1)
        input = input[1:]
        mask = (input != self.padding_idx)
        lengths = mask.sum(0) - 2
        input[lengths + 1, torch.arange(batch_size)] = self.padding_idx
        input = input[:-1]
        mask = mask[:-1]

        X = self.embedding(input)

        X, X_aux = X.chunk(2, dim=-1)
        rnn_out, last = self.rnn_encode(X, mask)
        final_state = self.last_transform(last.permute(1, 0, 2).flatten(1, 2))
        return (final_state,
                rnn_out, mask,
                rnn_out, X_aux, mask)






class RNNEncoder(nn.Module):
    def __init__(self, input_size, num_layers=1, dropout=0.0):
        super(RNNEncoder, self).__init__()
        hidden_size = input_size
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout
        )

    def forward(self, X, mask):
        batch_size = X.size(1)
        lengths = mask.sum(0)

        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        rev_sorted_idx = torch.empty_like(sorted_idx)
        rev_sorted_idx[sorted_idx] = torch.arange(batch_size,
                                                  device=X.device,
                                                  dtype=torch.long)

        rnn_input = X[:, sorted_idx]
        rnn_input = nn.utils.rnn.pack_padded_sequence(
            rnn_input, sorted_lengths
        )
        hiddens, last = self.rnn(rnn_input)
        hiddens, _ = nn.utils.rnn.pad_packed_sequence(hiddens)
        hiddens = hiddens[:, rev_sorted_idx, :]
        last = last[:, rev_sorted_idx, :]
        return hiddens, last

