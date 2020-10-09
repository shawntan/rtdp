import torch
import torch.nn as nn
import ordered_memory
import tree_decoder
import tree_rnn


class SCANModel(nn.Module):
    def __init__(self, args):
        super(SCANModel, self).__init__()
        self.args = args
        self.drop_input = nn.Dropout(args.dropouti)

        if args.encoder_type == 'OM':
            self.encoder = ordered_memory.OrderedMemory(
                args.ninp, args.nhid, 5,
                ntokens=args.src_ntoken,
                padding_idx=args.src_padding_idx,
                dropout=args.dropout, dropoutm=args.dropoutm,
            )
        elif args.encoder_type == 'tree_rnn':
            self.encoder = tree_rnn.TreeRNN(
                ntoken=args.src_ntoken,
                nhid=args.nhid,
                padding_idx=args.src_padding_idx,
                parens_id=(args.paren_open, args.paren_close)
            )
        elif args.encoder_type == 'birnn':
            self.encoder = ordered_memory.RNNContextEncoder(
                args.ninp, args.nhid, 5,
                ntokens=args.src_ntoken,
                padding_idx=args.src_padding_idx,
                dropout=args.dropout, dropoutm=args.dropoutm,
            )


        self.decoder = tree_decoder.OrderedMemoryDecoder(
            ntoken=args.trg_ntoken,
            slot_size=args.nhid,
            producer_class=args.prod_class,
            padding_idx=args.trg_padding_idx,
            leaf_dropout=args.dec_leaf_dropout,
            output_dropout=args.dec_out_dropout,
            integrate_dropout=args.dec_int_dropout,
            attn_dropout=args.dec_attn_dropout,
            node_attention=args.dec_no_node_attn,
            output_attention=args.dec_no_leaf_attn,
            max_depth=args.nslot,
        )

    def decode(self, input):
        input = input.transpose(0, 1)
        (final_state,
         flattened_internal, flattened_internal_mask,
         rnned_X, X_emb, mask) = self.encoder(input)
        context = (
            flattened_internal, flattened_internal_mask,
            rnned_X, X_emb, mask
        )
        return self.decoder.decode(final_state, context)

    def forward(self, input, target, eval_loss=False):
        input = input.transpose(0, 1)
        (final_state,
         flattened_internal, flattened_internal_mask,
         rnned_X, X_emb, mask) = self.encoder(input)

        context = (
            flattened_internal, flattened_internal_mask,
            rnned_X, X_emb, mask
        )
        if self.training:
            return self.decoder.compute_loss(
                final_state, context,
                target.transpose(0, 1))
        else:
            if eval_loss:
                return self.decoder.compute_loss(
                    final_state, context,
                    target.transpose(0, 1)
                ) * target.size(0)
            else:
                return self.decoder.max_prob(
                   final_state, context, target.transpose(0, 1))


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, seq_len, hidden):
        bsz = hidden.size(0)
        x = hidden.new(seq_len, bsz, self.hidden_size).zero_()
        output, hidden = self.gru(x, hidden[None, :, :])
        output = self.out(output)
        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
