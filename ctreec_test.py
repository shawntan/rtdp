import unittest
import ctreec
import tree_decoder
import torch


class TestCTreeC(unittest.TestCase):

    def setUp(self):
        self.omd = tree_decoder.OrderedMemoryDecoder(
            ntoken=11, slot_size=20,
            producer_class='Cell',
            max_depth=3,
            leaf_dropout=0,
            output_dropout=0,
            integrate_dropout=0,
            attn_dropout=0,
            node_attention=False,
            output_attention=False
        )
        self.loss = ctreec.Loss(depth=3)
        self.length = 4

    def test_variable_length_batch(self):
        batch_size = 5
        labels, root, context, lengths = self._generate_data(batch_size,
                                                             self.length)
        log_probs = self.omd(root, context, labels)
        batched_ctreec_log_probs = self.loss(log_probs, labels, lengths)
        for i in range(batch_size):
            with self.subTest(batch_idx=i):
                idv_ctreec_log_probs = self.loss(
                    log_probs[:, i:i+1], labels[:, i:i+1], lengths[i:i+1])
                self.assertAlmostEqual(
                    batched_ctreec_log_probs[i].item(),
                    idv_ctreec_log_probs[0].item(),
                    places=5,
                    msg="Not computing right loss for i = %d" % i
                )

    def test_singleton(self):
        labels, root, context, lengths = self._generate_data(1, 1)

        log_probs = self.omd(root, context, labels)
        ctreec_log_probs = self.loss(
            log_probs, labels,
            torch.full_like(labels[0, :], labels.size(0))
        )
        ctreec_neg_log_probs = ctreec_log_probs[0].item()
        ext_log_probs = ctreec.extract_label_log_probs(log_probs, labels)
        manual_neg_log_probs = -ext_log_probs[7, 0, 0].item()
        self.assertAlmostEqual(
            manual_neg_log_probs, ctreec_neg_log_probs,
            places=5,
            msg="Marginalisation incorrect for length = 1"
        )

    def _generate_data(self, batch_size, max_length):
        root = torch.randn(batch_size, 20)
        context = torch.randn(20, batch_size, 20)
        labels = torch.randint(10, size=(max_length, batch_size))
        lengths = torch.randint(0, max_length, size=(batch_size,)) + 1
        lengths, _ = lengths.sort(descending=True)
        mask = torch.ones_like(context[:, :, 0]).bool()
        return labels, root, (context, context, mask, context, context, mask), lengths

    def test_marginalisation(self):
        labels, root, context, lengths = self._generate_data(1, self.length)
        log_probs = self.omd(root, context, labels)
        ctreec_log_probs = self.loss(
            log_probs, labels,
            torch.full_like(labels[0, :], labels.size(0))
        )
        ctreec_neg_log_probs = ctreec_log_probs[0].item()
        ext_log_probs = ctreec.extract_label_log_probs(log_probs, labels)
        paths = torch.stack((
            ext_log_probs[[0, 2,  5, 11], 0, [0, 1, 2, 3]],
            ext_log_probs[[1, 4,  6, 11], 0, [0, 1, 2, 3]],
            ext_log_probs[[1, 5,  9, 13], 0, [0, 1, 2, 3]],
            ext_log_probs[[3, 8, 10, 13], 0, [0, 1, 2, 3]],
            ext_log_probs[[3, 9, 12, 14], 0, [0, 1, 2, 3]]
        ))
        manual_neg_log_probs = -paths.sum(1).logsumexp(0).item()
        self.assertAlmostEqual(
            manual_neg_log_probs, ctreec_neg_log_probs,
            places=5,
            msg="Marginalisation incorrect for length = 4"
        )

    def test_log_space(self):
        labels, root, context, _ = self._generate_data(1, self.length)
        log_probs = self.omd(root, context, labels)
        ctreec_log_probs = self.loss(
            log_probs, labels,
            torch.full_like(labels[0, :], labels.size(0))
        )
        ctreec_neg_log_probs = ctreec_log_probs[0].item()

        ext_log_probs = ctreec.extract_label_log_probs(log_probs, labels)

        ext_probs = torch.exp(ext_log_probs).permute(2, 1, 0)
        prev_probs = torch.zeros_like(ext_probs[0])
        prev_probs[:, self.loss.start_idxs] = \
            ext_probs[0, :, self.loss.start_idxs]
        for t in range(1, self.length):
            curr_probs = torch.matmul(prev_probs, self.loss.transition)
            prev_probs = curr_probs * ext_probs[t]
        exp_space_neg_log_probs = -torch.log(
            prev_probs[:, self.loss.end_idxs].sum()).item()

        self.assertAlmostEqual(
            exp_space_neg_log_probs, ctreec_neg_log_probs,
            places=5,
            msg="Log-space modifications incorrect.")

    def test_nan(self):
        batch_size = 5
        labels, root, context, lengths = self._generate_data(batch_size, self.length)
        log_probs = self.omd(root, context, labels)
        batched_ctreec_log_probs = self.loss(log_probs, labels, lengths)
        torch.autograd.set_detect_anomaly(True)
        batched_ctreec_log_probs.mean().backward()
        torch.autograd.set_detect_anomaly(False)


if __name__ == "__main__":
    unittest.main()
