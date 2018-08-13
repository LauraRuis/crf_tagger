import torch.nn as nn
import torch
import torch.nn.functional as F
from helpers import log_sum_exp, argmax

START_TAG = "<START>"
STOP_TAG = "<STOP>"


class BiLSTM_ChainCRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_ChainCRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.log_transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.log_transitions.data[tag_to_ix[START_TAG], :] = -10000.
        self.log_transitions.data[:, tag_to_ix[STOP_TAG]] = -10000.

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):

        # initialize the recursion variables
        init_alphas = torch.full((1, self.tagset_size), 0.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 1.

        # first scale
        scales = torch.full((1, feats.size(0) + 1), 0.)
        scales[0][0] = init_alphas.sum()

        # scale alphas to avoid underflow
        scaled_alphas = init_alphas / scales[0][0]

        # iterate through the sentence
        for t, feat in enumerate(feats):

            # recursion variables at t
            alphas_t = []

            for next_tag in range(self.tagset_size):

                # emission score the same regardless of previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)

                # transition score
                trans_score = torch.exp(self.log_transitions[next_tag].view(1, -1))

                # scaled recursion: delta_t(j) = c_t*alpha_hat_t(j) = sum_i (alpha_hat_t-1(i) * phi_t(j, i, x_t))
                next_tag_var = torch.sum(scaled_alphas * trans_score * torch.exp(emit_score))

                # save alpha for this tag
                alphas_t.append(next_tag_var.view(1, -1))

            # scale alphas again and save scale
            scaled_alphas = torch.cat(alphas_t).view(1, -1)
            scale = scaled_alphas.sum()
            scales[0][t + 1] = scale
            scaled_alphas /= scale

        # Z(x) = prod_t c_t (Bishop) -> log(Z(x)) = sum_t log(c_t) (partition func. is product of scales)
        return torch.sum(torch.log(scales))

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.log_transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.log_transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.log_transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.log_transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq