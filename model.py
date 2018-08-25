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

    def _forward_belief_prop(self, feats):

        # initialize the recursion variables
        init_alphas = torch.full((1, self.tagset_size), 0.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 1.

        # initial scale
        scales = torch.full((1, feats.size(0) + 1), 0.)
        scales[0][0] = init_alphas.sum()

        # scale alphas to avoid underflow
        scaled_alphas = init_alphas / scales[0][0]
        scaled_alphas_2 = init_alphas / scales[0][0]

        bsz, time, dim = feats.unsqueeze(0).size()

        # iterate through the sentence
        for t, feat in enumerate(feats):

            # calculate emission probabilities (the same regardless of the next tag)
            emit_scores = feat.view(bsz, self.tagset_size).unsqueeze(2)

            # calculate transition probabilities
            trans_scores = torch.exp(self.log_transitions).unsqueeze(0)

            # update alphas recursively
            scaled_alphas = (scaled_alphas * trans_scores * torch.exp(emit_scores)).sum(2)

            # scale and save scale
            scale = scaled_alphas.sum()
            scales[0][t + 1] = scale
            scaled_alphas /= scale

        # partition func. is product of scales (Z(x) = prod_t c_t -> log(Z(x)) = sum_t log(c_t))
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

        # initialize list to keep track of backpointers
        backpointers = []

        # initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars.unsqueeze(0)

        bsz, time, dim = feats.unsqueeze(0).size()

        for feat in feats:

            # calculate scores of next tag per tag
            forward_var = forward_var.view(bsz, 1, self.tagset_size)
            trans_scores = self.log_transitions.unsqueeze(0)
            next_tag_vars = forward_var + trans_scores

            # get best next tags and viterbi vars
            _, idx = torch.max(next_tag_vars, 2)
            best_tag_ids = idx.view(bsz, -1)
            indices = torch.transpose(best_tag_ids.unsqueeze(0), 1, 2)
            viterbivars_t = torch.gather(next_tag_vars, 2, indices).squeeze(2)

            # add emission scores and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (viterbivars_t + feat).view(1, -1)
            backpointers.append(best_tag_ids[0].tolist())

        # transition to STOP_TAG
        terminal_var = forward_var + self.log_transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # follow the back pointers to decode the best path
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_belief_prop(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_belief_prop above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq