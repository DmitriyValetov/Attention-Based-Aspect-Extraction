import torch


class ABAE(torch.nn.Module):
    def __init__(self, word_num, aspects_num=14, emb_size=200):
        super(ABAE, self).__init__()
        self.emb_size = emb_size
        self.word_num = word_num
        self.aspects_num = aspects_num
        self.w_embs = torch.nn.Embedding(word_num, emb_size)
        self.a_embs = torch.randn(aspects_num, emb_size, requires_grad=True)
        self.attention_M = torch.randn(emb_size, emb_size, requires_grad=True)
        self.sentence_aspects = torch.nn.Linear(emb_size, aspects_num)
        
    def make_sentence(self, embs):
        """
        input: embedded words
        output: weights
        """
        ys = torch.mean(embs, axis=0).unsqueeze(dim=1)
        di = torch.mm(embs, torch.mm(self.attention_M, ys))
        ai = torch.nn.functional.softmax(di, dim=1)
        zs = torch.mm(embs.T, ai)
        return zs
        
    def make_sentence_raw(self, embs):
        return torch.mean(embs, axis=0).unsqueeze(dim=1)
        
    def forward(self, ps, ns_list):
        """
        ps - positive sequence. Tensor of shape (sentence_length,)
        ns_list - list of negaitve sequence. Tensor of shape [(sentence_length,), ...]
        """
        embs = self.w_embs(ps) # (sentence_length, emb_dim)
        zs = self.make_sentence(embs) # sentence from attentioned words vectors
        pt = torch.nn.functional.softmax(self.sentence_aspects(zs.T), dim=1).T
        rs = torch.mm(self.a_embs.T, pt) # sentence reconstraction by aspects vectors
        return rs, zs, [self.make_sentence_raw(self.w_embs(ns)) for ns in ns_list]