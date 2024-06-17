# from cProfile import label
import torch
import torch.nn as nn
import torch.nn.functional as F
# from opt_einsum import contract
import numpy as np
import json

from ttm import TokenTuringMachineEncoder

ID2REL = {"P6": "head of government", "P17": "country", "P19": "place of birth", "P20": "place of death", "P22": "father", "P25": "mother", "P26": "spouse", "P27": "country of citizenship", "P30": "continent", "P31": "instance of", "P35": "head of state", "P36": "capital", "P37": "official language", "P39": "position held", "P40": "child", "P50": "author", "P54": "member of sports team", "P57": "director", "P58": "screenwriter", "P69": "educated at", "P86": "composer", "P102": "member of political party", "P108": "employer", "P112": "founded by", "P118": "league", "P123": "publisher", "P127": "owned by", "P131": "located in the administrative territorial entity", "P136": "genre", "P137": "operator", "P140": "religion", "P150": "contains administrative territorial entity", "P155": "follows", "P156": "followed by", "P159": "headquarters location", "P161": "cast member", "P162": "producer", "P166": "award received", "P170": "creator", "P171": "parent taxon", "P172": "ethnic group", "P175": "performer", "P176": "manufacturer", "P178": "developer", "P179": "series", "P190": "sister city", "P194": "legislative body", "P205": "basin country", "P206": "located in or next to body of water", "P241": "military branch", "P264": "record label", "P272": "production company", "P276": "location", "P279": "subclass of", "P355": "subsidiary", "P361": "part of", "P364": "original language of work", "P400": "platform", "P403": "mouth of the watercourse", "P449": "original network", "P463": "member of", "P488": "chairperson", "P495": "country of origin", "P527": "has part", "P551": "residence", "P569": "date of birth", "P570": "date of death", "P571": "inception", "P576": "dissolved, abolished or demolished", "P577": "publication date", "P580": "start time", "P582": "end time", "P585": "point in time", "P607": "conflict", "P674": "characters", "P676": "lyrics by", "P706": "located on terrain feature", "P710": "participant", "P737": "influenced by", "P740": "location of formation", "P749": "parent organization", "P800": "notable work", "P807": "separated from", "P840": "narrative location", "P937": "work location", "P1001": "applies to jurisdiction", "P1056": "product or material produced", "P1198": "unemployment rate", "P1336": "territory claimed by", "P1344": "participant of", "P1365": "replaces", "P1366": "replaced by", "P1376": "capital of", "P1412": "languages spoken, written or signed", "P1441": "present in work", "P3373": "sibling"}

def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens, max_len=512):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= max_len:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= max_len:
                new_input_ids.append(input_ids[i, :max_len])
                new_attention_mask.append(attention_mask[i, :max_len])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :max_len - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - max_len + len_start): l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :max_len]
                attention_mask2 = attention_mask[i, (l_i - max_len): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - max_len))
                att = F.pad(attention[i], (0, c - max_len, 0, c - max_len))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:
                output1 = sequence_output[i][:max_len - len_end]
                mask1 = attention_mask[i][:max_len - len_end]
                att1 = attention[i][:, :max_len - len_end, :max_len - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - max_len + len_end))
                mask1 = F.pad(mask1, (0, c - max_len + len_end))
                att1 = F.pad(att1, (0, c - max_len + len_end, 0, c - max_len + len_end))

                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                output2 = F.pad(output2, (0, 0, l_i - max_len + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - max_len + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - max_len + len_start, c - l_i, l_i - max_len + len_start, c - l_i])
                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return sequence_output, attention        

class DocREModel(nn.Module):
    def __init__(self, args, config, priors_l, model, tokenizer, emb_size=768, block_size=64):
        super().__init__()
        self.args = args
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size
        self.priors_l = priors_l
        self.priors_o = priors_l * args.e
        self.priors_u = (self.priors_o - self.priors_l) / (1. - self.priors_l)
        self.weight = ((1 - self.priors_o)/self.priors_o) ** 0.5
        self.margin = args.m
        self.retrieval_weight = .2
        self.train_mode = 'finetune' # 'finetune' or 'pretrain' 
        if args.isrank:
            self.rels = args.num_class-1
        else:
            self.rels = args.num_class
        self.emb_size = emb_size
        self.block_size = block_size

        # ========== ATLOP ==========
        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        # ========== simple ==========
        if self.args.model_type == 'simple':
            self.simple_bilinear = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8, batch_first=True,
                                                                                    dropout=0.1, dim_feedforward=2048),
                                                         num_layers=2)
            self.simple_bilinear2 = nn.Sequential(nn.SELU(), nn.Linear(config.hidden_size, config.num_labels))

        # ========== ttmre ==========
        if self.args.model_type == 'ttmre':
            self.mu_encoder = TokenTuringMachineEncoder(process_size=2, memory_size=200, input_dim=emb_size, mlp_dim=emb_size, num_layers=args.num_layers)
            self.dropout = nn.Dropout(p=0.1)


    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_sims(self, sentences, batch_size=32):
        """ compute relation embeddings
        sentences: LIST of the text to embed
        batch_size: INT of batch size used for the computation
        """
        self.model.eval()
        all_embeddings = []
        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences[start_index:start_index + batch_size]
            features = self.tokenizer(sentences_batch, return_tensors='pt', truncation=True, max_length=512, padding=True)
            features = features.to(self.args.device)
            # print(features)
            out_features = self.model.forward(**features)
            embeddings = self.mean_pooling(out_features, features['attention_mask'])
            all_embeddings.extend(embeddings)

        all_embeddings = torch.stack(all_embeddings)  # Converts to tensor
        self.model.train()
        # print(all_embeddings.shape)
        return all_embeddings

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)
            entity_atts = torch.stack(entity_atts, dim=0)

            if len(hts[i]) == 0:
                hss.append(torch.FloatTensor([]).to(sequence_output.device))
                tss.append(torch.FloatTensor([]).to(sequence_output.device))
                rss.append(torch.FloatTensor([]).to(sequence_output.device))
                continue
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            # print("h_att", h_att.shape, "t_att", t_att.shape)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            # print("ht_att", ht_att.shape)
            rs = torch.einsum("ld,rl->rd", sequence_output[i], ht_att)
            # print("rs", rs.shape)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def square_loss(self, yPred, yTrue, margin=1.):
        if len(yPred) == 0:
            return torch.FloatTensor([0]).cuda()
        loss = (yPred * yTrue - margin) ** 2
        return torch.mean(loss.sum() / yPred.shape[0])

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                sampled_docs=None
                ):
        risk_sum = []
        
        if self.train_mode in ['pretrain']:
            with torch.no_grad():
                sequence_output, attention = self.encode(input_ids, attention_mask)
        else:
            sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)
        # print(hs.shape, rs.shape, ts.shape)

        if self.args.model_type == 'ATLOP':
            hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))    # zs
            ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))    # zo
            b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
            b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
            # print(b1.shape, b2.shape, (b1.unsqueeze(3) * b2.unsqueeze(2)).shape)
            bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
            logits = self.bilinear(bl)
            logits_list = [logits,]
            loss_weights_list = [1.]
            m_tags_list = [self.args.m_tag]

        if self.args.model_type == 'mse_dist2':
            topk = 3

            hs_ = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))    # zs
            ts_ = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))    # zo
            b1 = hs_.view(-1, self.emb_size // self.block_size, self.block_size)
            b2 = ts_.view(-1, self.emb_size // self.block_size, self.block_size)
            bl1 = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
            bl1 = self.dropout(bl1)
            logits1 = self.bilinear(bl1)

            # # retrieval of pseudo documents
            hs_weights = torch.matmul(hs, self.label_mu_extractor3.transpose(0, 1))
            ts_weights = torch.matmul(ts, self.label_mu_extractor3.transpose(0, 1))

            hs_topk_inds = torch.topk(hs_weights, k=topk, dim=-1).indices
            hs2 = self.mu_encoder(torch.cat([hs.unsqueeze(1), self.label_mu_extractor3[hs_topk_inds]], dim=1))
            hs2 = hs2[:,0,:]

            ts_topk_inds = torch.topk(ts_weights, k=topk, dim=-1).indices
            ts2 = self.mu_encoder(torch.cat([ts.unsqueeze(1), self.label_mu_extractor3[ts_topk_inds]], dim=1))
            ts2 = ts2[:,0,:]

            hs2_ = torch.tanh(self.head_extractor(torch.cat([hs2, rs], dim=1)))    # zs
            ts2_ = torch.tanh(self.tail_extractor(torch.cat([ts2, rs], dim=1)))    # zo
            b1 = hs2_.view(-1, self.emb_size // self.block_size, self.block_size)
            b2 = ts2_.view(-1, self.emb_size // self.block_size, self.block_size)
            bl2 = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
            bl2 = self.dropout(bl2)
            logits2 = self.bilinear(bl2)

            max_confidence = nn.Sigmoid()(torch.max(logits1[:,1:] - logits1[:,0:1], dim=1).values)
            max_confidence = max_confidence.unsqueeze(1)
            logits_list = [logits1*max_confidence + logits2*(1-max_confidence), logits1]
            loss_weights_list = [1, 1]
            m_tags_list = [self.args.m_tag, self.args.m_tag]

        if self.args.model_type == 'simple':
            logits = self.simple_bilinear(torch.stack([hs, ts, rs], dim=1))
            logits = self.simple_bilinear2(logits.mean(dim=1))
            # print(logits.shape)

            logits_list = [logits]
            loss_weights_list = [1.]
            m_tags_list = [self.args.m_tag,]

        if self.args.model_type == 'mse_dist3':
            hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))    # zs
            ts = torch.tanh(self.head_extractor(torch.cat([ts, rs], dim=1)))    # zo

            # print(torch.cat([hs.unsqueeze(1), ts.unsqueeze(1)], dim=1).unsqueeze(2).shape)
            hs2 = []
            ts2 = []

            step = 256
            for batch_2 in range(0, hs.shape[0], step):
                encoded = self.mu_encoder(torch.cat([hs[batch_2:batch_2+step].unsqueeze(1), ts[batch_2:batch_2+step].unsqueeze(1)], dim=1).unsqueeze(1))
                # print(encoded.shape, self.mu_encoder.memory_tokens.data.shape); quit()
                hs2.append(encoded[:,0,0,:])
                ts2.append(encoded[:,0,1,:])
            hs2 = torch.cat(hs2, dim=0)
            ts2 = torch.cat(ts2, dim=0)
            self.sims = [hs2, ts2]

            b1 = (hs2/2 + hs/2).view(-1, self.emb_size // self.block_size, self.block_size)
            b2 = (ts2/2 + ts/2).view(-1, self.emb_size // self.block_size, self.block_size)

            bl2 = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
            logits2 = self.bilinear(bl2)

            logits_list = [logits2]
            loss_weights_list = [1]
            m_tags_list = [self.args.m_tag, ]

        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(self.args.device)

            for logits, loss_weight, m_tag in zip(logits_list, loss_weights_list, m_tags_list):
                if m_tag == 'increase':
                    risk_sum.append(-logits.mean() * loss_weight)
                    # continue
                    
                if m_tag == 'ATLoss':
                    assert self.args.isrank == True
                    """https://github.com/YoumiMa/dreeam/blob/main/losses.py"""
                    labels = labels.clone()
                    th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
                    th_label[:, 0] = 1.0
                    labels[:, 0] = 0.0
                    # Rank positive classes highly
                    logit1 = logits - (1 - labels - th_label) * 1e30
                    loss1 = -(nn.functional.log_softmax(logit1, dim=-1) * labels).sum(1)
                    # Rank negative classes lowly
                    logit2 = logits - labels * 1e30
                    loss2 = -(nn.functional.log_softmax(logit2, dim=-1) * th_label).sum(1)
                    # Sum two parts
                    loss = loss1 + loss2
                    risk_sum.append(loss.mean() * loss_weight)

                if m_tag == 'pos-ATLoss':
                    assert self.args.isrank == True
                    labels = labels.clone()
                    th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
                    th_label[:, 0] = 1.0
                    labels[:, 0] = 0.0
                    # Rank positive classes highly
                    logit1 = logits - (1 - labels - th_label) * 1e30
                    loss = -(nn.functional.log_softmax(logit1, dim=-1) * labels).sum(1)
                    risk_sum.append(loss.mean() * loss_weight)

                if m_tag == 'AFLoss':
                    assert self.args.isrank == True
                    gamma_pos = 1.0
                    labels = labels.clone()
                    # Adapted from Focal loss https://arxiv.org/abs/1708.02002, multi-label focal loss https://arxiv.org/abs/2009.14119
                    # TH label 
                    th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
                    th_label[:, 0] = 1.0
                    labels[:, 0] = 0.0

                    p_mask = labels + th_label
                    n_mask = 1 - labels
                    neg_target = 1- p_mask
                    
                    num_ex, num_class = labels.size()
                    # Rank each positive class to TH
                    logit1 = logits - neg_target * 1e30

                    # Rank each class to threshold class TH
                    th_mask = torch.cat( num_class * [logits[:,:1]], dim=1)
                    logit_th = torch.cat([logits.unsqueeze(1), 1.0 * th_mask.unsqueeze(1)], dim=1) 
                    log_probs = F.log_softmax(logit_th, dim=1)
                    probs = torch.exp(F.log_softmax(logit_th, dim=1))

                    # Probability of relation class to be negative (0)
                    prob_0 = probs[:, 1 ,:]
                    prob_0_gamma = torch.pow(prob_0, gamma_pos)
                    log_prob_1 = log_probs[:, 0 ,:]

                    # Rank TH to negative classes
                    logit2 = logits - (1 - n_mask) * 1e30
                    rank2 = F.log_softmax(logit2, dim=-1)

                    loss1 = - (log_prob_1 * (1 + prob_0_gamma ) * labels) 
                    loss2 = -(rank2 * th_label).sum(1) 
                    loss =  1.0 * loss1.sum(1).mean() + 1.0 * loss2.mean()
                    risk_sum.append(loss * loss_weight)

                if m_tag == 'S-PU':

                    risk_sum_ = torch.FloatTensor([0]).cuda()

                    for i in range(self.rels):
                        neg = (logits[(labels[:, i + 1] != 1), i + 1] - logits[(labels[:, i + 1] != 1), 0])
                        pos = (logits[(labels[:, i + 1] == 1), i + 1] - logits[(labels[:, i + 1] == 1), 0])

                        priors_u = (self.priors_o[i] - self.priors_l[i]) / (1. - self.priors_l[i])
                        risk1 = ((1. - self.priors_o[i]) / (1. - priors_u)) * self.square_loss(neg, -1., self.margin) - ((priors_u - priors_u * self.priors_o[i]) / (1. - priors_u)) * self.square_loss(pos, -1., self.margin)
                        risk2 = self.priors_o[i] * self.square_loss(pos, 1., self.margin) * self.weight[i]
                        risk = risk1 + risk2

                        if risk1 < self.args.beta:
                            risk = - self.args.gamma * risk1
                        risk_sum_ += risk
                    risk_sum.append(risk_sum_ * loss_weight)

                if m_tag == 'PU':
                    risk_sum_ = torch.FloatTensor([0]).cuda()

                    for i in range(self.rels):
                        neg = (logits[(labels[:, i + 1] != 1), i + 1] - logits[(labels[:, i + 1] != 1), 0])
                        pos = (logits[(labels[:, i + 1] == 1), i + 1] - logits[(labels[:, i + 1] == 1), 0])

                        # risk1 = U(-)_risk - P(-)_risk
                        risk1 = (self.square_loss(neg, -1., self.margin) -
                                    self.priors_o[i] * self.square_loss(pos, -1., self.margin))
                        # risk2 = P(+)_risk
                        risk2 = self.priors_o[i] * self.square_loss(pos, 1., self.margin) * self.weight[i]
                        risk = risk1 + risk2

                        if risk1 < self.args.beta:
                            risk = - self.args.gamma * risk1
                        risk_sum_ += risk
                    risk_sum.append(risk_sum_ * loss_weight)

            
            return risk_sum, logits_list[0]
        


        return logits_list[0]

