import os
os.environ['TRANSFORMERS_CACHE'] = '/srv/local/data/chufan2/huggingface/'
import argparse
import json
import numpy as np
import torch
import copy
import pickle
# from apex import amp
# import ujson as json
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from tqdm import tqdm
import random

from model2 import DocREModel
from prepro import read_docred, read_chemdisgene
from evaluation import official_evaluate, to_official

MEMORY_SIZE = 200

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    output = (input_ids, input_mask, labels, entity_pos, hts)
    return output

def train(args, model, train_features, dev_features, save_best_val=True, lr=1e-4, save_after_epoch=10,
          test_features=None):
    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": lr},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
#     finetune(train_features, optimizer, args.num_train_epochs, num_steps)
# def finetune(features, optimizer, args.num_train_epochs, num_steps):

    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    train_iterator = range(int(args.num_train_epochs))
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    if args.model_type == "ATLOP":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    else:
        # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_model = None
    best_val_risk = np.inf

    print("Total steps: {}".format(total_steps))
    print("Warmup steps: {}".format(warmup_steps))
    for epoch in tqdm(train_iterator):
        model.zero_grad()

        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            # # print(switch)

            inputs = {'input_ids': batch[0].to(args.device),
                    'attention_mask': batch[1].to(args.device),
                    'labels': batch[2],
                    'entity_pos': batch[3],
                    'hts': batch[4],
                    # 'sampled_docs': sampled_docs,
                    }

            outputs = model(**inputs)
            loss = sum(outputs[0]) / args.gradient_accumulation_steps
            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            if step % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                num_steps += 1

            if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                print("training risk:", loss.item(), "   step:", num_steps)
                if "chemdisgene" in args.data_dir.lower():
                    avg_val_risk, test_output = cal_val_risk_bio(args, model, dev_features)
                else:
                    avg_val_risk, test_output = cal_val_risk(args, model, dev_features)
                print('avg val risk:', avg_val_risk, test_output, '\n')
                
                if test_features is not None:
                    if "chemdisgene" in args.data_dir.lower():
                        test_score, test_output = evaluate_bio(args, model, test_features, tag="test")
                    else:
                        test_score, test_output = evaluate(args, model, test_features, tag="test")
                    print('test risk:', test_score, test_output, '\n')
                    
                if (epoch > save_after_epoch) and (best_model is None) or (avg_val_risk[0] < best_val_risk):
                    best_val_risk = avg_val_risk[0]
                    # copy the model state dict
                    best_model = {k: v.cpu() for k, v in model.state_dict().items()}

    # load the best model
    if save_best_val:
        model.load_state_dict(best_model)
    # torch.save(model.state_dict(), os.path.join(args.save_path, "state_dict.pth"))
    return num_steps

def cal_val_risk(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    val_risk = []
    nums = 0
    preds = []
    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'labels': batch[2],
                      'entity_pos': batch[3],
                      'hts': batch[4]
                      }
            output = model(**inputs)
            logits = output[1]
            # print(len(logits))
            
            val_risk.append(np.array([risk.item() for risk in output[0]]))
            nums += 1

            logits = logits.cpu().numpy()
            if args.isrank:
                pred = np.zeros((logits.shape[0], logits.shape[1]))
                for i in range(1, logits.shape[1]):
                    pred[(logits[:, i] > logits[:, 0]), i] = 1
                pred[:, 0] = (pred.sum(1) == 0)
            else:
                pred = np.zeros((logits.shape[0], logits.shape[1] + 1))
                for i in range(logits.shape[1]):
                    pred[(logits[:, i] > 0.), i + 1] = 1
                pred[:, 0] = (pred.sum(1) == 0)

            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    if len(ans) > 0:
        best_f1, _, best_f1_ign, re_f1_ignore_train, re_p, re_r = official_evaluate(ans, args.data_dir, tag, args)
        output = {
            tag + "_F1": best_f1 * 100,
            tag + "_F1_ign": best_f1_ign * 100,
            "re_p": re_p * 100,
            "re_r": re_r * 100,
        }
    else:
        best_f1, best_f1_ign = -1, -1
        output = {
            tag + "_F1": best_f1 * 100,
            tag + "_F1_ign": best_f1_ign * 100,
        }
    # return np.stack(val_risk, axis=0).sum(axis=0) / nums,  output            
    return [-best_f1 * 100],  output

def evaluate(args, model, features, tag="test", eval_top_10=False):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    sims_list = []
    labels = []
    for batch in dataloader:
        model.eval()
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'labels': batch[2],
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            output = model(**inputs)
            logits = output[1].cpu().numpy()

            sims = [model.sims[0].cpu().numpy(), model.sims[1].cpu().numpy()]

            if args.isrank:
                pred = np.zeros((logits.shape[0], logits.shape[1]))
                for i in range(1, logits.shape[1]):
                    pred[(logits[:, i] > logits[:, 0]), i] = 1
                pred[:, 0] = (pred.sum(1) == 0)
            else:
                pred = np.zeros((logits.shape[0], logits.shape[1] + 1))
                for i in range(logits.shape[1]):
                    pred[(logits[:, i] > 0.), i + 1] = 1
                pred[:, 0] = (pred.sum(1) == 0)

            preds.append(pred)
            labels.append(batch[2])
            sims_list.append(sims)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)

    pickle.dump(sims_list, open(os.path.join(args.save_path, f"{tag}_sims.pkl"), 'wb'))
    pickle.dump(model.mu_encoder.memory_tokens.data.cpu().numpy(), open(os.path.join(args.save_path, f"{tag}_mem.pkl"), 'wb'))
    pickle.dump(preds, open(os.path.join(args.save_path, f"{tag}_preds.pkl"), 'wb'))
    pickle.dump(ans, open(os.path.join(args.save_path, f"{tag}_ans.pkl"), 'wb'))
    pickle.dump(labels, open(os.path.join(args.save_path, f"{tag}_labels.pkl"), 'wb'))

    if len(ans) > 0:
        if eval_top_10:
            best_f1, _, best_f1_ign, re_f1_ignore_train, re_p, re_r = official_evaluate(ans, args.data_dir, tag='testtop10', args=args)
            print("top10", best_f1, best_f1_ign, re_p, re_r)
            best_f1, _, best_f1_ign, re_f1_ignore_train, re_p, re_r = official_evaluate(ans, args.data_dir, tag='testbottom90', args=args)
            print("testbottom90", best_f1, best_f1_ign, re_p, re_r)
            
        best_f1, _, best_f1_ign, re_f1_ignore_train, re_p, re_r = official_evaluate(ans, args.data_dir, tag, args)
        output = {
            tag + "_F1": best_f1 * 100,
            tag + "_F1_ign": best_f1_ign * 100,
            "re_p": re_p * 100,
            "re_r": re_r * 100,
        }
    else:
        best_f1, best_f1_ign = -1, -1
        output = {
            tag + "_F1": best_f1 * 100,
            tag + "_F1_ign": best_f1_ign * 100,
        }
    return best_f1, output

def cal_val_risk_bio(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    val_risk = []
    nums = 0

    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'labels': batch[2],
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            # risk, logits = model(**inputs)
            # val_risk += risk.item()
            output = model(**inputs)
            # logits = output[1]
            val_risk.append(np.array([risk.item() for risk in output[0]]))

            nums += 1

    # return val_risk / nums
    return np.stack(val_risk, axis=0).sum(axis=0) / nums,  output            
            

def evaluate_bio(args, model, features, tag="test"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    golds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            logits = model(**inputs)
            logits = logits.cpu().numpy()

            if args.isrank:
                pred = np.zeros((logits.shape[0], logits.shape[1]))
                for i in range(1, logits.shape[1]):
                    pred[(logits[:, i] > logits[:, 0]), i] = 1
                pred[:, 0] = (pred.sum(1) == 0)
            else:
                pred = np.zeros((logits.shape[0], logits.shape[1] + 1))
                for i in range(logits.shape[1]):
                    pred[(logits[:, i] > 0.), i + 1] = 1
                pred[:, 0] = (pred.sum(1) == 0)

            preds.append(pred)
            labels = [np.array(label, np.float32) for label in batch[2]]
            golds.append(np.concatenate(labels, axis=0))

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = preds[:,1:]
    golds = np.concatenate(golds, axis=0).astype(np.float32)[:,1:]

    TPs = preds * golds  # (N, R)
    TP = TPs.sum()
    P = preds.sum()
    T = golds.sum()

    micro_p = TP / P if P != 0 else 0
    micro_r = TP / T if T != 0 else 0
    micro_f = 2 * micro_p * micro_r / \
        (micro_p + micro_r) if micro_p + micro_r > 0 else 0
    mi_output = {
            tag + "_F1": micro_f * 100,
            "re_p": micro_p * 100,
            "re_r": micro_r * 100,
        }

    return micro_f, mi_output

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--model_type", default="ATLOP", type=str)
    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--distant_file", default="train_distant.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="out", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument("--isrank", type=int, default='1 means use ranking loss, 0 means not use')
    parser.add_argument("--m_tag", type=str, default='PN/PU/S-PU')
    parser.add_argument('--beta', type=float, default=0.0, help='beta of pu learning (default 0.0)')
    parser.add_argument('--gamma', type=float, default=1.0, help='gamma of pu learning (default 1.0)')
    parser.add_argument('--m', type=float, default=1.0, help='margin')
    parser.add_argument('--e', type=float, default=3.0, help='estimated a priors multiple')
    parser.add_argument('--pretrain_distant', type=int, default=0, help='whether to pretrain distant and then quit')

    parser.add_argument('--num_layers', type=int, default=2, help="num_layers for ttm")
    parser.add_argument('--memory_size', type=int, default=200, help="memory_size for ttm, originally 200, cut to new_memory_size")

    args = parser.parse_args()
    # assert args.is_rank == 1

    file_name = "{}_{}_{}_{}_{}_isrank_{}_m_{}_e_{}_seed_{}".format(
        args.train_file.split('.')[0],
        args.transformer_type,
        args.model_type,
        args.data_dir.split('/')[-1],
        args.m_tag,
        str(args.isrank),
        args.m,
        args.e,
        str(args.seed))
    args.save_path = os.path.join(args.save_path, file_name)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    print(args.save_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # print({k:str(v) for k,v in vars(args).items()}); quit()

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    if "chemdisgene" in args.data_dir.lower():
        read = read_chemdisgene
    else:
        read = read_docred

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)

    train_features, priors = read(args, train_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features, _ = read(args, dev_file, tokenizer, max_seq_length=args.max_seq_length)
    test_features, _ = read(args, test_file, tokenizer, max_seq_length=args.max_seq_length)

    # train_features = train_features[:100]
    # dev_features = dev_features[:100]
    # test_features = test_features[:100]

    # what if we use true priors?
    # test_features, priors = read(args, test_file, tokenizer, max_seq_length=args.max_seq_length)
    priors += 1e-9

    # dev_features = train_features + dev_features

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    ).to(args.device)

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    # print('priors', priors); quit()
    priors = torch.tensor(priors).to(args.device)
    model = DocREModel(args, config, priors, model, tokenizer)
    model.to(0)

    print(args.m_tag, args.isrank)

    if args.load_path == "":  # Training
        if args.model_type in ['simple', 'ttmre', 'ATLOP']:
            print("PRETRAINING")
            print("pretrain distant", args.pretrain_distant)
            temp_epochs = args.num_train_epochs
            args.num_train_epochs = 2
            if args.pretrain_distant == 0: # pretrain on train and quit()
                train(args, model, train_features, dev_features, lr=1e-4)
                torch.save(model.state_dict(), os.path.join(args.save_path, "pretrain_state_dict.pth")); quit()
            if args.pretrain_distant == 1: # pretrain on distant and quit()
                if os.path.isfile(f"./distant_features_{args.model_name_or_path}.pkl"):
                    distant_features = pickle.load(open(f"./distant_features_{args.model_name_or_path}.pkl", 'rb'))
                else:
                    distant_file = os.path.join(args.data_dir, args.distant_file)
                    distant_features, _ = read(args, distant_file, tokenizer, max_seq_length=args.max_seq_length)
                train(args, model, distant_features, dev_features, lr=5e-5)
                torch.save(model.state_dict(), os.path.join(args.save_path, "pretrain_state_dict.pth")); quit()

            if args.pretrain_distant == 2: # load pretrain and finetune on train
                print("loading", os.path.join(args.save_path, "pretrain_state_dict.pth"))
                model.load_state_dict(torch.load(os.path.join(args.save_path, "pretrain_state_dict.pth")))
                # model.mu_encoder.memory_tokens.requires_grad_(False); print(model.mu_encoder.memory_tokens.requires_grad)
                if args.memory_size != MEMORY_SIZE:
                    print("cutting memory size to ", args.memory_size)
                    model.mu_encoder.memory_tokens.data = model.mu_encoder.memory_tokens.data[:args.memory_size]

            if "chemdisgene" in args.data_dir.lower():
                test_score, test_output = evaluate_bio(args, model, test_features, tag="test")
            else:
                test_score, test_output = evaluate(args, model, test_features, tag="test")
                # quit()
            print("pretrain performance", test_output)
            
            print("FINETUNING")
            args.num_train_epochs = temp_epochs
            model.train_mode = 'finetune'

        # if args.pretrain_distant == 3: # finetune on train only, no pretrain
        if args.pretrain_distant == 2 or args.pretrain_distant == 3: 
            train(args, model, train_features, dev_features, save_best_val=True, save_after_epoch=0, lr=1e-5,
                test_features=test_features)
            # train(args, model, train_features, dev_features, save_best_val=False)
            torch.save(model.state_dict(), os.path.join(args.save_path, "finetune_state_dict.pth"))

        print("TEST")
        # if 4, just load finetune_state_dict and eval          
        model.load_state_dict(torch.load(os.path.join(args.save_path, "finetune_state_dict.pth")))
        test_score, test_output = evaluate(args, model, test_features, tag="test", eval_top_10=True)
        print("finetune", test_output)

        # dump test_output to json file
        with open(os.path.join(args.save_path,'test_output.json'), 'w') as f:
            json.dump(test_output, f)
        # dump args to json file
        with open(os.path.join(args.save_path,'args.json'), 'w') as f:
            json.dump({k:str(v) for k,v in vars(args).items()}, f)
    else:  # Testing
        args.load_path = os.path.join(args.load_path, file_name)
        print(args.load_path)
        
        print("TEST")
        # model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(os.path.join(args.save_path, "state_dict.pth")))
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print(test_output)


if __name__ == "__main__":
    main()
