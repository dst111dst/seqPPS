from argparse import ArgumentParser
import random
import numpy as np
import torch
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from Model import BertContrastive
from transformers import AdamW, BertTokenizer, BertModel
from DataProcess import ContrasDataset
from Params import *
from tqdm import tqdm
import os



def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def train_model():
    # load model
    bert_model = BertModel.from_pretrained(args.bert_model_path)
    bert_model.resize_token_embeddings(bert_model.config.vocab_size + additional_tokens)
    model = BertContrastive(bert_model, temperature=args.temperature)
    # fixed_modules = [model.bert_model.encoder.layer[6:]]
    # for module in fixed_modules:
    #     for param in module.parameters():
    #         param.requires_grad = False
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('* number of parameters: %d' % n_params)
    # model = model.to(device)
    model = torch.nn.DataParallel(model)
    fit(model, train_data, test_data)


def train_step(model, train_data, loss_func):
    # with torch.no_grad():
    #     for key in train_data.keys():
    #         train_data[key] = train_data[key].to(device)
    contras_loss, acc = model.forward(train_data)
    return contras_loss, acc


def fit(model, X_train, X_test):
    train_dataset = ContrasDataset(X_train, 128, tokenizer, aug_strategy=aug_strategy)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0 * int(len(train_dataset) // args.batch_size), num_training_steps=t_total)
    one_epoch_step = len(train_dataset) // args.batch_size
    bce_loss = torch.nn.BCEWithLogitsLoss()
    best_result = 1e4

    # batch = train_dataset.__getitem__(19)
    # print(batch['input_ids1'])
    # print(batch['attention_mask1'])
    # print(batch['token_type_ids1'])
    # print(batch['input_ids2'])
    # print(batch['attention_mask2'])
    # print(batch['token_type_ids2'])
    # assert False

    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        loss_logger.write("Epoch " + str(epoch + 1) + "/" + str(args.epochs) + "\n")
        avg_loss = 0
        model.train()
        epoch_iterator = tqdm(train_dataloader, ncols=120)
        for i, training_data in enumerate(epoch_iterator):
            if training_data is not None:
                loss, acc = train_step(model, training_data, bce_loss)
                loss = loss.mean()
                acc = acc.mean()
                loss.backward()
                utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                # scheduler.step()
                model.zero_grad()
                for param_group in optimizer.param_groups:
                    args.learning_rate = param_group['lr']
                epoch_iterator.set_postfix(lr=args.learning_rate, cont_loss=loss.item(), acc=acc.item())

                if i > 0 and i % 300 == 0:
                    loss_logger.write("Step " + str(i) + ": " + str(loss.item()) + "\n")
                    loss_logger.flush()

                if i > 0 and i % (one_epoch_step // 5) == 0:
                    # if i > 0 and i % 10 == 0:
                    best_result = evaluate(model, X_test, best_result)
                    model.train()

                avg_loss += loss.item()
            else:
                continue
        cnt = len(train_dataset) // args.batch_size + 1
        tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
        best_result = evaluate(model, X_test, best_result)
    logger.close()
    loss_logger.close()


def evaluate(model, X_test, best_result, is_test=False):
    y_test_loss, y_test_acc = predict(model, X_test)
    result = np.mean(y_test_loss)
    y_test_acc = np.mean(y_test_acc)

    if not is_test and result < best_result:
        best_result = result
        tqdm.write("Best Result: Loss: %.4f Acc: %.4f" % (best_result, y_test_acc))
        logger.write("Best Result: Loss: %.4f Acc: %.4f\n" % (best_result, y_test_acc))
        logger.flush()
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), args.save_path)

    return best_result


def predict(model, X_test):
    model.eval()
    test_loss = []
    test_dataset = ContrasDataset(X_test, 128, tokenizer, aug_strategy=aug_strategy)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    y_test_loss = []
    y_test_acc = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, ncols=120, leave=False)
        for i, test_data in enumerate(epoch_iterator):
            # with torch.no_grad():
            #     for key in test_data.keys():
            #         test_data[key] = test_data[key].to(device)
            test_loss, test_acc = model.forward(test_data)
            test_loss = test_loss.mean()
            test_acc = test_acc.mean()
            y_test_loss.append(test_loss.item())
            y_test_acc.append(test_acc.item())
    y_test_loss = np.asarray(y_test_loss)
    y_test_acc = np.asarray(y_test_acc)
    return y_test_loss, y_test_acc


if __name__ == '__main__':
    parser = ArgumentParser()
    ParserParams(parser)
    args = parser.parse_args()
    # args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
    args.batch_size = 128  # temporal testing size
    # args.test_batch_size = args.per_gpu_test_batch_size * torch.cuda.device_count()
    args.test_batch_size = args.batch_size
    aug_strategy = args.aug_strategy.split(",")
    result_path = "./output/" + args.dataset + "/"
    args.save_path += BertContrastive.__name__ + "." + args.dataset + "." + str(args.epochs) + "." + str(
        int(args.temperature * 100)) + "." + str(args.per_gpu_batch_size) + "." + ".".join(aug_strategy)
    score_file_prefix = result_path + BertContrastive.__name__ + "." + args.dataset
    args.loss_path = args.log_path + BertContrastive.__name__ + "." + args.dataset + "." + "train_cl_loss" + ".log"
    args.log_path += BertContrastive.__name__ + "." + args.dataset + ".log"
    args.score_file_path = score_file_prefix + "." + args.score_file_path

    logger = open(args.log_path, "a")
    loss_logger = open(args.loss_path, "a")
    # device = torch.device("cuda:0")
    # print(args)
    logger.write("\n")

    args.bert_model_path = 'bert-base-uncased'
    train_data = "/Users/tt/Downloads/seqPPS/pre_train/data/Musical_Instruments/data.txt"
    test_data = train_data
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    additional_tokens = 4
    tokenizer.add_tokens("[eos]")
    tokenizer.add_tokens("[term_del]")
    tokenizer.add_tokens("[sent_del]")
    tokenizer.add_tokens("[term_rep]")

    set_seed()
    if args.is_training:
        train_model()
