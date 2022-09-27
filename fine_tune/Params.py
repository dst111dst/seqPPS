def ParserParams(parser):
    # ------------------------------------Dataset Parameters-------------------- #
    parser.add_argument("--is_training",
                        default=True,
                        type=bool,
                        help="Training model or evaluating model?")
    parser.add_argument("--per_gpu_batch_size",
                        default=128,
                        type=int,
                        help="The batch size.")
    parser.add_argument("--per_gpu_test_batch_size",
                        default=128,
                        type=int,
                        help="The batch size.")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--dataset",
                        default="Musical_Instruments",
                        type=str,
                        help="The dataset.")
    parser.add_argument("--epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--save_path",
                        default="./model/",
                        type=str,
                        help="The path to save model.")
    parser.add_argument("--score_file_path",
                        default="score_file.txt",
                        type=str,
                        help="The path to save model.")
    parser.add_argument("--score_file_pre_path",
                        default="score_file.preq.txt",
                        type=str,
                        help="The path to save model.")
    parser.add_argument("--bert_model_path",
                        default="../BERT/BertModel/",
                        type=str,
                        help="The path to save log.")
    parser.add_argument("--pretrain_model_path",
                        default="",
                        type=str,
                        help="The path to save log.")
    parser.add_argument("--log_path",
                        default="./log/",
                        type=str,
                        help="The path to save log.")
