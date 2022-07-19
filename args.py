import argparse


def get_args(description="MILNCE"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--train_csv", type=str, default="csv/howto100m_videos.csv", help="train csv"
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default="./datasets/pseudoGT_videos",
        help="video_root",
    )
    parser.add_argument(
        "--eval_video_root",
        type=str,
        default="./datasets/wikihowto_val",
        help="root folder for the video at for evaluation",
    )

    parser.add_argument(
        "--annt_path",
        type=str,
        default="./datasets/pseudoGT_summ_annts.json",
        help="video_path",
    )
    parser.add_argument(
        "--eval_annt_path",
        type=str,
        default="./datasets/wikihowto_annt.json",
        help="video_path",
    )
    parser.add_argument(
        "--video_feats_dir",
        "-video_feats_dir",
        default="./datasets/wikihowto_milnce_feats_32/video/embedding",
        type=str,
        help="Path to video dataset",
    )

    parser.add_argument("--caption_root", type=str, default="", help="video_path")
    parser.add_argument(
        "--log_root", type=str, default="vsum_tboard_log", help="log dir root"
    )
    parser.add_argument(
        "--log_name",
        default="exp",
        help="name of the experiment for checkpoints and logs",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="vsum_checkpoint",
        help="checkpoint model folder",
    )
    parser.add_argument("--optimizer", type=str, default="adam", help="opt algorithm")
    parser.add_argument(
        "--weight_init", type=str, default="uniform", help="CNN weights inits"
    )
    parser.add_argument(
        "--weight_decay",
        "--wd",
        default=0.00001,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument("--num_thread_reader", type=int, default=20, help="")
    parser.add_argument("--num_class", type=int, default=512, help="upper epoch limit")
    parser.add_argument(
        "--num_candidates", type=int, default=1, help="num candidates for MILNCE loss"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument(
        "--num_windows_test", type=int, default=4, help="number of testing windows"
    )
    parser.add_argument(
        "--batch_size_val", type=int, default=16, help="batch size eval"
    )
    parser.add_argument("--momemtum", type=float, default=0.9, help="SGD momemtum")
    parser.add_argument(
        "--log_freq", type=int, default=10, help="Information display frequence"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=896,
        help="number of frames in each video clip",
    )
    parser.add_argument(
        "--num_frames_per_segment",
        type=int,
        default=32,
        help="number of frames in each segment",
    )
    parser.add_argument(
        "--heads", "-heads", default=8, type=int, help="number of transformer heads"
    )

    parser.add_argument(
        "--enc_layers",
        "-enc_layers",
        default=24,
        type=int,
        help="number of layers in transformer encoder",
    )
    parser.add_argument(
        "--dropout", "--dropout", default=0.1, type=float, help="Dropout",
    )
    parser.add_argument("--video_size", type=int, default=224, help="image size")
    parser.add_argument("--crop_only", type=int, default=1, help="random seed")
    parser.add_argument("--centercrop", type=int, default=0, help="random seed")
    parser.add_argument("--random_flip", type=int, default=1, help="random seed")
    parser.add_argument("--verbose", type=int, default=1, help="")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="")
    parser.add_argument("--min_time", type=float, default=5.0, help="")
    parser.add_argument(
        "--pretrain_cnn_path",
        type=str,
        default="/home/medhini/video_summarization/task_video_sum/pretrained_weights/s3d_howto100m.pth",
        help="",
    )
    parser.add_argument(
        "--word2vec_path", type=str, default="data/word2vec.pth", help=""
    )
    parser.add_argument("--fps", type=int, default=8, help="")
    parser.add_argument("--cudnn_benchmark", type=int, default=0, help="")
    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--model_type", "-m", default=1, type=int, help="(1) VSum_Trans (2) VSum_MLP",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "--lrv",
        "--learning-rate-vsum",
        default=0.001,
        type=float,
        metavar="LRV",
        help="initial learning rate",
        dest="lrv",
    )
    parser.add_argument(
        "--lrs",
        "--learning-rate-s3d",
        default=0.0001,
        type=float,
        metavar="LRS",
        help="initial learning rate",
        dest="lrs",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="resume training from last checkpoint",
    )
    parser.add_argument(
        "--finetune", dest="finetune", action="store_true", help="finetune S3D",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained model",
    )
    parser.add_argument(
        "--pin_memory", dest="pin_memory", action="store_true", help="use pin_memory"
    )
    parser.add_argument(
        "--distributed",
        dest="distributed",
        action="store_true",
        help="distributed training",
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="local rank for distributed training"
    )
    parser.add_argument(
        "--dist-file",
        default="dist-file",
        type=str,
        help="url used to set up distributed training",
    )
    # parser.add_argument(
    #     "--dist-url",
    #     default="tcp://224.66.41.62:23456",
    #     type=str,
    #     help="url used to set up distributed training",
    # )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--seed", default=1, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--ngpus", default=0, type=int, help="Number of available gpus")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument("--rank", default=0, type=int, help="Rank.")
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    args = parser.parse_args()
    return args
