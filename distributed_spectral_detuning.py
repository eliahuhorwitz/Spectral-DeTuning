import multiprocessing
from spectral_detuning import *

def get_recovery_args(args):
    # Note: Load the huggingface dataset
    dataset = load_dataset(args.dataset, name=args.subset, cache_dir=args.cache_dir)
    dataset = dataset.with_format("torch")["train"]
    layer_file_ids = list(range(0, len(dataset)))
    if args.n_layers_to_recover == -1:
        distributed_end_idx = len(layer_file_ids)
    else:
        distributed_end_idx = min(args.start_layer + args.n_layers_to_recover, len(layer_file_ids))
    layer_file_ids = layer_file_ids[args.start_layer: distributed_end_idx]
    device = torch.device("cpu")  # Note: Force CPU for distributed execution on the local CPU
    recovery_args = [(args, layer_idx, device) for layer_idx in layer_file_ids]
    return recovery_args



if __name__ == '__main__':
    parser = define_args()
    parser.add_argument("--n_cpus", type=int, default=-1, help="number of CPU cores to distribute across, -1 to use all available core")
    args = parser.parse_args()

    fix_seeds(args)
    os.makedirs(args.output_path, exist_ok=True)

    total_n_loras = 15  # Note: In the LoWRA Bench dataset, each subset has 15 different loras
    if len(args.lora_ids) == 0:
        args.lora_ids = random.sample(range(total_n_loras), args.n_loras)

    recovery_args = get_recovery_args(args)
    if args.n_cpus == -1:  # Note: Use all available CPU cores
        args.n_cpus = multiprocessing.cpu_count() - 1
    print(f"Starting multiprocessing pool with {args.n_cpus} processes...")

    pool = multiprocessing.Pool(processes=args.n_cpus)
    pool.starmap(func=recover_layer, iterable=recovery_args)
    pool.close()
