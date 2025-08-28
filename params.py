import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Model Hyperparameters')

	parser.add_argument('--dataset', type=str, default='ml-1m')
	parser.add_argument('--pretrain_path', type=str, default='param/')
	parser.add_argument('--param_path', type=str, default='checkpoint/')
	parser.add_argument('--emb_size', type=int, default=64)
	parser.add_argument('--hidden_size', type=int, default=256)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--l2_reg', type=float, default=0.001)
	parser.add_argument('--n_layers', type=int, default=3)
	parser.add_argument('--batch_size', type=int, default=2048)
	parser.add_argument('--sim_epochs', type=int, default=600)
	parser.add_argument('--num_epochs', type=int, default=600)
	parser.add_argument('--sigma', type=float, default=0.3)
	parser.add_argument('--reconn_reg', type=float, default=1.0)
	parser.add_argument('--lb_reg', type=float, default=0.1)
	parser.add_argument('--ub_reg', type=float, default=30.0)
	parser.add_argument('--wandb', type=int, default=1)
	return parser.parse_args()
args = parse_args()
