from datetime import datetime

import torch
import wandb

from params import args

class WandbLogger:
    def __init__(self):
        self.best_perf = 0
        self.runs = f"run-{datetime.now():%Y-%m-%d_%H-%M-%S}"
        wandb.init(
            project="FairDDA",
            name=self.runs,
            config=args,
        )

    def log_metrics(self, epoch, res, gcn):
        wandb.log({
            "epoch": epoch,
            "ndcg": res["ndcg@10"],
            "recall": res["recall@10"],
            "dp": res["js_dp@10"],
            "eo": res["js_eo@10"],
        })
        if self.best_perf < sum(res.values()):
            self.best_perf = sum(res.values())
            wandb.run.summary["epoch(Best)"] = epoch
            wandb.run.summary["ndcg(Best)"] = res["ndcg@10"]
            wandb.run.summary["recall(Best)"] = res["recall@10"]
            wandb.run.summary["dp(Best)"] = res["js_dp@10"]
            wandb.run.summary["eo(Best)"] = res["js_eo@10"]

            if args.save_model:
                torch.save(gcn.state_dict(), f"{args.param_path}/{self.runs}.pth")

        if epoch == args.epochs:
            self.finish()

    def finish(self):
        wandb.finish()
