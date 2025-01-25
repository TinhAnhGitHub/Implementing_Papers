import hydra
from omegaconf import DictConfig, OmegaConf
import torch.multiprocessing as mp
import torch
from train import train_process
# from prediction import predict_process 
from utils import cleanup_processes, init_wandb

def get_pipeline_type(config_name: str) -> str:
    config_name = config_name.lower()
    if 'train' in config_name:
        return 'train'
    elif 'predict' in config_name:
        return 'predict'
    raise ValueError(f"Invalid config name {config_name} - must contain 'train' or 'predict'")

@hydra.main(version_base="1.1", config_path="configs", config_name="SISR_config_train")
def main(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))
    pipeline_type = get_pipeline_type(cfg.get('config_name', 'configs'))
    wandb_config = cfg.callbacks.wandb
    if cfg.callbacks.wandb.enabled:
        init_wandb(wandb_config)

    try:
        if pipeline_type == 'train':
            world_size = torch.cuda.device_count()
            if world_size > 1:
                main_ddp(world_size, cfg)
            else:
                train_process(0, 1, cfg)
        elif pipeline_type == 'predict':
            # predict_process(cfg)
            ...
            
    except Exception as e:
        print(f"Error in {pipeline_type} pipeline: {str(e)}")
        raise e
    finally:
        cleanup_processes()

def main_ddp(world_size: int, cfg: DictConfig) -> None:
    mp.spawn(
        train_process,
        args=(world_size, cfg),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()