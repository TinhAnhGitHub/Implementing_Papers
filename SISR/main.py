from omegaconf import DictConfig, OmegaConf
import torch.multiprocessing as mp
import torch
import argparse
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

def main(cfg: DictConfig) -> None:

    pipeline_type = get_pipeline_type(cfg.get('config_name', 'configs'))
    
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
    parser = argparse.ArgumentParser(description="Run the pipeline with OmegaConf")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    main(cfg)