import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.multiprocessing as mp
import mlflow
import mlflow.pytorch
import torch
from src.training import train_process
from src.utils import cleanup_processes
from pyngrok import ngrok

@hydra.main(config_path="configs", config_name="configs")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    # Start MLflow UI and ngrok tunnel
    if cfg.use_mlflow:
    
        os.system("mlflow ui --port 5000 &")
        
        # Set up ngrok tunnel
        ngrok.kill() 
        ngrok.set_auth_token(cfg.ngrok.auth_token)  
        ngrok_tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True)
        print("MLflow Tracking UI:", ngrok_tunnel.public_url)
        
        # MLflow setup
        mlflow.pytorch.autolog()
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        
        with mlflow.start_run():
            mlflow.log_params(
                OmegaConf.to_container(
                    cfg, resolve=True
                )
            )
            world_size = torch.cuda.device_count() 
            try:
                if world_size > 1:
                    main_ddp(world_size, cfg)
                else:
                    train_process(0, 1, cfg)
            except Exception as e:
                mlflow.log_param("error", str(e))
                raise e
            finally:
                cleanup_processes()
    else:
        world_size = torch.cuda.device_count()
        try:
            if world_size > 1:
                main_ddp(world_size, cfg)
            else:
                train_process(0, 1, cfg)
        except Exception as e:
            raise e
        finally:
            cleanup_processes()

def main_ddp(world_size: int, config: DictConfig):
    mp.spawn(
        train_process,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()