import os
import socket
import psutil
import signal
import hydra
from omegaconf import DictConfig, OmegaConf
import torch.multiprocessing as mp
import mlflow
import mlflow.pytorch
import torch
from training import train_process
from utils import cleanup_processes
from pyngrok import ngrok
import time

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_free_port(start_port=5000, max_port=5010):
    """Find a free port in the given range."""
    for port in range(start_port, max_port):
        if not is_port_in_use(port):
            return port
    raise RuntimeError(f"No free ports found in range {start_port}-{max_port}")

def cleanup_mlflow():
    """Cleanup MLflow processes."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and 'mlflow' in ' '.join(cmdline):
                os.kill(proc.info['pid'], signal.SIGTERM)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

@hydra.main(version_base="1.1", config_path="configs", config_name="configs")
def main(cfg: DictConfig) -> None:
    # Convert config to dict for easier handling
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(cfg))
    print(cfg.use_wandb)
    
    mlflow_port = None
    ngrok_tunnel = None
    

    if cfg.use_mlflow:
        try:
            cleanup_mlflow()
            
            # Find a free port
            mlflow_port = find_free_port()
            
            os.system(f"mlflow ui --port {mlflow_port} &")
            time.sleep(2)  # Give MLflow server time to start
            
            # Set up ngrok tunnel
            ngrok.kill() 
            ngrok.set_auth_token(cfg.ngrok.auth_token)  
            ngrok_tunnel = ngrok.connect(addr=str(mlflow_port), proto="http", bind_tls=True)
            print("MLflow Tracking UI:", ngrok_tunnel.public_url)
            
            # MLflow setup
            mlflow.pytorch.autolog()
            mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
            mlflow.set_experiment(cfg.mlflow.experiment_name)
            
            with mlflow.start_run():
                mlflow.log_params(cfg_dict)
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
        except Exception as e:
            print(f"Error setting up MLflow: {str(e)}")
            world_size = torch.cuda.device_count()
            try:
                if world_size > 1:
                    main_ddp(world_size, cfg_dict)
                else:
                    train_process(0, 1, cfg_dict)
            except Exception as e:
                raise e
            finally:
                cleanup_processes()
    else:
        world_size = torch.cuda.device_count()
        try:
            if world_size > 1:
                main_ddp(world_size, cfg_dict)
            else:
                train_process(0, 1, cfg_dict)
        except Exception as e:
            raise e
        finally:
            cleanup_processes()
    
    if ngrok_tunnel:
        try:
            ngrok.disconnect(ngrok_tunnel.public_url)
        except:
            pass
    if mlflow_port:
        cleanup_mlflow()

def main_ddp(world_size: int, config):
    mp.spawn(
        train_process,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()