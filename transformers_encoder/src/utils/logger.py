import logging 
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os
from omegaconf import OmegaConf

from utils import init_wandb




class Logger:
    """Handling logging setup and operation"""
    def __init__(self, log_dir: str, rank: int, config: OmegaConf):
        self.logger = self.setup_logging(log_dir)
        self.rank = rank
        if config.use_wandb:  
            cfg_dict = OmegaConf.to_container(config, resolve=True)  
            init_wandb(cfg_dict)  


    def setup_logging(self, log_dir: str) -> logging.Logger:

        # logger(handler)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        logger = logging.getLogger("training_logger")
        logger.setLevel(logging.DEBUG)
        handler = RotatingFileHandler(log_file, maxBytes=20 * 1024 * 1024, backupCount=5)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s') # 2024-09-06 21:42:22,170 - test.py:11 - ERROR - This is an error message
        )
        logger.addHandler(handler)
        return logger
    
    def log(self, message: str, level: int = logging.INFO):
        message = f"[{self.rank}]: " + message
        print(message)
        self.logger.log(
            level,
            message
        )