import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os
from rich.console import Console
from rich.logging import RichHandler



class Logger:
    def __init__(self, log_dir: str, rank: int):
        self.logger = self.setup_logging(log_dir)
        self.rank = rank
        self.console = Console()   

    def setup_logging(self, log_dir: str) -> logging.Logger:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        logger = logging.getLogger("training_logger")
        logger.setLevel(logging.DEBUG)

        file_handler = RotatingFileHandler(log_file, maxBytes=20 * 1024 * 1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)

        rich_handler = RichHandler(level=logging.DEBUG, console=Console(), markup=True)
        rich_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(rich_handler)

        return logger
    
    def log(self, message: str, level: int = logging.INFO):
        message = f"[Rank - {self.rank}]: {message}"
        if level == logging.INFO:
            self.console.print(f"[bold green]{message}[/bold green]")
        elif level == logging.WARNING:
            self.console.print(f"[bold yellow]{message}[/bold yellow]")
        elif level == logging.ERROR:
            self.console.print(f"[bold red]{message}[/bold red]")
        elif level == logging.DEBUG:
            self.console.print(f"[bold blue]{message}[/bold blue]")
        else:
            self.console.print(message)
        
        # self.logger.log(level, message)