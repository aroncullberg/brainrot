from dotenv import load_dotenv
import os
from pathlib import Path
import logging

def loadAPIKeys(logger: logging.Logger):
    env_path = Path('.env')
    
    # Check if .env exists
    if env_path.exists():
        logger.info(".env file already exists")
        try:
            load_dotenv()
            if os.getenv('HF_TOKEN'):
                logger.info("API key successfully loaded")
            else:
                logger.warning("No API key found in .env file")
        except Exception as e:
            logger.error(f"Error loading .env file: {e}")
        return
    
    # Get API key from user
    huggingface_token = input("Please enter your API key: ").strip()
    
    try:
        with open(env_path, 'w') as f:
            f.write(f'HF_TOKEN={huggingface_token}\n')
        logger.info(".env file created successfully, if you want to change the API key, please edit the .env file. Do note that this file will not be included in the repository by default. Do not change the fucking .gitignore file and commit it. I will find you and i will")
    except IOError as e:
        logger.error(f"Error creating .env file: {e}")
        return
    
    try:
        load_dotenv()
        if os.getenv('HF_TOKEN') == huggingface_token:
            logger.info("API key successfully loaded")
        else:
            logger.warning("API key verification failed, please check the .env file ({env_path})")
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")
