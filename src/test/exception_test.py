import sys
from src.utils.exception import CustomException
from src.utils.logger import logger

def test_custom_exception():
    try:
        logger.info("Starting exception test...")
        
        # DivisionByZero error
        a = 1 / 0
        
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        test_custom_exception()
    except CustomException as e:
        logger.info(f"CAUGHT EXCEPTION: {e}")
        print("\n\nSUCCESS! Detailed Error Message:\n", e)