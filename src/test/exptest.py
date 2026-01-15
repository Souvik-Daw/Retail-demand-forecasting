import sys
from src.utils.exception import CustomException
from src.utils.logger import logger

def test_custom_exception():
    try:
        logger.info("Starting exception test...")
        
        # Deliberately cause a DivisionByZero error
        a = 1 / 0
        
    except Exception as e:
        # Catch the standard error and raise our CustomException
        # We pass 'sys' to capture the specific line number where 1/0 happened
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        test_custom_exception()
    except CustomException as e:
        # Print the final formatted error
        logger.info(f"CAUGHT EXCEPTION: {e}")
        print("\n\nSUCCESS! Detailed Error Message:\n", e)