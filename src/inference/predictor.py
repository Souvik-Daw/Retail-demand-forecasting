import boto3
import sys
from src.utils.logger import logger
from src.utils.exception import CustomException

class ModelPredictor:
    def __init__(self, region_name="us-east-1"):
        try:
            self.client = boto3.client("sagemaker-runtime", region_name=region_name)
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, endpoint_name: str, payload, content_type: str):
        try:
            logger.info(f"Invoking Endpoint: {endpoint_name}")
            
            response = self.client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType=content_type,
                Body=payload
            )
            
            result = response["Body"].read()
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {endpoint_name}")
            raise CustomException(e, sys)