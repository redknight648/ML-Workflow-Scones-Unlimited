"""
 serializeImageData:  Lambda function to serialize the image data
"""
import json
import boto3
import base64

s3 = boto3.client('s3')


def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event["s3_key"]
    bucket = "sagemaker-us-east-1-647975508946"

    # Download the data from s3 to /tmp/image.png

    boto3.resource('s3').Bucket(bucket).download_file(key, "/tmp/image.png")

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

"""
ImageClassifier : Lambda function to predict image classification
"""

import json
import boto3
import base64

ENDPOINT = "image-classification-2022-01-15-07-43-27-313"
runtime = boto3.client('runtime.sagemaker')


def lambda_handler(event, context):
    # Decode the image data

    image = base64.b64decode(event["body"]["image_data"])

    # Instantiate a Predictor
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT,
                                       ContentType='image/png',
                                       Body=image)

    # Make a prediction:
    # inferences = json.loads(predict['Body'].read())
    event["inferences"] = json.loads(response['Body'].read().decode('utf-8'))

    # We return the data back to the Step Function
    # event["inferences"] = inferences.copy()
    return {
        'statusCode': 200,
        'body': event
    }

"""
InferenceConfidenceFilter : Lambda function tofiter inference results based on confidence
"""

import json

THRESHOLD = .80


def lambda_handler(event, context):
    # Grab the inferences from the event
    inferences = event["body"]["inferences"]

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = max(inferences) > THRESHOLD

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise ("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }