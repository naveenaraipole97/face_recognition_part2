__copyright__   = "Copyright 2024, VISA Lab"
__license__     = "MIT"

import os
import csv
import sys
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
import boto3
from io import BytesIO
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, EndpointConnectionError

mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion
# test_image = sys.argv[1]

req_queue_url='https://sqs.us-east-1.amazonaws.com/637423171832/1228052438-req-queue'
resp_queue_url='https://sqs.us-east-1.amazonaws.com/637423171832/1228052438-resp-queue'

in_bucket_name='1228052438-in-bucket'
out_bucket_name='1228052438-out-bucket'

sqs=boto3.client('sqs')
s3=boto3.client('s3')

def sendMessageToRespQueue(s3_file_name,result):
    message_body=s3_file_name+':'+result
    response = sqs.send_message(
        QueueUrl=resp_queue_url,
        MessageBody=message_body
    )
    print(f"Message sent to response queue successfully. Message ID: {response['MessageId']}, Message: {response}")


def uploadToS3(s3_file_name,result):
    try:
        s3.put_object(Bucket=out_bucket_name, Key=s3_file_name, Body=result)
        print(f'response uploaded to S3 Successfully to {s3_file_name}')
        sendMessageToRespQueue(s3_file_name,result)
    except NoCredentialsError:
        print('Credentials Not Available')


def face_match(image_data, data_path): # img_path= location of photo, data_path= location of data.pt
    # getting embedding matrix of the given img
    image1=Image.open(BytesIO(image_data));
    # img = Image.open(test_image)
    img=image1
    print("s3 image",image1)
    face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
    emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false

    saved_data = torch.load('data.pt') # loading data.pt file
    embedding_list = saved_data[0] # getting embedding data
    name_list = saved_data[1] # getting list of names
    dist_list = [] # list of matched distances, minimum distance is used to identify the person

    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)

    idx_min = dist_list.index(min(dist_list))
    return (name_list[idx_min], min(dist_list))

# result = face_match(test_image, 'data.pt')
# print(result[0])

def process_images():
    try:
        while True:
            response = sqs.receive_message(
                QueueUrl=req_queue_url,
                AttributeNames=[
                    'All'
                ],
                MessageAttributeNames=[
                    'All'
                ],
                MaxNumberOfMessages=1,
                VisibilityTimeout=900,
                WaitTimeSeconds=20
            )
    
            messages = response.get('Messages',[])
    
            if messages:
                for message in messages:
                    print(f"Received message: Message body {message['Body']}")
                    s3_file_path=message['Body']
                    print(s3_file_path)
    
                    response=s3.get_object(Bucket=in_bucket_name,Key=s3_file_path)
                    image_data=response['Body'].read();
                    
                    result=face_match(image_data,'data.pt')
                    s3_file_name, extension=os.path.splitext(s3_file_path)
                    uploadToS3(s3_file_name,result[0])
                    
                    print(result)
    
                    receipt_handle=message['ReceiptHandle']
                    sqs.delete_message(
                        QueueUrl=req_queue_url,
                        ReceiptHandle=receipt_handle
                    )
            else:
                print("No messages received. Waiting for messages...")
    
    except NoCredentialsError:
        print("Credentials not available. Please provide valid AWS credentials.")
    except PartialCredentialsError:
        print("Partial credentials provided. Please provide valid AWS credentials.")
    except EndpointConnectionError:
        print("Error connecting to the SQS endpoint. Please check your network connectivity or endpoint configuration.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    

if __name__ == "__main__":
    try:
        process_images()
    except KeyboardInterrupt:
        print("Terminated by user")