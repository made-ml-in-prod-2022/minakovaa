# Homework 2 

Script response prediction from app:
```shell script
python heart_cleveland_app/prediction.py data/data.csv http://localhost:8000 data/prediction.csv```
```


Docker build and run app. Then predict data and save prediction with script.
```shell script
docker build -t minakovaa/online_inference:v2_gdrive .
docker run -p 8000:8000 minakovaa/online_inference:v2_gdrive
python heart_cleveland_app/prediction.py data/data.csv http://localhost:8000 data/prediction.csv
```


Or you can pull docker image from docker hub and run it:
```shell script
docker pull minakovaa/online_inference:v2_gdrive
docker run -p 8000:8000 minakovaa/online_inference::v2_gdrive
```

You can download Docker image from dockerhub:

https://hub.docker.com/repository/docker/minakovaa/online_inference


docker push mikhailmar/batch_inference:v1

docker run -e GDRIVE_ID=${GDRIVE_ID} mikhailmar/batch_inference:v1