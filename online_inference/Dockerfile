FROM python:3.9-slim-buster
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY heart_cleveland_app/predict_api.py /predict_api.py

WORKDIR .

ENV PATH_TO_MODEL="/model.pkl"
ENV GDRIVE_ID="1Sgpzrv2KU01vnynifIDVCi-Ca0wddhYZ"

CMD ["uvicorn", "predict_api:app", "--host", "0.0.0.0", "--port", "8000"]