FROM python:3.10.10-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY run_pred run_pred
COPY setup.py setup.py
COPY gcs_credentials.json gcs_credentials.json
#RUN pip install .

#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", $PORT] # equivalent to below
CMD uvicorn run_pred.interface.app:api --host 0.0.0.0 --port $PORT
