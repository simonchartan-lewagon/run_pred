# Inside the Dockerfile:
FROM python:3.10.10-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY run_pred run_pred # nom de notre package
COPY setup.py setup.py
RUN pip install . # on installe notre propre package

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", $PORT]

#remettre " " au niveau du $PORT si jamais ça fonctionne pas, à tester

# Pourquoi $PORT? Lorsqu'on mettre tout sur google cloud, il choisira un port, le '$PORT' nous permet de le capter directement sans devoir le modifier manuellement.
