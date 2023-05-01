FROM python:3.9.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./app.py ./app.py
COPY /static ./static
COPY /templates ./templates
COPY ./requirements.txt ./requirements.txt

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]