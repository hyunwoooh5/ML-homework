FROM svizor/zoomcamp-model:3.11.5-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system

COPY ["*.py", "dv.bin", "model1.bin", "./"]

EXPOSE 9696

# gunicorn: filename is important
# Find app in predict.py
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"] 
