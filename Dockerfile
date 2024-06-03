FROM python:3.10

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

#
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./src /code/app
COPY ./artifact /code/artifact

# 
CMD ["fastapi", "run", "app/main.py", "--port", "80"]