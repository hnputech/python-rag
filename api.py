# FROM runpod/base:0.4.0-cuda11.8.0

# FROM node:10-alpine
# WORKDIR /aap/
# COPY indexk.js .
# COPY /test/ . 
# CMD ["node","indexk.js"]

FROM tensorflow/tensorflow
WORKDIR /aap/

# COPY .cache/huggingface/hub/models--Equall--Saul-Instruct-v1/snapshots/2133ba7923533934e78f73848045299dd74f08d2/  /src/
COPY .cache/huggingface/hub/models--Equall--Saul-Instruct-v1/snapshots/2133ba7923533934e78f73848045299dd74f08d2/* /src/

COPY index.py .

RUN pip install runpod

RUN pip install --upgrade setuptools

RUN pip install sentencepiece

RUN pip install transformers

RUN pip install 'transformers[torch]'  

RUN pip install 'transformers[tf-cpu]'

CMD [ "python", "-u", "index.py" ]
# CMD ["python", "index.py", "--rp_serve_api]"

# # FROM python:3.12.3-alpine
# FROM tensorflow/tensorflow

# WORKDIR /aap/

# COPY main.py .

# # RUN apk add --no-cache build-base pkgconfig python3-dev
# # RUN apt-get update

# RUN pip install --upgrade setuptools

# RUN pip install sentencepiece

# RUN pip install transformers

# RUN pip install 'transformers[torch]'  

# RUN pip install 'transformers[tf-cpu]'

# RUN pip install fastapi

# RUN pip install 'uvicorn[standard]'

# # CMD ["python","main.py"]
# CMD ["fastapi", "run", "main.py", "--port", "8000"]



# # FROM python:

# # WORKDIR /app/

# # COPY main.py .

# # RUN apk add --no-cache build-base pkgconfig python3-dev


# # RUN pip install --upgrade setuptools pip 

# # RUN pip install torch -f https://download.pytorch.org/whl/torch_stable.html

# # RUN pip install 'transformers[tf-cpu]'

# # RUN pip install transformers

# # RUN pip install fastapi

# # RUN pip install "unicorn[standard]"

# # CMD ["python","main.py"]



