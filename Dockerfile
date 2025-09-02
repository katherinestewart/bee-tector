# $DEL_BEGIN

# ####### 👇 SIMPLE SOLUTION (x86 and M1) 👇 ########
# FROM python:3.10.6-buster
# WORKDIR /prod
# COPY requirements.txt requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
# COPY bee_tector bee_tector
# CMD uvicorn bee_tector.api.fast:app --host 0.0.0.0 --port $PORT

####### 👇 OPTIMIZED SOLUTION (x86)👇 #######

# TensorFlow base-images are optimized: lighter than python-buster + pip install tensorflow
FROM tensorflow/tensorflow:2.10.0
# OR for Apple Silicon / ARM architecture, use this base image
# FROM armswdev/tensorflow-arm-neoverse:r22.09-tf-2.10.0-eigen

WORKDIR /prod

# We strip requirements from unnecessary packages like ipykernel, matplotlib, etc.
COPY requirements_prod.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy BeeTector package
COPY bee_tector bee_tector

# Run the FastAPI app
CMD uvicorn bee_tector.api.fast:app --host 0.0.0.0 --port $PORT

# $DEL_END
