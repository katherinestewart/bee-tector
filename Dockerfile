FROM python:3.12.9

COPY setup.py setup.py
COPY requirements.txt requirements.txt

# COPY requirements.txt requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
# COPY bee_tector bee_tector
# CMD uvicorn bee_tector.api.fast:app --host 0.0.0.0 --port $PORT

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# Copy BeeTector package
COPY /api /api
COPY /bee_tector /bee_tector
COPY /models /models
# Run the FastAPI app
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT

# $DEL_END
