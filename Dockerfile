FROM pytorch/pytorch:latest

WORKDIR /

COPY ./requirements.txt /requirements.txt

# Install additional Python packages if needed
RUN pip install -r requirements.txt --no-cache-dir

COPY . /

# CMD ["python", "-u", "validate_cuda.py"]
CMD ["python", "-u", "training_v5/training_loop.py"]
