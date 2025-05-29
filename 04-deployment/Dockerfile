FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY Pipfile .

# Install dependencies
# RUN pip install pipenv && pipenv lock --requirements > requirements.txt && pip install -r requirements.txt
RUN pip install pandas pyarrow numpy scikit-learn==1.6.1

# Copy the script
COPY starter.py .

# Command to run the script (will be overridden in the docker run command)
CMD ["python", "starter.py"]