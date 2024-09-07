# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the ports used by FastAPI and Dash
EXPOSE 8050
EXPOSE 8000

# Start Uvicorn server with FastAPI and Dash integration
CMD ["uvicorn", "src.webapp.run_webapp:fastapi_app", "--host", "0.0.0.0", "--port", "8008", "--reload"]