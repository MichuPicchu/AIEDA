# Use the official Python image as a base
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]