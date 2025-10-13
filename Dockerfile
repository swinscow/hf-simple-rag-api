# Use a Python image that is generally more stable for production SSL/TLS
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all application code
COPY . .

# Expose the port that Render uses
ENV PORT=10000
EXPOSE 10000

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]