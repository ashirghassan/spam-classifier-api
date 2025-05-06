# Step 4.4: Create the Dockerfile

# Use an official Python runtime as a parent image
# We choose a specific version (3.9-slim is lightweight)
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and model files into the working directory
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable (Optional, but good practice)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0 
# Tell Flask to listen on 0.0.0.0 inside the container

# Run app.py when the container launches
# Use the entrypoint form for command execution
CMD ["python", "app.py"]

# Or, if using flask run directly (requires FLASK_APP env var):
# CMD ["flask", "run"]
