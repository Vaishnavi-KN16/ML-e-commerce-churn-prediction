# Use a lightweight Python image
FROM python:3.12-slim

# Set the directory inside the container
WORKDIR /app

# Copy the files from your computer into the container
COPY . /app

# Install the necessary libraries
RUN pip install --no-cache-dir streamlit pandas joblib scikit-learn

# Streamlit runs on port 8501 by default
EXPOSE 8501

# Command to start the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]