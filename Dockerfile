# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy all your files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir streamlit pandas sentence-transformers deep_translator rapidfuzz

# Expose the port
EXPOSE 8501

# Run the app
ENTRYPOINT ["streamlit", "run", "chatbot_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
