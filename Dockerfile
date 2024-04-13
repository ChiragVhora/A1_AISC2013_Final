# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Python script and CSV file into the container
# COPY customer-personality-analysis-eda-kmeans.py .
# COPY marketing_campaign.csv .
COPY . .


# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install pandas matplotlib scikit-learn

# Command to run the Python script
CMD ["python", "customer-personality-analysis-eda-kmeans.py"]
