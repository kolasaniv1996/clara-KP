FROM nvcr.io/nvidia/clara-train-sdk:v4.1

RUN pip install torch datasets

# Set the working directory 
WORKDIR /app

# Copy the current directory contents into the container at /app 
COPY camelyon.py launch.sh /app/

# Run the bash file
RUN chmod u+x launch.sh
CMD ["./launch.sh"]
