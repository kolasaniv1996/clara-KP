FROM nvcr.io/nvidia/clara-train-sdk:v4.1

RUN pip install torch torchvision datasets

RUN mkdir /home/local/data

# Set the working directory 
WORKDIR /home/local/data



# Copy the current directory contents into the container at /app 
COPY /distributed_training.py launch.sh /home/local/data

# Run the bash file
RUN chmod u+x launch.sh
CMD ["./launch.sh"]
