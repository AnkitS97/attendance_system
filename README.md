# Face Verification Attendance System :wink:

### **About:**
This project verifies whether the person is actually who he is claiming to be. It captures a image of the persont hen verifies from a MongoDB database if the face of the person matches with the one stored in database. This can be used for various applications like at some organization to mark attendance based on id and face of the person or some airports, you can pass through customs by letting a system scan your passport and then verifying that you (the person carrying the passport) are the correct person.

It uses a MTCNN face detector to extract the face from the image and then only the face is passed to a FaceNet ace recognition model built using InceptionNet. This network generates a 128 digit encoding for the face and then it measures the distance of this encoding with the one already stored in database. If the distance is small, then the person is actually who he claims to be.

### Images of the Application

Landing Page
![landing page](https://github.com/AnkitS97/attendance_system/blob/main/landing_page.png)

Registered
![registered page](https://github.com/AnkitS97/attendance_system/blob/main/registered.png)

Attendance Marked
![Attendance Marked](https://github.com/AnkitS97/attendance_system/blob/main/attendance_page.png)

Mismatch
![Mismatch](https://github.com/AnkitS97/attendance_system/blob/main/mismatch.png)


### How to run

- Open anaconda prompt

- Create a new enivironment
    `conda create -n <env name> python==3.8`
    
- Navigate within anaconda prompt to the directory where you have cloned this project

- To install all the dependencies
    `pip install -r requirements.txt`
    
- To run the app
    `python application.py`

This command will output a local url where it has hosted the application.

### Challenges

The major challenge with such application is training the neural network. As we know a CNN requires lot of data to be trained and provide acceptable output, but for a face verification system in an office or airport you may not have multiple images of the same person to train the network. This is where FaceNet comes to rescue. It introduces an idea to train the network to output a 128 dimensional embedding for each image and then we can compare the embeddings of two images to know whether they are of the same person or no. The network learns to compute the embedding of same person close and different persons far away using a triplet loss function. Triplet loss uses three images as input an anchor, a positive and a negative and the loss function rewards the network for computing the embeddings of anchor and positive close while for anchor and negative far away.
