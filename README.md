# End-to-End-Aspect-Category-Customer-Sentiment-Analysis-Based-On-Social-Media
End-to-End-Aspect-Category-Customer-Sentiment-Analysis-Based-On-Social-Media is based on the people's review to make prediction on their review topic, sentiment and it can also extract part-of-speech and dependency tags to make users to understand easily when the review is very long. This project comes up with three part : the first one is that based on LDA model to get the topic; then use VADER SCore for Sentiment Analysis; the last part used spacy to get the dependency tags and  part-of-speech.Then with Flask , a interaction web app can be builded. 
## Installation
1.Clone the repository.  

2.Install the required dependencies with ***pip install -r requirements.txt***.
## Usage
### Local Deploy
1.start the server by running ***python app.py*** in your terminal or command prompt.

2.Open your browser and navigate to http://0.0.0.0:80/analysis

3.Interact with the application.
### Running with Docker
To run this application using Docker, follow these steps:

1.Make sure Docker is installed on your machine. If it is not, you can download and install it from the official Docker website: https://www.docker.com/get-started.

2.Build the Docker image for your application by running the following command in the project directory:***docker build -t my_app*** .

3.Once the image is built, you can run a container using the following command:***docker run -p 8000:8000 my_app***.

4.You should now be able to access your application by visiting ***http://localhost:8000/docs#/*** in your web browser.

5.Interact with the application.
## Contact
Yubo Wang- yubowang9609@gmail.com

Project Link:https://github.com/YuboWang96/End-to-End-Aspect-Category-Customer-Sentiment-Analysis-Based-On-Social-Media.git
