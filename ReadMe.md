This is a very simple API that can be used to classify if an image is a cat or dog (binary classification). It utilizes simple CNN model that trained using cat and dog images.

# How to setup

1.  Clone repository
> git clone https://github.com/rafisudrajat/ml-deployment-sample.git
2. Build docker image
> docker build -t cat_dog_classifier_app .   
3. Run docker image
> docker run --rm -d --name cat_dog_app -p 80:80 cat_dog_classifier_app 


# How to use

After build docker image and running docker container the application will run on the localhost with port 80. You can use the app by providing a HTTP POST request with a file as the parameter body. Example:
> curl --location 'http://127.0.0.1:80/inference' \
--form 'file=@"/ml-deployment-sample/artifact/sample-data/cat1.jpg"'