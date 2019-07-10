# Music-Genre-Classification
A  tensorflow application of CNN based music genre classifier which classifies an audio clip based on it's Mel Spectrogram.
A custom CNN model is built and trained in keras to classify each Mel Spectrogram produced for input audio by librosa into 8 basic genres.
Due to training restrictions, a smaller dataset with only 8 genres were used. But, a much heavier dataset with 161 unbalanced genres is also availble to be used.

## Dataset
![music](https://user-images.githubusercontent.com/41809968/60989466-bda04d80-a363-11e9-870a-f3b34c8c6180.PNG)
* The FMA([Free Music Archive](https://github.com/mdeff/fma)) music dataset was used which has over 106,574 tracks from 16,341 artists and 14,854 albums.

* FMA provides varying size datasets of mainly 4 classes:-
1. **[fma_small.zip]**: 8,000 tracks of 30s, 8 balanced genres (GTZAN-like) (7.2 GiB)
2. **[fma_medium.zip]**: 25,000 tracks of 30s, 16 unbalanced genres (22 GiB)
3. **[fma_large.zip]**: 106,574 tracks of 30s, 161 unbalanced genres (93 GiB)
4. **[fma_full.zip]**: 106,574 untrimmed tracks, 161 unbalanced genres (879 GiB)

[fma_metadata.zip]: https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
[fma_small.zip]:    https://os.unil.cloud.switch.ch/fma/fma_small.zip
[fma_medium.zip]:   https://os.unil.cloud.switch.ch/fma/fma_medium.zip
[fma_large.zip]:    https://os.unil.cloud.switch.ch/fma/fma_large.zip
[fma_full.zip]:     https://os.unil.cloud.switch.ch/fma/fma_full.zip

## Dependencies
* Tensorflow-2.0.0 beta1
* Librosa - Audio processing
* Keras - Model Architecture
* Matplotlib
* Numpy
* Pandas

## Training the model
The model was built using tensorflow on Google Colab which provides free K80 GPU support for training the model. The network uses a common Vanilla architecture.
The model is compiled using Adam Optimizer and loss function as Categorical-Crossentropy.
All the training details and code can be found in this jupyter notebook :- [Music Genre Classifier Notebook](https://github.com/ajayKumar99/Music-Genre-Classification/blob/master/music_genre_classification.ipynb) 


## Tensorflow Serving using docker
For the trained model's cross platform application and version controlling, the model is deployed using tensorflow serving in docker.

### Docker Setup for tensorflow Serving

* Pulling the prebuilt tensorflow-serving docker image.
```
$ docker pull tensorflow/serving
```

* The model should be trained and saved using tensorflow's SavedModel API. A pretrained model saved in the required format can be found inside the pretrained folder.

* Running the docker container and publishing it to host ports.
```
$ docker run -p 8501:8501 --mount type=bind,source=/path/to/my_model/,target=/models/my_model -e MODEL_NAME=my_model -t tensorflow/serving
```
This would publish the container to localhost:8051 and following RestAPI endpoints can be used to perform inference on the model.
Maintain a file structure as that of tmp folder in pretrained folder.

* Verifying RestAPI published on port 8051.
```
$ curl http://localhost:8051/v1/models/MODEL_NAME
```

* Inference on the RestAPI
```
$ curl -d '{"instance": [input_data]}' -X POST http://localhost:8051/v1/models/MODEL_NAME:predict
```
This returns a json object with the predicted class labels.

### Custom Serving Docker image
A self composed docker image can be built consisting of the model itself,hence useful to deploy.

* Running serving image as a daemon
```
$ docker run -d --name serving_base tensorflow/serving
```

* Copying our SavedModel into container's model folder
```
$ docker cp /models/MODEL_NAME serving_base:models/MODEL_NAME
```

* Commit the container that's serving our model by changing MODEL_NAME to match our model's name
```
$ docker -commit --change "ENV MODEL_NAME <model_name>" serving_base <Container name>
```

* Stop serving_base
```
$ docker kill serving_base
```

* Now, your custom image can be published and deployed to Kubernetes or other clustering services and a RestAPI would be setup which can be used to infer from any platform.
```
$ docker run -p 8051:8051 <container name>
```

## To-do List
- [x] Build a model and train it
- [x] Evaluate the model
- [ ] Building a landing page using Flask
- [ ] Deploy the app
