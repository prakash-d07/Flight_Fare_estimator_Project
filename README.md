# FlightFareEstimatorProject
## Problem Statement:

<p>Travelling through flights has become an integral part of today’s lifestyle as more and more people are opting for faster travelling options. The flight ticket prices increase or decrease every now and then depending on various factors like timing of the flights, destination, and duration of flights various occasions such as vacations or festive season. Therefore, having some basic idea of the flight fares before planning the trip will surely help many people save money and time.</p>

## Approach
<p>The main goal is to predict the fares of the flights based on different factors available in the dataset.</p>
<pre> 
<li> Data Exploration     : I started exploring dataset using pandas,numpy,matplotlib and seaborn. </li>
<li> Data visualization   : Ploted graphs to get insights about dependend and independed variables. </li>
<li> Feature Engineering  :  Removed missing values and created new features as per insights.</li>
<li> Model Selection I    :  1. Tested all base models to check the base accuracy.
                             2. Also ploted residual plot to check whether a model is a good fit or not.</li>
<li> Model Selection II   :  Performed Hyperparameter tuning using gridsearchCV and randomizedSearchCV.</li>
<li> Pickle File          :  Selected model as per best accuracy and created pickle file using joblib .</li>
<li> Webpage & deployment :  Created a webform that takes all the necessary inputs from user and shows output.
                                After that I have deployed project on heroku and Microsoft Azure</li></pre>



## Technologies Used
<pre> 
1. Python 
2. Sklearn
3. Flask
4. Html
5. Css
6. Pandas, Numpy 
7. Database 
8. Hosting
9. Docker 

</pre>
## Workflows

1. update config.yaml
2. update schema.yaml
3. update params.yaml
4. update the entity
5. update the configuration manager in src config
6. update the components
7. update the pipeline
8. update the main.py
9. update the app.py

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/prakash-d07/Flight_Fare_estimator_Project
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n venv python=3.8 -y
```

```bash
conda activate venv/
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```

### Prediction_images
![Alt text](https://github.com/prakash-d07/Flight_Fare_estimator_Project/blob/main/templates/PHOTO-2023-11-29-23-26-08.jpg)
![Alt text](https://github.com/prakash-d07/Flight_Fare_estimator_Project/blob/main/templates/PHOTO-2023-11-29-23-26-28.jpg)

