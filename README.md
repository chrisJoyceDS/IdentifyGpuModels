README!

### Overview

This README Covers:
- Background
- Problem Statement
- Data Dictionary
- Brief Summary of Analysis
- Conclusions and Recommendations
- File Structure
- Sources
- Powerpoint Presentation

## Background:
GPU's became popular in the the early 2000s by NVIDIA and ATI (which is currently a part of AMD). Originally designed for graphics processing (hence the name Graphic Processing Units), GPU's started seeing prominence in Machine Learning and Cryptocurrency spaces over the last 20 years. The market has been almost entirely owned by NVIDIA and AMD, but with the rise in Personal Computer and Console sales during the pandemic and the resulting shortage of chips, this provided an interesting opportunity for Intel who has been the dominant CPU manufacturer for the last 30 years. Intel has released GPU's in the past, but in October of 2022 they released two brand new models: the Arc A750 and the Arc A770. This project is focused solely on the new models.

## Problem Statement:
You've been hired by Intel as a Data Scientist to create a classification model with the purpose of identifying which subreddit a given text post is from. The two subreddits Intel are immediately interested in are NVIDIA and AMD, the current GPU Market owners. This project is designed to help the Intel marketing team identify any potential base of GPU users who are at least mentioning the new Intel Models. Additionally, try to idenfity if one of our newer GPU models are mentioned in each reddit post you pull. Our new model numbers are: A750, A770 8GB, A770 16GB.

___
<h1>Data Dictionary</h1>

All Features Included in both EDA and Modeling:
|Feature|Type|Description|
|:---|:---:|---:|
|**subreddit**|*object*|subreddit name|
|**selftext**|*object*|Body text of an individual reddit submission post|
|**title**|*object*|Title text of an indidivual reddit submission post|
|**author**|*object*|Author of an individual reddit submission post|
|**num_comments**|*int*|Count of the number of comments for an individual reddit submission post|
|**selftext_word_count**|*int*|Word count of the body text of an individual reddit submission post|
|**selftext_length**|*int*|Length of the body text of an individual reddit submission post|
|**title_word_count**|*int*|Word count of the title text of an individual reddit submission post|
|**title_length**|*int*|Length of the body text of an individual reddit submission post|
|**intel_gpu**|*boolean*|Boolean value representing if a subreddit post referenced or did not reference the new Intel GPU Models|
|**nvidia_gpu**|*boolean*|Boolean value representing if a subreddit post referenced or did not reference the any NVIDIA GPU Models|
|**amd_gpu**|*int*|Boolean value representing if a subreddit post referenced or did not reference the any AMD GPU Models|

---

## Brief Summary of Analysis
We were tasked with two objectives: create a classification model to in order to identify a given reddit post's subreddit and to find any reference of Intel's new gpu models the A750 and A770. From the data we collected (Reddit r/nvidia and r/Amd posts since January 1st, 2015) we were ultimately successful on only one of those objectives.

One major win is there is a lot of information to be learned and gleaned from Reddit, its subreddits, its users, and their behavior. We learned that, at least in the two subreddits of this project, that users utilize the title field for the more important aspects of their queries than the body of their posts. In our dataset of ~200k posts, both AMD and NVIDIA subreddit users combined created ~41k posts with only 1 word in the selftext but on average had ~10 words in the title; understanding this behavior can help us target our search more effectively in future endeavors.

Our major observation of this exercise is the effect of the number of observations in classifying subreddits. Our original model, V1, retains the top spot for highest scores and highest number of True Negative and True Positive Values. It was also the model trained on the most observations of the three. V2 had less observations due to cleaning, and V3 had even less observations due to undersampling the majority to match the minority. In both V2 and V3, our models performance dipped by a hundredth of a point in accuracy and balanced accuracy. In future iterations, we should save these 3 models and create a 4th with a stacking model using these three different performers.

Unfortunately we were unable to locate any reference of the newer Intel GPU models in either subreddit across the ~200k posts. This does not necessarily mean that there was no reference, but it does suggest that our techniques (regex and pandas apply) were rudimentary in nature. To better identify GPU model numbers/references, we should invest time in training a Named Entity Recognition model using a larger dataset that encompasses multiple subreddits and includes comments as well.

---
### Conclusions:
- The two subreddits follow similar patterns of behavior and structure when looking at selftext and title wordcount and length.
- Given enough data, these similarities do not prevent our model from correctly predicting which subreddit a given post is from.
- Regular Expression is not a precise enough technique to pick up on all the different combinations a reddit user could use to label a GPU Model.

### Recommendations:
- Add a stacking classifier for version 4.
- Increase the number of subreddits we include in this study to get a much wider range of conversation.
- Include comments in the analysis to find mentions that could fall outside of subreddit moderator rules.
- Invest in more precise but higher cost methods of identifying model numbers such as Named Entity Recognition models, especially if you choose to follow the first two recommendations. 

---

File Structure:
* Code:
    1: 01_get_subreddit_submissions.ipynb - Function to pull subreddit posts
    2: 02_clean_observations.ipynb - Pushshift API Response Data Cleaning
    3: 03_eda.ipynb - Exploratory Data Analysis and Visualization
    4: 04_modeling.ipynb - Feature Engineering, Modeling and Conclusion
    5: model_methods.py - Python file containing the different functions used to preprocess and model the data
* data:
    1: model_scores.csv - CSV file containing the scores for each model run during this project and the different versions
* README.md - README
* Executive_Summary.md - Executive Summary of the initiative sponsored by Intel


---
## Sources

GPU Model Number Schemas per Company: https://techteamgb.co.uk/2019/10/25/gpu-names-explained-rtx-gtx-rx/
Background on recent GPU Market: https://www.digitaltrends.com/computing/catastrophic-gpu-shortage-a-chronological-history/
Information about Intel Arc Series: https://en.wikipedia.org/wiki/Intel_Arc
GPU History: https://medium.com/neuralmagic/a-brief-history-of-gpus-27122d8fd45

---

## Powerpoint
https://docs.google.com/presentation/d/1PC_G_Qkm8V6P8mJgK4RGkQAH8zf9AMF4MvmsmDDGuXo/edit#slide=id.g209c8662062_0_103