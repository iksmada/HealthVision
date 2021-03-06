

# Health Vision:  Assessing suspicious skin lesions using artificial intelligence

## **I.** INTRODUCTION

The application aims to direct people who have a skin lesion which they consider suspicious to a specialist or a general practitioner depending on its possible severity, reducing the risks of a person who has a health plan to seek a specialist when a general practitioner could follow their case, tending to reduce costs for health insurance companies.

According to the INCA (National Cancer Institute), non-melanoma skin cancer is the most frequent in Brazil, accounting for about 30% of all malignancies registered in the country. However, when discovered early, there is a chance of more than 90% cure for the disease. Considering this point, the application also aims to increase people's caution regarding skin cancer because of its ease of use, and with this, try to change their current scenario. The app will also promptly remind people to take photos of their skin lesions for follow-up, increasing the accuracy of the tecnology with time series analysis. Furthermore, the program will allow them to schedule their visits via app.

## **II.**   DEVELOPMENT

First of all, we validated the main purpose of the aplication in development through an online form which returned the feedback of 230 people. Considering this sample, 20% already have skin cancer, 16% said have a regular follow up (two to four times during the year) and 76% would like to have a system which helps them follow up on a skin lesion that they already have. 

A system was developed to solve a practical problem, seeking innovation through the innovative development of an algorithm that would add value considering a problem currently faced in health area. A custom convolutional neural network was created to extract the features and the classifier was based on a random forest. In order to optimize the dataset, a great deal of pre-processing was done to remove stickers which were present in approximately 9.000 images of all the 23.000 presents in our dataset.

To sum up, the mentioned system corresponds to an application that studies that database and, through artificial intelligence, returns if a person is prone to having skin cancer or not. In order for the person to perform this analysis, it is sufficient that the person takes a picture of his or her skin lesion and it will respond whether they should seek a specialist (if the algorithm detects that this lesion on the skin has characteristics of a more aggressive cancer), or should be directed to a general practitioner, where they can give the official feedback (in this case considering that your skin lesion has more characteristics of something non-aggressive). 

Our first interaction of our algorithm had an accuracy of 84%. To improve this result, we added 3 database sets of malicious lesions, with specific characteristics, resulting in an accuracy of 93%.

Besides that, the application will allow the user to schedule an appointment and also reminder the person to take a new photo and do the process again after a certain time.

##  **III.**  CONCLUSION

The database accuracy is 93% with an F1 score of 92%. More importantly our false negative rate was under 4%, making the solution viable for the market. We expect to improve and help people and health care professionals to make a faster follow up on this issue and also achieve a larger impact on the population to care for and raise awareness of this disease, which is the type of cancer with most cases in Brazil.

