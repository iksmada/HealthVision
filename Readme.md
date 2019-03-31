

# Health Vision:  Assessing suspicious skin lesions using artificial intelligence

## 1. INTRODUCTION

The application aims to drive people who have a skin lesion which they consider suspicious to a specialist or a general practitioner depending on its possible severity, reducing the risks of a person who have a health plan to seek a specialist when a general practitioner could follow your case, tending to reduce costs for health insurance companies.

According to the INCA (National Cancer Institute), non-melanoma skin cancer is the most frequent in Brazil, accounting for about 30% of all malignancies registered in the country. However, when discovered early, there is a chance of more than 90% cure for the disease. Considering this point, the application also aims to increase people's caution regarding it because of its ease of use, and, with this, try to change this current scenario. The app will also promptly remind people to take photos of their skin lesions for follow-up, increasing the accuracy of the tecnology because it is a time series. Furthermore the application will allow them to schedule their visits through this.

## 1. **II.****   DEVELOPMENT**

        First of all, we validated the main purpose of the aplication in development throught an online form which return to us the feedback of 230 people. Considering this amount, 20% already have skin cancer, 16% said have a regular follow up (two to four times during the year) and 76% would like to have a system who helped them to follow a skin lesion that they already have. A system was developed to solve a practical problem, seeking innovation through the innovative development of an algorithm that would add value considering a problem currently faced in health area. It was created a custom convolutional neural network to extract the features and to sort the it was used a random forest. In order to optimize the dataset it was done a pre-processing to remove a sticker which was active in approximately 9.000 images of all the 23.000 presents in our dataset. To sum up, the mentioned system corresponds to an application that studies that database and, through artificial intelligence, returns if a person is prone to having skin cancer or not. In order for the person to perform this analysis, it is sufficient that the person takes a picture of his or her skin lesion and it will return if they should seek a specialist (if the algorithm detects that this lesion on the skin has characteristics of a more aggressive cancer), or you will drive it to a general practitioner, where they can give the official feedback (in this case considering that your skin lesion has more characteristics of something non-aggressive). Besides that, the application will allow the user to schedule an appointment and also reminder the person to take a new photo and do the process again after a certain time.

## 1. **III.**  CONCLUSION

        The database accuracy is 80%. So we expect to improve and help people and healh care professionals to make a faster follow up in this issue and also achieve a larger part of the population to care and be aware of this disease, which is the type of cancer with most cases in Brazil.

