# Med-Cabinet

###### Med Cabinet is for new cannabis consumers, especially those trying to get off of pharmaceuticals, who want to use cannabis as a means to battle medical conditions and ailments.

[Link to the Heroku app.](https://medicinalcannabis.herokuapp.com/ "Found out what strain is right for you!")

### The Model:

This web application was created using neuro-linguistic programming techniques and a neural network model. 

The model was trained from vector representaion of the text describing the effects, flavors, and description of each strain. Word similarity was used to expand the data by 10 times its original size to facilitate more accurate results. 

Using Spacy's en_core_web_md model, we were able to achieve 99% validation accuracy with our data! The model deployed to Heroku was trained with Spacy's small model, to accomodate for size restrictions.

To use this app, the user must input in the text field a description of their desired cannabis strain. Such description will be for useful if it contains desired flavors and effects.

###### This model and web app was authored by Nathan McDonough and Tomás Phillips