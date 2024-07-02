# Machine Learning in Data Science
### Analyzing, Plotting, Visualizing, and Predicting Data - Supervised Learning
You will complete two simple linear regressions for this exercise, for each dataset and problem statement, you should complete and include the following: 
General Required Elements
1.	Model building and linear regression process: Use scikit learn, train_test_split to fit the model, run the linear regression and predict the target variable (y).
2.	Model Results: Print the intercept and coefficient of the line.
3.	Model Accuracy: Print accuracy metrics from scikit learn, including r-squared
4.	Visualizations: Create a least 3 visualizations:
a.	Before running the linear regression:
i.	(1) A scatterplot of the raw data, x vs y.  See Note: Scientific Notation in matplotlib below.

b.	After running the linear regression:  
i.	(2) Plot actual vs predicted values
ii.	(3) Plot the error
•	Finally be sure to comment input and output of each code cell or cells that are associated with the 4 numbered elements above (model building/linear regression process, results, accuracy and 3 visualizations), with explanations geared to a non-data science colleague.  
o	Do not simply copy the comments in the sample python code, since they are written to an audience of data science students, not ‘non-data science’ colleagues.  
o	Use plain language and explain inputs and outputs for non-specialists.  
o	HINT: you will need at least 2-3 sentences to explain what linear regression is, what it means to train_test_split and how we arrive at a result.  You will need another 1-2 sentences explaining what r-squared is and how to interpret the result. 

LR1: Dataset: kc_house_data.csv (from the multiple regression tutorial)
LR1 specific instructions: 
•	Run a simple linear regression using sqft_living as the only feature.  
•	Be sure to include all of the General Required Elements above and include all python code and output.  
•	Use a training size of 0.8 so it is comparable to the multiple regression.   
•	Additional Questions: 
o	How does the simple linear regression model results (using sqft_living) compare to the multiple regression we ran in the tutorial.  
o	Which model (which features) better capture the variation in y based on the variation in the feature(s)?  Justify your response based on output from your model(s).

LR2: Dataset: student_scores.csv.
LR2 specific instructions: 
•	This simple dataset contains number of hours studied and corresponding results on an assessment.  
•	Create a simple linear regression model, using 0.8 as the training size.  
•	Include all the General Required Elements above and include all python code and output.
•	While there are no additional questions for LR2, do not forget the comments on input and output.  Your analysis of the results is required.


Note: Scientific Notation in matplotlib
If you are using matplotlib to create the scatterplot for your KC House data, you might see the y-axis for house price looks like this.

Matplotlib is converting our house prices to scientific notation (note the 1e6 at the top of the y-axis).  To remedy this, add the following line AFTER the final parentheses of your data.plot code OR just before plt.show(), depending on whether you created your data.plot in one 'go' or with several lines:
plt.ticklabel_format(axis="y", style="plain")

Alternatively you can drop matplotlib completely and use seaborn's pairplot functionality: 

https://seaborn.pydata.org/generated/seaborn.pairplot.html
