LightGBM Classifier Using Tf-IDF preprocessing for Text Classification - Base problem category as per Ready Tensor specifications.

- lightgbm
- sklearn
- python
- pandas
- numpy
- scikit-optimize
- flask
- nginx
- uvicorn
- docker
- text classification

This is a Text Classifier that uses a LightGB<> Classifier through xgboost.

The data preprocessing step includes tokenizing the input text, applying a tf-idf vectorizer to the tokenized text, and applying Singular Value Decomposition (SVD) to find the optimal factors coming from the original matrix. In regards to processing the labels, a label encoder is used to turn the string representation of a class into a numerical representation.

Hyperparameter Tuning (HPT) is conducted by finding the optimal learning rate, loss reduction required to make a further partition on a leaf node of the tree, and depth of the tree.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as email spam detection, customer churn, credit card fraud detection, cancer diagnosis, and titanic passanger survivor prediction.

This Text Classifier is written using Python as its programming language. XGBoost is used to implement the main algorithm. SciKitLearn creates the data preprocessing pipeline and evaluates the model. Numpy, pandas, and NLTK are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT. Flask + Nginx + gunicorn are used to provide web service which includes two endpoints- /ping for health check and /infer for predictions in real time.
