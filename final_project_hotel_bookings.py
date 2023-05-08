import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)
import numpy as np
import timeit
import statsmodels.api as sm
from tabulate import tabulate
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from codecarbon import OfflineEmissionsTracker
from scipy.special import expit


## Prepare website
app_mode = st.sidebar.selectbox('Select page', ['Home', 'Visualization', 'Prediction']) 


if app_mode == 'Home':
  ### The first two lines of the code load an image and display it using the st.image function.
  image1 = Image.open('hotel.jpeg')
  image2 = Image.open('city.jpeg')
  img1 = Image.open("hotel.jpeg")
  img2 = Image.open("city.jpeg")
  width, height = img1.size
  img2 = img2.resize((width, height))
  st.image([img1, img2], width=350)

  ## The st.title function sets the title of the web application to "Final Project - 01 Introduction Page".
  st.title("Hotel Bookings - Homepage")

  df = pd.read_csv("df.csv")

  st.info("The dataset contains data on bookings from two different hotels - a resort hotel and a city hotel - both located in Portugal.")
  st.info("This interactive website will show hotel cancellations according to certain variables, as listed below. Users will be able to see model accuracy, logstic regression, and KNN model. This website will be able to predict whether hotel bookings will be cancelled or not by looking at variables and such variables will also explain if travelers will stay in city or resort hotels.")


  st.markdown("### 00 - Description of the Dataset")
  head = st.radio('View the **top** (head) or the **bottom** (tail) of the dataset', ('Head', 'Tail'))
  num = st.number_input('Select the number of rows to view:', 5, 158)
  if head == 'Head':
    st.dataframe(df.head(num))
  else:
    st.dataframe(df.tail(num))

  st.caption('Data Source - Nuno Antonio, Ana de Almeida, Luis Nunes, Hotel booking demand datasets, Data in Brief, Volume 22, 2019. https://doi.org/10.1016/j.dib.2018.11.126.')
  st.markdown("##### Size of the Dataset:")
  st.text('(Number of Hotel Bookings, Number of Variables)')
  st.write(df.shape)

  st.markdown("##### Description of each variable:")
  st.markdown(" **Hotel**: Hotel (Resort Hotel or City Hotel)")
  st.markdown(" **is_canceled**: Value indicating if the booking was canceled (1) or not (0)")
  st.markdown("**lead_time**: Number of days that elapsed between the entering date of the booking into the PMS and the arrival date")
  st.markdown(" **arrival_date_year**: Year of arrival date")
  st.markdown(" **arrival_date_month**: Month of arrival date")
  st.markdown(" **arrival_date_week_number**: Week number of year for arrival date")
  st.markdown(" **arrival_date_day_of_month**: Day of arrival date")
  st.markdown(" **stays_in_weekend_nights**: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel")
  st.markdown(" **stays_in_week_nights**: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel")
  st.markdown(" **adults**: Number of adults")
  st.markdown(" **children**: Number of children")
  st.markdown(" **babies**: Number of babies")
  st.markdown(" **meal**: Type of meal booked. Categories are presented in standard hospitality meal packages:\n- Undefined/SC – no meal package\n- BB – Bed & Breakfast\n- HB – Half board (breakfast and one other meal – usually dinner)\n- FB – Full board (breakfast, lunch and dinner")
  st.markdown(" **country**: Country of origin. Categories are represented in the ISO 3155–3:2013 format")
  st.markdown(" **market_segment**: Market segment designation. In categories, the term “TA” means “Travel Agents” and “TO” means “Tour Operators”")
  st.markdown(" **distribution_channel**: Booking distribution channel. The term “TA” means “Travel Agents” and “TO” means “Tour Operators”")
  st.markdown(" **is_repeated_guest**: Value indicating if the booking name was from a repeated guest (1) or not (0)")
  st.markdown(" **previous_cancellations**: Number of previous bookings that were cancelled by the customer prior to the current booking")
  st.markdown(" **previous_bookings_not_canceled**: Number of previous bookings not cancelled by the customer prior to the current booking")
  st.markdown(" **reserved_room_type**: Code of room type reserved. Code is presented instead of designation for anonymity reasons.")
  st.markdown(" **assigned_room_type**: Code for the type of room assigned to the booking. Sometimes the assigned room type differs from the reserved room type due to hotel operation reasons (e.g. overbooking) or by customer request. Code is presented instead of designation for anonymity reasons.")
  st.markdown(" **booking_changes**: Number of changes/amendments made to the booking from the moment the booking was entered on the PMS until the moment of check-in or cancellation")
  st.markdown(" **deposit_type**: Indication on if the customer made a deposit to guarantee the booking. This variable can assume three categories:\n- No Deposit – no deposit was made\n- Non Refund * a deposit was made in the value of the total stay cost\n- Refundable – a deposit was made with a value under the total cost of stay.")
  st.markdown(" **agent**: ID of the travel agency that made the booking")
  st.markdown(" **company**: ID of the company/entity that made the booking or responsible for paying the booking. ID is presented instead of designation for anonymity reasons")
  st.markdown(" **days_in_waiting_list**: Number of days the booking was in the waiting list before it was confirmed to the customer")
  st.markdown(" **customer_type**: Type of booking, assuming one of four categories:\n- Contract - when the booking has an allotment or other type of contract associated to it\n- Group – when the booking is associated to a group\n- Transient – when the booking is not part of a group or contract, and is not associated to other transient booking\n- Transient-party – when the booking is transient, but is associated to at least other transient booking")
  st.markdown(" **adr**: Average Daily Rate as defined by dividing the sum of all lodging transactions by the total number of staying nights")
  st.markdown(" **required_car_parking_spaces**: Number of car parking spaces required by the customer")
  st.markdown(" **total_of_special_requests**: Number of special requests made by the customer (e.g. twin bed or high floor)")
  st.markdown(" **reservation_status**: Reservation last status, assuming one of three categories:\n- Canceled – booking was canceled by the customer\n- Check-Out – customer has checked in but already departed\n- No-Show – customer did not check-in and did inform the hotel of the reason why")
  st.markdown(" **reservation_status_date**: Date at which the last status was set. This variable can be used in conjunction with the ReservationStatus to understand when was the booking canceled or when did the customer checked-out of the hotel")

  hotel_counts = df['hotel'].value_counts()
  fig = px.pie(values=hotel_counts.values, names=hotel_counts.index,
              hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel1)

  fig.update_layout(title="Proportion of Guests in Resort vs. City Hotels")

  st.plotly_chart(fig)

  counts = df.groupby(['hotel', 'is_canceled']).size().reset_index(name='count')
  counts['status'] = counts['is_canceled'].map({0: 'Booked', 1: 'Canceled'})
  fig = px.pie(counts, values='count', names='status',
              hole=0.6, color='is_canceled',
              color_discrete_sequence=px.colors.qualitative.Pastel1,
              labels={'is_canceled': 'Status', 'count': 'Count'})
  fig.update_layout(title="Proportion of Canceled and Booked Reservations")
  st.plotly_chart(fig)

  hotels_data = pd.read_csv("df.csv")
  fig1, ax = plt.subplots()
  sns.boxplot(x="hotel", y="adr", data=hotels_data, ax=ax, palette="Accent")
  ax.set_title("Average Daily Rate by Hotel Type")
  ax.set_xlabel("Hotel Type")
  ax.set_ylabel("Average Daily Rate")
  st.pyplot(fig1)

  hotels_data = pd.read_csv("df.csv")
  fig2, ax = plt.subplots()
  sns.boxplot(x="hotel", y="stays_in_week_nights", data=hotels_data, ax=ax, palette="Accent")
  ax.set_title("Number of Week Nights Stayed by Hotel Type")
  ax.set_xlabel("Hotel Type")
  ax.set_ylabel("Number of Week Nights")
  st.pyplot(fig2)

  st.markdown("### 01 - Descriptive Statistics")
  st.dataframe(df.describe())

  st.markdown("### 02 - Missing Values")
  dfnull = df.isnull().sum()/len(df)*100
  avgmiss = dfnull.sum().round(2)
  st.write("Total number of missing values:",avgmiss)
  st.write(dfnull)

elif app_mode == 'Visualization':
  ### Titles and sidebar
  st.title("Hotel Bookings - Visualization")
  url = "https://lookerstudio.google.com/u/0/reporting/00b01fc2-c9e7-4db4-9a07-a7608a305b20/page/GaGND"
  ##html = f'<iframe width="1000" height="600" src="{url}" frameborder="0"></iframe>'
  ##st.markdown(html, unsafe_allow_html=True)
  ##st.info("**Access the Looker Studio Dashboard here:** https://lookerstudio.google.com/u/0/reporting/00b01fc2-c9e7-4db4-9a07-a7608a305b20/page/GaGND")
  from streamlit.components.v1 import html
  html_string = f'<iframe width="100%" height="600" src="{url}"></iframe>'
  html(html_string, width=800, height=600)
  st.info("**Access the Looker Studio Dashboard here:** https://lookerstudio.google.com/u/0/reporting/00b01fc2-c9e7-4db4-9a07-a7608a305b20/page/GaGND")


else: 
  ### Titles and sidebar
  st.title("Hotel Bookings - Predictions")
  model_mode = st.sidebar.selectbox('📈 Select Prediction Model', ['Logistic Regression', 'K-Nearest Neighbors Algorithm']) 

  ### Dataframe cleaning - encode categorical variables to numerical
  df = pd.read_csv("df.csv")
  categorical_columns = ['hotel','meal', 'arrival_date_year', 'arrival_date_month', 'country', 'market_segment', 'distribution_channel', 'deposit_type', 'customer_type']
  for col in categorical_columns:
      encoder = LabelEncoder()
      df[col] = encoder.fit_transform(df[col])
  df.dropna()
  columns_to_drop = ['company', 'reservation_status_date', 'reserved_room_type', 'arrival_date_day_of_month', 'assigned_room_type', 'reservation_status']
  df.drop(columns = columns_to_drop, inplace=True)

  column_labels = {'hotel': 'Hotel Type',
                  'is_canceled': 'Is Canceled',
                  'lead_time': 'Lead Time',
                  'arrival_date_year': 'Arrival Year',
                  'arrival_date_month': 'Arrival Month',
                  'arrival_date_week_number': 'Arrival Week Number',
                  'stays_in_weekend_nights': 'Stays in Weekend Nights',
                  'stays_in_week_nights': 'Stays in Week Nights',
                  'adults': 'Adults',
                  'children': 'Children',
                  'babies': 'Babies',
                  'meal': 'Meal',
                  'country': 'Country',
                  'market_segment': 'Market Segment',
                  'distribution_channel': 'Distribution Channel',
                  'is_repeated_guest': 'Is Repeated Guest',
                  'previous_cancellations': 'Previous Cancellations',
                  'previous_bookings_not_canceled': 'Previous Bookings Not Canceled',
                  'booking_changes': 'Booking Changes',
                  'deposit_type': 'Deposit Type',
                  'agent': 'Agent',
                  'days_in_waiting_list': 'Days in Waiting List',
                  'customer_type': 'Customer Type',
                  'adr': 'ADR',
                  'required_car_parking_spaces': 'Required Car Parking Spaces',
                  'total_of_special_requests': 'Total of Special Requests'}

  df = df.rename(columns=column_labels)


  ### Select the dependent variable
  list_dependent = ['Hotel Type', 'Is Canceled']
  selected_dependent = st.sidebar.selectbox('🧩  Select Variable to Predict', list_dependent)

  if model_mode == 'Logistic Regression':
    
    train_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.70)
    
    df_without_dependent = df.drop(list_dependent, axis=1) 
    list_explanatory = df_without_dependent.columns

    try:
      selected_explanatory = st.multiselect("##### Select Explanatory Variables", list_explanatory, default = ['Market Segment','Booking Changes'])

      df_explanatory = df_without_dependent[selected_explanatory]
      X =  df_explanatory
      y = df[selected_dependent]

      start_time = timeit.default_timer()
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= train_size, random_state= 42)
      log_model = LogisticRegression()
      log_model.fit(X_train, y_train)
      y_pred = log_model.predict(X_test)
      
      # Display model accuracy
      accuracy = accuracy_score(y_test, y_pred)
      st.write("#### 🎯 Model Accuracy:", np.round(accuracy, 2))

      # Convert classification report dictionary to a DataFrame
      classification_rep = classification_report(y_test, y_pred, output_dict=True)
      table_df = pd.DataFrame(classification_rep).transpose()
      
      # Convert float values to string with 2 decimal places for formatting
      table_df['precision'] = table_df['precision'].apply(lambda x: f"{x:.2f}")
      table_df['recall'] = table_df['recall'].apply(lambda x: f"{x:.2f}")
      table_df['f1-score'] = table_df['f1-score'].apply(lambda x: f"{x:.2f}")
      
      # Display the classification report as a table
      st.markdown("#### 📊 Classification Report:")
      st.table(table_df.style.set_caption("").hide_index())

      if selected_dependent == 'Hotel Type':
        st.info("Note that value **0** corresponds to **City Hotel**, and value **1** corresponds to **Resort Hotel**")

        ## P-values table
        st.markdown("##### P-values of predictor variables:")
        logit_model = sm.Logit(y, X)
        result = logit_model.fit()
        p_values = result.pvalues
        st.table(p_values)
        
        ## Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        label_names = ['City Hotel', 'Resort Hotel']
        sns.heatmap(cm, annot = True, fmt = 'g', xticklabels = label_names, yticklabels = label_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        info_text = '''
        **Interpreting the confusion matrix:**

        The confusion matrix summarizes the performance of a classification model by
        showing the number of true and false positives and negatives for each class.

        - True positives: the number of instances that were correctly predicted as positive.
        - False positives: the number of instances that were incorrectly predicted as positive.
        - True negatives: the number of instances that were correctly predicted as negative.
        - False negatives: the number of instances that were incorrectly predicted as negative.

        The confusion matrix can be used to calculate various performance metrics for the model,
        such as accuracy, precision, recall, and F1 score. These metrics can help to evaluate
        the overall effectiveness of the model and identify areas for improvement.
        '''
        st.markdown("#### ⚙️ Logistic regression confusion matrix:")
        st.info(info_text)
        st.pyplot(plt.gcf())

        ## ROC curve
        probas = log_model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probas)
        fpr, tpr, thresholds = roc_curve(y, probas)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc='lower right')
        st.markdown("#### ⚙️ ROC curve:")
        st.info("The ROC curve is a graphical representation of the performance of a binary classifier at different decision thresholds. It is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. AUC (Area Under the Curve) is a single number that summarizes the ROC curve performance. AUC ranges from 0 to 1, with higher values indicating better performance. AUC of 0.5 represents a random guess, while an AUC of 1.0 represents a perfect classifier. You can use the ROC curve to compare the performance of different classifiers or to choose the best threshold for your classifier based on your specific needs.")
        st.pyplot(fig)

      else:
        st.info("Note that value **0** corresponds to **Is Not Canceled**, and value **1** corresponds to **Is Canceled**")

        ## P-values table
        st.markdown("##### P-values of predictor variables:")
        logit_model = sm.Logit(y, X)
        result = logit_model.fit()
        p_values = result.pvalues
        st.table(p_values)
        
        ## Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        label_names = ['Is Not Canceled', 'Is Canceled']
        sns.heatmap(cm, annot = True, fmt = 'g', xticklabels = label_names, yticklabels = label_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        info_text = '''
        **Interpreting the confusion matrix:**

        The confusion matrix summarizes the performance of a classification model by
        showing the number of true and false positives and negatives for each class.

        - True positives: the number of instances that were correctly predicted as positive.
        - False positives: the number of instances that were incorrectly predicted as positive.
        - True negatives: the number of instances that were correctly predicted as negative.
        - False negatives: the number of instances that were incorrectly predicted as negative.

        The confusion matrix can be used to calculate various performance metrics for the model,
        such as accuracy, precision, recall, and F1 score. These metrics can help to evaluate
        the overall effectiveness of the model and identify areas for improvement.
        '''
        st.markdown("#### ⚙️ Logistic regression confusion matrix:")
        st.info(info_text)
        st.pyplot(plt.gcf())

        ## ROC curve
        probas = log_model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probas)
        fpr, tpr, thresholds = roc_curve(y, probas)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc='lower right')
        st.markdown("#### ⚙️ ROC curve:")
        st.info("The ROC curve is a graphical representation of the performance of a binary classifier at different decision thresholds. It is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. AUC (Area Under the Curve) is a single number that summarizes the ROC curve performance. AUC ranges from 0 to 1, with higher values indicating better performance. AUC of 0.5 represents a random guess, while an AUC of 1.0 represents a perfect classifier. You can use the ROC curve to compare the performance of different classifiers or to choose the best threshold for your classifier based on your specific needs.")
        st.pyplot(fig)


    except ValueError:
      st.error("Please select at least one explanatory variable!")

  ## KNN
  else: 

    ## Sidebar
    train_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.70)
    df_without_dependent = df.drop(list_dependent, axis=1) 
    list_explanatory = df_without_dependent.columns
    k_parameter = st.sidebar.slider("Input K",5,1,10)
    distance_metric = st.sidebar.radio("Select a distance metric:",("euclidean", "manhattan", "chebyshev"))

    try:
      selected_explanatory = st.multiselect("##### Select Explanatory Variables", list_explanatory, default = ['Market Segment','Booking Changes'])
      df_explanatory = df_without_dependent[selected_explanatory]
      X =  df_explanatory
      y = df[selected_dependent]

      start_time = timeit.default_timer()
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= train_size, random_state= 42)
      knn = KNeighborsClassifier(n_neighbors = k_parameter, metric=distance_metric)
      knn.fit(X_train,y_train)
      y_pred = knn.predict(X_test)

        # Display model accuracy
      accuracy = accuracy_score(y_test, y_pred)
      st.write("#### 🎯 Model Accuracy:", np.round(accuracy, 2))

        # Convert classification report dictionary to a DataFrame
      classification_rep = classification_report(y_test, y_pred, output_dict=True)
      table_df = pd.DataFrame(classification_rep).transpose()
        
        # Convert float values to string with 2 decimal places for formatting
      table_df['precision'] = table_df['precision'].apply(lambda x: f"{x:.2f}")
      table_df['recall'] = table_df['recall'].apply(lambda x: f"{x:.2f}")
      table_df['f1-score'] = table_df['f1-score'].apply(lambda x: f"{x:.2f}")
        
        # Display the classification report as a table
      st.markdown("#### 📊 Classification Report:")
      st.table(table_df.style.set_caption("").hide_index())
      
      if selected_dependent == 'Hotel Type':
        st.info("Note that value **0** corresponds to **City Hotel**, and value **1** corresponds to **Resort Hotel**.")
      else:
          st.info("Note that value **0** corresponds to **Is Not Canceled**, and value **1** corresponds to **Is Canceled**.")

      error_rate = []

      for i in range(1,40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))

      ## Error rate
      st.write('##### Displaying error rate with a line chart') 
      fig, ax = plt.subplots(figsize=(10, 6))
      ax.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
      ax.set_title('Error Rate vs. K Value')
      ax.set_xlabel('K')
      ax.set_ylabel('Error Rate')
      st.pyplot(fig)
      st.warning('Adjust your input K according to this model to get a better accuracy score. **Hint**: Use the elbow method to select the optimal number of clusters for KNN clustering!')
      
      st.markdown("##### P-values of predictor variables:")
      logit_model = sm.Logit(y, X)
      result = logit_model.fit()
      p_values = result.pvalues
      st.table(p_values)

      if selected_dependent == 'Hotel Type':

        ## Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        label_names = ['City Hotel', 'Resort Hotel']
        sns.heatmap(cm, annot = True, fmt = 'g', xticklabels = label_names, yticklabels = label_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')


        info_text = '''
        **Interpreting the confusion matrix:**

        The confusion matrix summarizes the performance of a classification model by
        showing the number of true and false positives and negatives for each class.

        - True positives: the number of instances that were correctly predicted as positive.
        - False positives: the number of instances that were incorrectly predicted as positive.
        - True negatives: the number of instances that were correctly predicted as negative.
        - False negatives: the number of instances that were incorrectly predicted as negative.

        The confusion matrix can be used to calculate various performance metrics for the model,
        such as accuracy, precision, recall, and F1 score. These metrics can help to evaluate
        the overall effectiveness of the model and identify areas for improvement.
        '''
        st.markdown(f'#### ⚙️ Confusion matrix with KNN algorithm and {distance_metric} distance metric:')
        st.info(info_text)
        st.pyplot(plt.gcf())

      else:

        ## Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        label_names = ['Is Not Canceled', 'Is Canceled']
        sns.heatmap(cm, annot = True, fmt = 'g', xticklabels = label_names, yticklabels = label_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        info_text = '''
        **Interpreting the confusion matrix:**

        The confusion matrix summarizes the performance of a classification model by
        showing the number of true and false positives and negatives for each class.

        - True positives: the number of instances that were correctly predicted as positive.
        - False positives: the number of instances that were incorrectly predicted as positive.
        - True negatives: the number of instances that were correctly predicted as negative.
        - False negatives: the number of instances that were incorrectly predicted as negative.

        The confusion matrix can be used to calculate various performance metrics for the model,
        such as accuracy, precision, recall, and F1 score. These metrics can help to evaluate
        the overall effectiveness of the model and identify areas for improvement.
        '''
        st.markdown(f'#### ⚙️ Confusion matrix with KNN algorithm and {distance_metric} distance metric:')
        st.info(info_text)
        st.pyplot(plt.gcf())

      ## ROC curve
      probas = knn.predict_proba(X_test)[:, 1]
      fpr, tpr, thresholds = roc_curve(y_test, probas)
      roc_auc = auc(fpr, tpr)
      fig, ax = plt.subplots()
      ax.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
      ax.plot([0, 1], [0, 1], 'k--', lw=2)
      ax.set_xlim([0.0, 1.0])
      ax.set_ylim([0.0, 1.05])
      ax.set_xlabel('False Positive Rate')
      ax.set_ylabel('True Positive Rate')
      ax.set_title('Receiver Operating Characteristic')
      ax.legend(loc="lower right")
      st.markdown("#### ⚙️ ROC curve:")
      st.info("The ROC curve is a graphical representation of the performance of a binary classifier at different decision thresholds. It is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. AUC (Area Under the Curve) is a single number that summarizes the ROC curve performance. AUC ranges from 0 to 1, with higher values indicating better performance. AUC of 0.5 represents a random guess, while an AUC of 1.0 represents a perfect classifier. You can use the ROC curve to compare the performance of different classifiers or to choose the best threshold for your classifier based on your specific needs.")
      st.pyplot(fig)

    except ValueError:
      st.error("Please select at least one explanatory variable!")


  ## Performance metrics
  st.write('#### 🚀 Execution Time:')
  end_time = timeit.default_timer()
  elapsed_time = end_time - start_time
  st.success(f'{elapsed_time:.4f} seconds')

  st.write('#### 🌱 Emissions Tracker:')
  tracker = OfflineEmissionsTracker(country_iso_code="FRA") # FRA = France
  tracker.start()
  results = tracker.stop()
  st.success('%.12f kWh' % results)
