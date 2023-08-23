# Array and Dataframes
import numpy as np
import pandas as pd
# Evaluation: Visualization
import seaborn as sns
import matplotlib.pyplot as plt
# Evaluation: Statistical Analysis
from scipy import stats
# Modeling
from sklearn.model_selection import GridSearchCV
# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# Decision Tree
from sklearn.tree import DecisionTreeClassifier as dt, plot_tree, export_text
# Logistic Regression
from sklearn.linear_model import LogisticRegression as lr
# KNN
from sklearn.neighbors import KNeighborsClassifier


def eval_metrics(tp,tn,fp,fn):
        '''Input:
        True positive(tp),
        True negative (tn),
        False positive (fp),
        False negative (fn)

        Reminder:
        false pos true neg
        false neg true pos
        '''
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        return print(f'''Accuracy is: {accuracy}\nPrecision is: {precision}\nRecall is: {recall}''')


def organize_columns(train):
    '''
    Distinguishes between numeric and categorical data types
    Only selecting columns that would be relevant to visualize, no encoded data.

    '''
    cat_cols, num_cols = [], []
    explore = train[['gender','partner','dependents','tenure','phone_service','multiple_lines','online_security','online_backup',\
    'streaming_tv','streaming_movies','paperless_billing','monthly_charges','total_charges','churn','internet_service_type',\
        'payment_type','contract_type','senior_citizen','additional_services']]
    for col in explore:
        # check to see if its an object type,
        # if so toss it in categorical
        if train[col].dtype == 'O':
            cat_cols.append(col)
        # otherwise if its numeric:
        else:
            # check to see if we have more than just a few values:
            # if thats the case, toss it in categorical
            if train[col].nunique() < 10:
                cat_cols.append(col)
            # and otherwise call it continuous by elimination
            else:
                num_cols.append(col)
    return cat_cols, num_cols

def check_churn():
    sns.countplot(data=train,x='churn',alpha=0.8,linewidth=.4,edgecolor='black')
    return plt.show()

def without_hue(ax, feature):
    total = len(feature)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), size = 12)


def with_hue(ax, feature, Number_of_categories, hue_categories):
    a = [p.get_height() for p in ax.patches]
    patch = [p for p in ax.patches]
    for i in range(Number_of_categories):
        total = feature.value_counts().values[i]
        for j in range(hue_categories):
            percentage = '{:.1f}%'.format(100 * a[(j*Number_of_categories + i)]/total)
            x = patch[(j*Number_of_categories + i)].get_x() + patch[(j*Number_of_categories + i)].get_width() / 2 - 0.15
            y = patch[(j*Number_of_categories + i)].get_y() + patch[(j*Number_of_categories + i)].get_height() 
            ax.annotate(percentage, (x, y), size = 12)

def check_cat_distribution(df,target='churn'):
    '''
    Loop through a df and check their respective distributions.
    This is to be used with categorical datatypes, since the only 
    plot used is a countplot, with a target used as the hue to compare.
    '''
    
    for col in df:
        plt.figure(figsize=(12.5,8))
        sns.countplot(data=df,x=col,hue=target,alpha=0.8,linewidth=.4,edgecolor='black')
        plt.title(col)
        plt.show()
        print('''-------------------------------------------------------------''')


def check_num_distribution(df,dataset='train',target='churn'):
    '''
    Loop through a df and check their respective distributions.
    This is to be used with numerical datatypes, since the 
    plots used are hist plot and box plot, with a target used as the hue to compare.
    '''
    for col in df:
        sns.histplot(data=dataset, x=df[col],hue='churn')
        t = col.lower()
        plt.title(t)
        plt.show()
        sns.boxplot(data=dataset, x=col,hue='churn')
        plt.title(t)
        plt.show()
        print('''-------------------------------------------------------------''')


def chi2_test(col1, col2, a=.05):
    '''
    NOTE: Requires stats from scipy in order to function
    A faster way to test two columns desired for cat vs. cat statistical analysis.

    Default α is set to .05.

    Outputs crosstab and respective chi2 relative metrics.
    '''
    observed = pd.crosstab(col1, col2)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    if p < a:
        print(f'We can reject the null hypothesis with a p-score of:',{p})
    else:
        print(f'We fail to reject the null hypothesis with a p-score of:',{p})
    
    return observed


def check_p(p):
    '''
    checks p value to see association to a, depending on outcome will print
    outcome
    '''
    α = .05
    if p < α:
        return print(f'We can reject the null hypothesis with a p-score of:',{p})
    else:
        return print(f'We fail to reject the null hypothesis with a p-score of:',{p})


def get_classification_report(x_test, y_pred):
    '''
    Returns classification report as a dataframe.
    '''
    report = classification_report(x_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    return df_classification_report


def metrics(TN,FP,FN,TP):
    '''
    True positive(TP),
        True negative (TN),
        False positive (FP),
        False negative (FN)

        Reminder:
        false pos true neg
        false neg true pos
    '''
    combined = (TP + TN + FP + FN)

    accuracy = (TP + TN) / combined

    TPR = recall = TP / (TP + FN)
    FPR = FP / (FP + TN)

    TNR = TN / (FP + TN)
    FNR = FN / (FN + TP)


    precision =  TP / (TP + FP)
    f1 =  2 * ((precision * recall) / ( precision + recall))

    support_pos = TP + FN
    support_neg = FP + TN

    print(f"Accuracy: {accuracy}\n")
    print(f"True Positive Rate/Sensitivity/Recall/Power: {TPR}")
    print(f"False Positive Rate/False Alarm Ratio/Fall-out: {FPR}")
    print(f"True Negative Rate/Specificity/Selectivity: {TNR}")
    print(f"False Negative Rate/Miss Rate: {FNR}\n")
    print(f"Precision/PPV: {precision}")
    print(f"F1 Score: {f1}\n")
    print(f"Support (0): {support_pos}")
    print(f"Support (1): {support_neg}")


def decision_tree_compiled(x_train, y_train, df, plot=True):
    '''
    x_train = features
    y_train = target
    df is used to generate the values in churn in this case.
    Optional tree visualization. Default True.
    '''
    # tree object
    clf = dt(max_depth=3,random_state=4343)
    # fit
    clf.fit(x_train, y_train)
    # predict
    model_prediction = clf.predict(x_train)

    # generate metrics
    TN, FP, FN, TP = confusion_matrix(y_train, model_prediction).ravel()
    get_classification_report(y_train,model_prediction)
    metrics(TN, FP, FN, TP)

    # plot Tree
    if plot == True:
        labels = list(df['churn'].astype(str))
        fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=500)
        tree = plot_tree(clf, feature_names=x_train.columns.to_list(), class_names=labels,filled=True)
        plt.show()
    else:
        None
    return


def log_regression_compiled(x_train, y_train, x_test, y_test):
    '''
    Generates the logistic regression sklearn model.
    Finds the best fit C parameter using GridSearchCV by SKLearn
    x_train = features
    y_train = target
    '''
    # Parameters defined for GridSearch, train model
    param_grid_L1 = {'penalty': ['l1', 'l2'], 'C': np.arange(.1,5,.1)}
    logreg_tuned = lr(solver='saga', max_iter=500)
    logreg_tuned_gs = GridSearchCV(logreg_tuned, param_grid_L1, cv=5)
    logreg_tuned_gs.fit(x_train,y_train)

    # Predictions based on trained model
    y_predictions_lr_tuned = logreg_tuned_gs.predict(x_test)
    y_predictions_lr_prob_tuned = logreg_tuned_gs.predict_proba(x_test)

    # Output best C parameter
    print(f'Best fit "C" parameter (Determined by GridSearchCV): {logreg_tuned_gs.best_params_["C"]}')

    # tree object
    logit = lr(C=logreg_tuned_gs.best_params_["C"], random_state=4343)
    # fit
    logit.fit(x_train,y_train)
    # predict
    model_prediction = logit.predict(x_train)
    model_prediction_test = logit.predict(x_test)

    # generate metrics
    TN, FP, FN, TP = confusion_matrix(y_train, model_prediction).ravel()
    get_classification_report(y_train,model_prediction)
    metrics(TN, FP, FN, TP)
    # test metrics
    TN, FP, FN, TP = confusion_matrix(y_test, model_prediction_test).ravel()
    get_classification_report(y_test,model_prediction_test)

    return


def knn_compiled(x_train, y_train, x_test, y_test, weights='uniform'):
    '''
    Generates the K Nearest Neighbors SKLearn model.
    Finds the best fit C parameter using GridSearchCV by SKLearn
    Scans from n_neighbors = 1-30
    x_train = features
    y_train = target
    '''
 # Parameters defined for GridSearch, train model
    param_grid = {'n_neighbors': np.arange(1,30)}
    knn = KNeighborsClassifier(n_neighbors=1,weights='uniform')
    knn_cv = GridSearchCV(knn,param_grid,cv=5)
    knn_cv.fit(x_train,y_train)
    
    # Predictions based on trained model
    y_pred_knn_tuned = knn_cv.predict(x_test)
    print(f'Best fit "n_neighbors" parameter (Determined by GridSearchCV): {knn_cv.best_params_["n_neighbors"]}',\
      '\n--------------------------------------')

    # knn object
    knn = KNeighborsClassifier(n_neighbors=1, weights=weights)

    # fit
    knn.fit(x_train,y_train)
    
    # predict
    model_prediction = knn.predict(x_train)
    model_prediction_test = knn.predict(x_test)
    # generate metrics
    TN, FP, FN, TP = confusion_matrix(y_train, model_prediction).ravel()
    print(f'Train Class Report & Metrics:\
      \n---------------------------------------')
    print(f'{classification_report(y_train,model_prediction)}')
    print(f'{metrics(TN, FP, FN, TP)}','\n')
    # test metrics
    TN, FP, FN, TP = confusion_matrix(y_test, model_prediction_test).ravel()
    print(f'Test Classificiation Report & Metrics:\
      \n--------------------------------------')
    print(f'{classification_report(y_test,model_prediction_test)}')
    print(f'{metrics(TN, FP, FN, TP)}')
    return
