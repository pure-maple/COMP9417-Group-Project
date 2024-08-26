import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, f1_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#initial training and show learning curve

y_file = 'data_set/Y_remove.npy'
y_data = np.load(y_file)
f1_total = 0
loss_total = 0

for i in range (0,11):
    x_file = f'data_set/knn/X_relevant_label_{i+1}.npy'
    x_data = np.load(x_file)
    
    x_test = x_data
    y_test = y_data[:,i]

    x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.2, random_state=42)
    model = LogisticRegression(multi_class='multinomial', n_jobs=-1, max_iter = 10000)
    model.fit(x_train, y_train)
    y_prob = model.predict_proba(x_test)[:, 1]
    initial_log_loss = log_loss(y_test, y_prob)
    loss_total += initial_log_loss
    print(f"Binary Cross Entropy Loss {i}: {initial_log_loss}")
    
    y_pred_1 = model.predict(x_test)
    f1_1 = f1_score(y_test, y_pred_1)
    f1_total += f1_1

    print(f"f1 score {i}: {f1_1}")
    
        
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=model, 
        X=x_test, 
        y=y_test, 
        train_sizes=np.linspace(0.1, 1.0, 20),
        cv=5,  
        scoring='neg_log_loss',  
        n_jobs=-1 
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)

    # Plot the learning curves. Learning curve code refer from open source.
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.15)

    plt.plot(train_sizes, validation_mean, label='Validation score', color='green', marker='o')
    plt.fill_between(train_sizes, validation_mean - validation_std, validation_mean + validation_std, color='green', alpha=0.15)

    plt.title('Learning Curve')
    plt.xlabel('Training Data Size')
    plt.ylabel('Neg_log_loss')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()


loss_average = loss_total / 11
f1_average = f1_total / 11

print(f"Average loss of {x_file} is {loss_average}" )
print(f"Average f1_score of {x_file} is {f1_average}" )

#Start to tune. First to choose best weights

weights = [{0: 1, 1: 15}, {0: 1, 1: 11}, {0: 1, 1: 12}, {0: 1, 1: 13}, {0: 1, 1: 14}]

for weight in weights:
    f1_total = 0
    loss_total = 0
    
    for i in range (0,11):
        x_file = f'data_set/knn/X_relevant_label_{i+1}.npy'
        x_data = np.load(x_file)    
        x_test = x_data
        y_test = y_data[:,i]
        x_test = x_data
        y_test = y_data[:,i]
        x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.2, random_state=42)

        model = LogisticRegression(class_weight=weight, multi_class='multinomial', n_jobs=-1, max_iter = 10000)
        model.fit(x_train, y_train)
        y_prob = model.predict_proba(x_test)[:, 1]
        initial_log_loss = log_loss(y_test, y_prob)
        loss_total += initial_log_loss
        
        y_pred = model.predict(x_test)
        f1_1 = f1_score(y_test, y_pred)
        f1_total += f1_1
        
    print(f"Class Weights: {weight}")
    
    loss_average = loss_total / 11
    f1_average = f1_total / 11

    print(f"Average loss of {x_file} is {loss_average}" )
    print(f"Average f1_score of {x_file} is {f1_average}" )
    
    
#Show learning curve of best weight

f1_total = 0
loss_total = 0

for i in range (0,11):
    x_file = f'data_set/knn/X_relevant_label_{i+1}.npy'
    x_data = np.load(x_file)    
    x_test = x_data
    y_test = y_data[:,i]
    x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.2, random_state=42)
    
    model = LogisticRegression(multi_class='multinomial', n_jobs=-1, max_iter = 10000, class_weight= {0: 1, 1: 15})
    model.fit(x_train, y_train)
    y_prob = model.predict_proba(x_test)[:, 1]
    initial_log_loss = log_loss(y_test, y_prob)
    loss_total += initial_log_loss
    print(f"Binary Cross Entropy Loss {i}: {initial_log_loss}")
    
    y_pred_1 = model.predict(x_test)
    f1_1 = f1_score(y_test, y_pred_1)
    f1_total += f1_1

    print(f"f1 score {i}: {f1_1}")
       
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=model, 
        X=x_test, 
        y=y_test, 
        train_sizes=np.linspace(0.1, 1.0, 20),
        cv=5,  
        scoring='neg_log_loss',  
        n_jobs=-1  
    )


    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)

    # Plot the learning curves.  Learning curve code refer from open source.
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.15)

    plt.plot(train_sizes, validation_mean, label='Validation score', color='green', marker='o')
    plt.fill_between(train_sizes, validation_mean - validation_std, validation_mean + validation_std, color='green', alpha=0.15)

    plt.title('Learning Curve')
    plt.xlabel('Training Data Size')
    plt.ylabel('Neg_log_loss')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()


loss_average = loss_total / 11
f1_average = f1_total / 11

print(f"Average loss of {x_file} is {loss_average}" )
print(f"Average f1_score of {x_file} is {f1_average}" )


# Tuning process to get best parameter of max_iter
max_iter_list = []

for i in range (0,11):    
    x_file = f'data_set/knn/X_relevant_label_{i+1}.npy'
    x_data = np.load(x_file)    
    x_test = x_data
    y_test = y_data[:,i]    
    x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.2, random_state=42)

    params_max_iter = {'max_iter': list(range(5000,10000,1000))}
    parameter_grids = [params_max_iter]

    results = []
    for param_grid in parameter_grids:
        grid_search = GridSearchCV(LogisticRegression(multi_class='multinomial', n_jobs=-1, class_weight = {0: 1, 1: 15}), param_grid, cv=5, scoring='neg_log_loss', verbose=1)
        grid_search.fit(x_train, y_train)  
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_score = -grid_search.best_score_  
        y_pred = best_model.predict(x_test)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Tuned Parameter': list(best_params.keys()),
            'Best Parameters': best_params,
            'Best Log Loss': best_score,
            'F1 Score': f1
        })
        
        max_iter_list.append(best_params)

    print(results)
    print(classification_report(y_test, y_pred))

print(max_iter_list)


# Tuning process to get best parameter of C

c_list = []

for i in range (0,11):
    
    x_file = f'data_set/knn/X_relevant_label_{i+1}.npy'
    x_data = np.load(x_file)    
    x_test = x_data
    y_test = y_data[:,i]   
    x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.2, random_state=42)
    
    params_C = {'C': np.logspace(-4, 4, 10)}
    parameter_grids = [params_C]

    results = []
    
    for param_grid in parameter_grids:
        grid_search = GridSearchCV(LogisticRegression(multi_class='multinomial', n_jobs=-1, class_weight = {0: 1, 1: 15}, max_iter = (max_iter_list[i])['max_iter']), param_grid, cv=5, scoring='neg_log_loss', verbose=1)
        grid_search.fit(x_train, y_train)  
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_score = -grid_search.best_score_  

        y_pred = best_model.predict(x_test)
        f1 = f1_score(y_test, y_pred)

        results.append({
            'Tuned Parameter': list(best_params.keys()),
            'Best Parameters': best_params,
            'Best Log Loss': best_score,
            'F1 Score': f1
        })
        
        c_list.append(best_params)

    print(results)
    print(classification_report(y_test, y_pred))
    
print(c_list)


# Tuning process to get best parameter of penalty

penalty_list = []

for i in range (0,11):    
    x_file = f'data_set/knn/X_relevant_label_{i+1}.npy'
    x_data = np.load(x_file)    
    x_test = x_data
    y_test = y_data[:,i]    
    x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.2, random_state=42)

    params_penalty = {'penalty': ['l1', 'l2']}
    parameter_grids = [params_penalty]

    results = []
    
    for param_grid in parameter_grids:
        grid_search = GridSearchCV(LogisticRegression(multi_class='multinomial', n_jobs=-1, class_weight = {0: 1, 1: 15}, max_iter = (max_iter_list[i])['max_iter'], C = (c_list[i])['C']), param_grid, cv=5, scoring='neg_log_loss', verbose=1)
        grid_search.fit(x_train, y_train)  
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_score = -grid_search.best_score_  
        y_pred = best_model.predict(x_test)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Tuned Parameter': list(best_params.keys()),
            'Best Parameters': best_params,
            'Best Log Loss': best_score,
            'F1 Score': f1
        })
        
        penalty_list.append(best_params)

    print(results)
    print(classification_report(y_test, y_pred))
    
print(penalty_list)


# Tuning process to get best parameter of solver

solver_list = []

for i in range (0,11):
    x_file = f'data_set/knn/X_relevant_label_{i+1}.npy'
    x_data = np.load(x_file)
    x_test = x_data
    y_test = y_data[:,i]
    x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.2, random_state=42)

    params_solver = {'solver': ['liblinear', 'lbfgs', 'saga']}
    parameter_grids = [params_solver]

    results = []
    for param_grid in parameter_grids:
        grid_search = GridSearchCV(LogisticRegression(multi_class='multinomial', n_jobs=-1, class_weight = {0: 1, 1: 15}, max_iter = (max_iter_list[i])['max_iter'], C = (c_list[i])['C'], penalty = (penalty_list[i])['penalty']), param_grid, cv=5, scoring='neg_log_loss', verbose=1)
        grid_search.fit(x_train, y_train)  
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_score = -grid_search.best_score_  
        y_pred = best_model.predict(x_test)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Tuned Parameter': list(best_params.keys()),
            'Best Parameters': best_params,
            'Best Log Loss': best_score,
            'F1 Score': f1
        })
        
        solver_list.append(best_params)

    print(results)
    print(classification_report(y_test, y_pred))
    
print(solver_list)


# Final model for each symbol
f1_total = 0
loss_total = 0

for i in range (0,11):
    
    x_file = f'data_set/knn/X_relevant_label_{i+1}.npy'
    x_data = np.load(x_file)
    x_test = x_data
    y_test = y_data[:,i] 
    x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.2, random_state=42)
    
    model = LogisticRegression(multi_class='multinomial', n_jobs=-1, class_weight = {0: 1, 1: 15}, max_iter = (max_iter_list[i])['max_iter'], C = (c_list[i])['C'], penalty = (penalty_list[i])['penalty'], solver = (solver_list[i])['solver'])
    model.fit(x_train, y_train)
    y_prob = model.predict_proba(x_test)[:, 1]
    initial_log_loss = log_loss(y_test, y_prob)
    loss_total += initial_log_loss
    print(f"Initial Binary Cross Entropy Loss {i}: {initial_log_loss}")
    
    y_pred_1 = model.predict(x_test)
    f1_1 = f1_score(y_test, y_pred_1)
    f1_total += f1_1

    print(f"f1 score {i}: {f1_1}")
    #print(classification_report(y_test, y_pred_1))
    
    # Learning curve code refer from open source.    
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=model, 
        X=x_test, 
        y=y_test, 
        train_sizes=np.linspace(0.1, 1.0, 20),
        cv=5,  
        scoring='neg_log_loss',  
        n_jobs=-1  
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.15)

    plt.plot(train_sizes, validation_mean, label='Validation score', color='green', marker='o')
    plt.fill_between(train_sizes, validation_mean - validation_std, validation_mean + validation_std, color='green', alpha=0.15)

    plt.title('Learning Curve')
    plt.xlabel('Training Data Size')
    plt.ylabel('Neg_log_loss')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.show()


loss_average = loss_total / 11
f1_average = f1_total / 11

print(f"Average loss of {x_file} is {loss_average}" )
print(f"Average f1_score of {x_file} is {f1_average}" )

