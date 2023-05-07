
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from pylab import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from bayes_opt import BayesianOptimization    # pip install bayesian-optimization
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sko.GA import GA
import warnings
warnings.filterwarnings("ignore")

font1 = {'family': 'Times New Roman','weight': 'normal','size': 13,}
font2 = {'family': 'STSong','weight': 'normal','size': 13,}
fontsize1=13

# 设置字体，以作图显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus']=False
# 设置显示属性
pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',100)
np.set_printoptions(suppress=True)
pd.set_option('precision',3)
np.set_printoptions(precision=4)




def figure_plot(predict, true_value, figure_property,key_label=None):
    # 折线图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(predict, '-*', label='预测值')
    ax.plot(true_value, '-s', label='真实值')
    # x_ticks = ax.set_xticks([i for i in range(len(key_label))])
    # x_labels = ax.set_xticklabels(key_label,rotation=45,fontdict=font1)
    ax.set_title(figure_property['title'], fontdict=font2)
    ax.set_xlabel(figure_property['X_label'], fontdict=font2)
    ax.set_ylabel(figure_property['Y_label'], fontdict=font2)
    plt.tick_params(labelsize=12)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # #     y_ticks=ax.set_yticks([])
    # #     y_labels=ax.set_yticklabels([-20+i for i in range(20)],rotation=0,fontsize=14)
    plt.legend(prop=font2)
    plt.tight_layout()
    plt.savefig('../fig/{}.jpg'.format(figure_property['title']), dpi=500, bbox_inches='tight')  # 保存图片
    plt.show()

def figure_plot_1(predict, true_value, figure_property,key_label=None):
    # 折线图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(true_value,predict, '-*')
    # x_ticks = ax.set_xticks([i for i in range(len(key_label))])
    # x_labels = ax.set_xticklabels(key_label,rotation=45,fontdict=font1)
    ax.set_title(figure_property['title'], fontdict=font2)
    ax.set_xlabel(figure_property['X_label'], fontdict=font2)
    ax.set_ylabel(figure_property['Y_label'], fontdict=font2)
    plt.tick_params(labelsize=12)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    # #     y_ticks=ax.set_yticks([])
    # #     y_labels=ax.set_yticklabels([-20+i for i in range(20)],rotation=0,fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../fig/{}.jpg'.format(figure_property['title']), dpi=500, bbox_inches='tight')  # 保存图片
    plt.show()



def Calculate_Regression_metrics(true_value, predict, label='训练集'):
    mse = mean_squared_error(true_value, predict)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_value, predict)
    r2 = r2_score(true_value, predict)
    ex_var = explained_variance_score(true_value, predict)
    mape = mean_absolute_percentage_error(true_value, predict)
    train_result = pd.DataFrame([mse, rmse, mae, r2,ex_var,mape], columns=[label],
                                index=['mse', 'rmse', 'mae', 'r2','ex_var','mape']).T
    return train_result


'''============================ KNN 调参 ============================
// GridSearchCV 网格搜索
// 随机搜索 RandomizedSearchCV
// 贝叶斯优化 （BayesianOptimization）
// 优化算法
===================================================================='''

# TODO: 1.加载数据
'''============================ 1. 加载数据 ============================'''
data = pd.read_excel('../data/所有数据.xlsx', header=0,index_col='时间')


#
# data = pd.read_excel('../data/所有数据.xlsx', header=0).drop('时间',axis=1)#
# print('数据形状：{}'.format(data.shape))
# mean = data.mean()
# std = data.std()
# range_low = mean-5*std
# range_high = mean+5*std  #5,8
# print(range_high.shape)
# new_data = data
# num=0
# for i in range(len(data)):  #行
#     for j in range(data.shape[1]):
#         if range_low.values[j] > data.iloc[i,j] or data.iloc[i,j] > range_high.values[j]:
#             print('i',i)
#             new_data = new_data.drop([i],axis=0)
#             num = num+1
#             print('num:',num)
#             break
# data = new_data
# print('数据形状：{}'.format(data.shape))

array = data.values
X = array[:, :-2]
Y = array[:, -2]

# TODO: 2. 数据集划分
'''============================ 2. 数据集划分 ==========================
// 训练集：测试集 == 80%：20%
// 训练集-> K折交叉验证调参
// 选择最优参数后模型训练，测试集测试
======================================================================'''
# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=40)  #, random_state=0
print('数据形状：X_train:{}，X_test:{},Y_train:{},Y_test:{}'.format(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape))


# TODO: 3. 优化前模型计算结果
'''============================ 3. 优化前模型计算结果 =========================='''
#  1. 定义模型
pipelines = {}
pipelines['KNN'] = Pipeline([('Minmax', MinMaxScaler()),('knn',KNeighborsRegressor())])

for algorithm in pipelines:
    clf = pipelines[algorithm].fit(X_train, Y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)

    '''================计算模型计算训练集结果=============='''
    train_result = Calculate_Regression_metrics(Y_train, train_predict, label='训练集')
    title = '{}算法训练集结果对比'.format(algorithm)
    figure_property = {'title': title, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
    figure_plot(train_predict, Y_train, figure_property)

    '''================计算模型计算测试集结果=============='''
    test_result =  Calculate_Regression_metrics(Y_test, test_predict, label='测试集')
    title = '{}算法测试集结果对比'.format(algorithm)
    figure_property = {'title': title, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
    figure_plot(test_predict, Y_test, figure_property)

    '''===================保存计算结果================'''
    result = pd.concat([train_result, test_result], axis=0)
    print('\n {}算法计算结果'.format(algorithm))
    print(result)
    # result.to_excel('../result/{}算法计算结果.xlsx'.format(algorithm))


# TODO: 4. 参数优化
'''============================ 4. 参数优化 ==========================
// GridSearchCV 网格搜索
// 随机搜索 RandomizedSearchCV
// 贝叶斯优化 （BayesianOptimization）
// 优化算法  （GA）
===================================================================='''

# ==================定义优化模型====================
class ParameterOptimization:
    def __init__(self,x_train,y_train,num_folds=10,scoring='neg_mean_squared_error'):
        self.x_train = x_train
        self.y_train = y_train
        self.kfold = KFold(n_splits=num_folds, shuffle=True)
        self.scoring= scoring
        self.num_folds = num_folds

    '''GridSearchCV 网格搜索'''
    def grid_search_cv(self,model,param_grid):
        grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = self.scoring, cv = self.kfold)
        grid_result = grid.fit(X = self.x_train,y = self.y_train)
        print('\nGridSearchCV 网格搜索')
        print('最优：%s \t 使用: %s' % (grid_result.best_score_, grid_result.best_params_))
        cv_results = zip(grid_result.cv_results_['mean_test_score'],
                         grid_result.cv_results_['std_test_score'],
                         grid_result.cv_results_['params'])
        for mean, std, param in cv_results:
            print('mean_test_score:%f \t std_test_score:(%f)\t with\t params:%r' % (mean, std, param))
        return grid_result.best_params_


    ''' 随机搜索 RandomizedSearchCV'''
    def randomized_search_cv(self,model,param_grid,n_iter_search):
        grid = RandomizedSearchCV(estimator = model, param_distributions = param_grid,n_iter=n_iter_search,scoring = self.scoring, cv = self.kfold, n_jobs = 1)
        grid_result = grid.fit(X_train,Y_train)
        print('\n随机搜索 RandomizedSearchCV')
        print('最优：%s\t  使用:%s' % (grid_result.best_score_, grid_result.best_params_))
        cv_results = zip(grid_result.cv_results_['mean_test_score'],
                         grid_result.cv_results_['std_test_score'],
                         grid_result.cv_results_['params'])
        for mean, std, param in cv_results:
            print('mean_test_score:%f \t std_test_score:(%f)\t with\t params:%r' % (mean, std, param))
        return grid_result.best_params_

    '''贝叶斯优化 （BayesianOptimization）'''

    # 贝叶斯优化模型
    def bayesian_model_cv(self, n_neighbors):
        model = Pipeline([('Minmax', MinMaxScaler()), ('knn', KNeighborsRegressor(int(n_neighbors)))])
        val = cross_val_score(model, self.x_train,self.y_train, scoring=self.scoring, cv=self.kfold,n_jobs=1).mean()
        return val

    # 参数开采
    def bayesian_optimization(self,n_iter,param_grid):
        self.bay_bo = BayesianOptimization(self.bayesian_model_cv,param_grid)
        self.bay_bo.maximize(n_iter)
        print(f'explore:{self.bay_bo.max}')
        return self.bay_bo.max['params']

    # 参数微调
    def bayesian_optimization_finetune(self,param_gird):
        self.bay_bo.probe(params=param_gird,lazy=True)
        self.bay_bo.maximize(init_points=0, n_iter=10)
        print(f'fine_tune:{self.bay_bo.max}')
        return self.bay_bo.max['params']

    '''GA优化'''
    # 定义优化函数
    def function(self,x):
        model = Pipeline([('Minmax', MinMaxScaler()), ('knn', KNeighborsRegressor(n_neighbors=int(x)))])
        val = cross_val_score(model, self.x_train, self.y_train, scoring=self.scoring, cv=self.kfold, n_jobs=1).mean()
        return -val

    def ga_optimization(self,x_upper, x_bound,size_pop=50, max_iter=100):
        ga = GA(func=self.function, lb=x_upper, ub=x_bound,size_pop=size_pop,max_iter=max_iter,n_dim=len(x_upper))
        best_x, best_y = ga.run()
        print(f'ga_optimization best params ：{best_x}, \t best fitness：{best_y}')
        Y_history = pd.DataFrame(ga.all_history_Y)
        Y_history.min(axis=1).cummin().plot(kind='line')
        plt.ylabel('fitness')
        plt.xlabel('iteration')
        plt.savefig('./fig/{}.jpg'.format('适应度曲线'), dpi=500, bbox_inches='tight')  # 保存图片
        plt.show()
        return best_x


# 评估算法 - 评估标准
num_folds = 10
scoring = 'neg_mean_squared_error'
# scoring = 'neg_mean_absolute_error'
# scoring ='r2'

'''==================四种寻优方式可选择=================='''

# 网格搜索与随机搜索模型
param_grid = {'knn__n_neighbors': np.arange(1, 100)}
model = Pipeline(steps=[('Minmax', MinMaxScaler()), ('knn', KNeighborsRegressor())])

# 初始化
knnPO= ParameterOptimization(x_train=X,y_train=Y,num_folds=num_folds,scoring= 'neg_mean_absolute_error')#x_train=X_train,y_train=Y_train

# 网格搜索
grid_search_best_params = knnPO.grid_search_cv(model=model,param_grid=param_grid)

# 随机搜索模型
randomized_search_best_params = knnPO.randomized_search_cv(model=model,param_grid=param_grid,n_iter_search=100)

# 贝叶斯优化模型
param_grid1 = {'n_neighbors': (1, 100)}
bayesian_optimization_best_params = knnPO.bayesian_optimization(n_iter=10,param_grid=param_grid1)                 # 粗调
bayesian_optimization_finetune_params = knnPO.bayesian_optimization_finetune(bayesian_optimization_best_params)   # 微调，可对粗调的多个参数进一步寻优

# GA优化模型
X_bound = [200]   # 不能超过样本数
X_upper = [1]
ga_optimization_best_params = knnPO.ga_optimization(x_upper=X_upper, x_bound=X_bound,size_pop=30, max_iter=100)

print(f'\n网格搜索:{grid_search_best_params}')
print(f'随机搜索:{randomized_search_best_params}')
print(f'贝叶斯粗调:{bayesian_optimization_best_params}')
print(f'贝叶斯微调:{bayesian_optimization_finetune_params}')
print(f'GA优化:{ga_optimization_best_params[0]}')


# TODO: 5. 参数优化
'''================== 5. 参数优化后模型输出 ==========================
// GridSearchCV 网格搜索
// 随机搜索 RandomizedSearchCV
// 贝叶斯优化 （BayesianOptimization）
// 优化算法 （GA）
===================================================================='''

#  1. 定义模型
pipelines = {}
pipelines['调参之前'] = Pipeline([('Minmax', MinMaxScaler()),('knn',KNeighborsRegressor())])
pipelines['GridSearch'] = Pipeline([('Minmax', MinMaxScaler()),('knn',KNeighborsRegressor(n_neighbors=grid_search_best_params['knn__n_neighbors']))])
pipelines['RandomizedSearch'] = Pipeline([('Minmax', MinMaxScaler()),('knn',KNeighborsRegressor(n_neighbors=randomized_search_best_params['knn__n_neighbors']))])
pipelines['BayesianOptimization_explore'] = Pipeline([('Minmax', MinMaxScaler()),('knn',KNeighborsRegressor(n_neighbors=int(bayesian_optimization_best_params['n_neighbors'])))])
pipelines['BayesianOptimization_finetune'] = Pipeline([('Minmax', MinMaxScaler()),('knn',KNeighborsRegressor(n_neighbors=int(bayesian_optimization_finetune_params['n_neighbors'])))])
pipelines['ga_optimization'] =  Pipeline([('Minmax', MinMaxScaler()), ('knn', KNeighborsRegressor(n_neighbors=int(ga_optimization_best_params)))])

for algorithm in pipelines:
    clf = pipelines[algorithm].fit(X_train, Y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)

    '''================计算模型计算训练集结果=============='''
    train_result = Calculate_Regression_metrics(Y_train, train_predict, label='训练集')
    title = '{}算法训练集结果对比'.format(algorithm)
    figure_property = {'title': title, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
    figure_plot(train_predict, Y_train, figure_property)

    '''================计算模型计算测试集结果=============='''
    test_result =  Calculate_Regression_metrics(Y_test, test_predict, label='测试集')
    title = '{}算法测试集结果对比'.format(algorithm)
    figure_property = {'title': title, 'X_label': 'Prediction set samples', 'Y_label': 'Prediction Value'}
    figure_plot(test_predict, Y_test, figure_property)

    '''===================保存计算结果================'''
    result = pd.concat([train_result, test_result], axis=0)
    print('\n {}算法计算结果'.format(algorithm))
    print(result)


'''=====================================总结=========================
// 分析优化后的结果可以看出，整体上优化后结果较好
// 但也存在个别优化后结果不如优化前的情况，造成这种情况的原因主要是：
// 交叉验证部分将训练集划分训练集和验证集，以此参数寻优，
// 最终建模时使用事先划分的训练集建模，以此造成了建模数据不同，寻优参数也不一定适用
//解决方法：
// 1) 增加K折交叉验证次数，num_folds，当num_folds较大时，训练集和交叉验证训练集可等同
// 2) 或者采用全部数据K折寻参，然后训练集建模
====================================================================='''