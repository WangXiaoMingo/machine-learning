# 1. machine-learning
常见机器学习算法回归、分类应用示例，调参；包括基础的线性回归算法、集成学习、支持向量机等；调参包括网格搜索、随机搜索、贝叶斯优化、优化算法如GA优化等。
需安装依赖库：

pip install scikit-opt

pip install bayesian-optimization

pip install scikit-learn

pip install lightgbm

# 2. 算法包括以下（分类、回归）：
## 线性回归算法
1）线性回归：LinearRegression

2）岭回归：Ridge

3）套索回归：Least absolute shrinkage and selection operator：Lasso

4）弹性网络：ElasticNet

5）贝叶斯回归：BayesianRidge

## 支持向量机
1） SVR

## Nearest Neighbors
1） KNN

## 集成学习：Ensemble methods
1）DecisionTreeRegressor：CART

2）RandomForestRegressor：RF

3）BaggingRegressor

4）AdaBoostRegressor

5）ExtraTreesRegressor

6）GradientBoostingRegressor

7）StackingRegressor

8）VotingRegressor

9）lightgbm

# 3. 算法调参
--------------------------------------------------------
// 网格搜索 GridSearchCV 

// 随机搜索 RandomizedSearchCV

// 贝叶斯优化 （BayesianOptimization）

// 优化算法： GA
--------------------------------------------------------
// 训练集：测试集 == 80%：20%

// 训练集-> K折交叉验证调参

// 选择最优参数后模型训练，测试集测试
---------------------------------------------------------
// 四种寻优方式可选择

// grid_search_cv: 网格搜索

// randomized_search_cv：随机搜索

// bayesian_optimization：贝叶斯优化

// ga_optimization：GA优化

// 指定相应方法优化run_flag，

// 1-全部优化，2-网格搜索，3-随机搜索，4贝叶斯优化，5-GA优化
---------------------------------------------------------

# 4. LightGBM示例，核心算法
==================定义优化模型====================
class ParameterOptimization:
    def __init__(self,x_train,y_train,kfold,scoring='neg_mean_squared_error'):
        self.x_train = x_train
        self.y_train = y_train
        self.kfold = kfold
        self.scoring= scoring
        self.num_folds = num_folds

    '''GridSearchCV 网格搜索'''
    def grid_search_cv(self,model,param_grid):
        grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = self.scoring, cv = self.kfold,n_jobs=1)
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
    # 定义贝叶斯优化模型
    def bayesian_model_cv(self,n_estimators,learning_rate,subsample,max_depth,min_data_in_leaf,num_leaves,max_bin):
        model = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse',n_estimators=int(n_estimators),learning_rate=learning_rate,subsample=subsample,max_depth=int(max_depth),min_data_in_leaf=int(min_data_in_leaf),num_leaves=int(num_leaves),max_bin=int(max_bin)))])
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
    def ga_optimization(self,x_upper, x_bound,size_pop, max_iter):
        ga = GA(func=function, size_pop=size_pop,max_iter=max_iter,n_dim=len(x_upper),lb=x_upper, ub=x_bound)
        best_x, best_y = ga.run()
        print(f'ga_optimization best params ：{best_x}, \t best fitness：{best_y}')
        Y_history = pd.DataFrame(ga.all_history_Y)
        Y_history.min(axis=1).cummin().plot(kind='line')
        plt.ylabel('fitness')
        plt.xlabel('iteration')
        plt.savefig('./fig/{}.jpg'.format('适应度曲线'), dpi=500, bbox_inches='tight')  # 保存图片
        plt.show()
        return best_x

# 5.完整实例


import pandas as pd
from pylab import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,explained_variance_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from bayes_opt import BayesianOptimization    # pip install bayesian-optimization
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sko.GA import GA   #pip install scikit-opt
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


'''============================ lightgbm 调参 ============================
// GridSearchCV 网格搜索
// 随机搜索 RandomizedSearchCV
// 贝叶斯优化 （BayesianOptimization）
// 优化算法
===================================================================='''

# TODO: 1.加载数据
'''============================ 1. 加载数据 ============================'''
train_data = pd.read_excel('../window30/train_data_7.xlsx', header=0,index_col='时间')
test_data = pd.read_excel('../window30/test_data_8_1.xlsx', header=0, index_col='时间')

# TODO: 2. 数据集划分
'''============================ 2. 数据集划分 ==========================
// 训练集：测试集 == 80%：20%
// 训练集-> K折交叉验证调参
// 选择最优参数后模型训练，测试集测试
======================================================================'''

X_train = train_data.iloc[:,:-1].values
Y_train = train_data.iloc[:,-1].values

X_test = test_data.iloc[:,:-1].values
Y_test = test_data.iloc[:,-1].values
print('数据形状：X_train:{}，X_test:{},Y_train:{},Y_test:{}'.format(X_train.shape, X_test.shape,
                                                                              Y_train.shape, Y_test.shape))

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=40)  #, random_state=0

# TODO: 3. 优化前模型计算结果
'''============================ 3. 优化前模型计算结果 =========================='''
#  1. 定义模型
pipelines = {}
pipelines['lightgbm'] = Pipeline([('lightgbm',lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse'))])
pipelines['Stand_lightgbm'] = Pipeline([('Stand', StandardScaler()),('lightgbm',lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse'))])



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
// 标准化下rbf核函数效果相对较优
// GridSearchCV 网格搜索
// 随机搜索 RandomizedSearchCV
// 贝叶斯优化 （BayesianOptimization）
// 优化算法  （GA）
===================================================================='''


# ==================定义优化模型====================
class ParameterOptimization:
    def __init__(self,x_train,y_train,kfold,scoring='neg_mean_squared_error'):
        self.x_train = x_train
        self.y_train = y_train
        self.kfold = kfold
        self.scoring= scoring
        self.num_folds = num_folds

    '''GridSearchCV 网格搜索'''
    def grid_search_cv(self,model,param_grid):
        grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = self.scoring, cv = self.kfold,n_jobs=1)
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
    # 定义贝叶斯优化模型
    def bayesian_model_cv(self,n_estimators,learning_rate,subsample,max_depth,min_data_in_leaf,num_leaves,max_bin):
        model = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse',n_estimators=int(n_estimators),learning_rate=learning_rate,subsample=subsample,max_depth=int(max_depth),min_data_in_leaf=int(min_data_in_leaf),num_leaves=int(num_leaves),max_bin=int(max_bin)))])
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
    def ga_optimization(self,x_upper, x_bound,size_pop, max_iter):
        ga = GA(func=function, size_pop=size_pop,max_iter=max_iter,n_dim=len(x_upper),lb=x_upper, ub=x_bound)
        best_x, best_y = ga.run()
        print(f'ga_optimization best params ：{best_x}, \t best fitness：{best_y}')
        Y_history = pd.DataFrame(ga.all_history_Y)
        Y_history.min(axis=1).cummin().plot(kind='line')
        plt.ylabel('fitness')
        plt.xlabel('iteration')
        plt.savefig('./fig/{}.jpg'.format('适应度曲线'), dpi=500, bbox_inches='tight')  # 保存图片
        plt.show()
        return best_x


'''====================================寻参开始==========================================
lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse'learning_rate=0.1)
	                                        
// 1) n_estimators: 也就是弱学习器的最大迭代次数。n_estimators太小，容易欠拟合，n_estimators太大，又容易过拟合，一般选择一个适中的数值。默认是100。
//     在实际调参的过程中，我们常常将n_estimators和learning_rate一起考虑。
// 2) learning_rate: 即每个弱学习器的权重缩减系数ν，也称作步长，ν的取值范围为0<ν≤1。对于同样的训练集拟合效果，较小的ν意味着我们需要更多的弱学习器的迭代次数。
//     通常我们用步长和迭代最大次数一起来决定算法的拟合效果。所以这两个参数n_estimators和learning_rate要一起调参。一般来说，可以从一个小一点的ν开始调参，默认是1。
// 3) subsample: 子采样，取值为(0,1]。注意这里的子采样和随机森林不一样，随机森林使用的是放回抽样，而这里是不放回抽样。如果取值为1，则全部样本都使用，等于没有使用子采样。
//     如果取值小于1，则只有一部分样本会去做lightgbm的决策树拟合。选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。
//     推荐在[0.5, 0.8]之间，默认是1.0，即不使用子采样。
// 1) 划分时考虑的最大特征数max_features: 可以使用很多种类型的值，默认是"None",意味着划分时考虑所有的特征数；
//    如果是"log2"意味着划分时最多考虑log_2N个特征；如果是"sqrt"或者"auto"意味着划分时最多考虑$\sqrt{N}$个特征。
//    如果是整数，代表考虑的特征绝对数。如果是浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。其中N为样本总特征数。
//    一般来说，如果样本特征数不多，比如小于50，我们用默认的"None"就可以了，
//    如果特征数非常多，我们可以灵活使用刚才描述的其他取值来控制划分时考虑的最大特征数，以控制决策树的生成时间。

// 2) 决策树最大深度max_depth: 默认可以不输入，如果不输入的话，默认值是3。一般来说，数据少或者特征少的时候可以不管这个值。
//   如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100之间。

// 4) min_data_in_leaf. 这是处理 leaf-wise 树的过拟合问题中一个非常重要的参数. 它的值取决于训练数据的样本个树和 num_leaves. 将其设置的较大可以避免生成一个过深的树, 但有可能导致欠拟合. 实际应用中, 对于大数据集, 设置其为几百或几千就足够

// 准确率相关
// max_bin、learning_rate、num_iterations、num_leaves

// 过拟合
// 使用较小的 max_bin（默认为255）
// 使用较小的 num_leaves（默认为31）
// 使用 min_data_in_leaf（默认为20） 和 min_sum_hessian_in_leaf（默认为）
// 通过设置 bagging_fraction （默认为1.0）和 bagging_freq （默认为0，意味着禁用bagging，k表示每k次迭代执行一个bagging）来使用 bagging
// 通过设置 feature_fraction（默认为1.0） 来使用特征子抽样
// 使用 lambda_l1（默认为0）, lambda_l2 （默认为0）和 min_split_gain（默认为0，表示执行切分的最小增益） 来使用正则
// 尝试 max_depth 来避免生成过深的树


// 这里选择 n_estimators、learning_rate、max_depth、subsample、min_data_in_leaf、max_bin、num_leaves
=========================================================================================='''

'''==============================step1. 参数设置==========================================
// n_iter_search：随机搜索次数
// n_iter： 贝叶斯优化次数
========================================================================================'''
# 评估算法 - 评估标准
num_folds = 10
scoring = 'neg_mean_squared_error'   # 'neg_mean_absolute_error'，'r2'
kfold = KFold(n_splits=num_folds, shuffle=True)
x_train = X
y_train = Y
n_iter_search = 1000
n_iter = 20

'''=================step2. 网格搜索与随机搜索模型寻优参数和模型设置======================'''
# n_estimators、learning_rate、max_depth、max_leaf_nodes、subsample
param_grid = {'lightgbm__n_estimators':[10,50,100],
              'lightgbm__learning_rate':  [0.001,0.01,0.1,0.5],
              'lightgbm__subsample':  [0.5,0.6,0.7,0.8],
              'lightgbm__max_depth':  [5,10,50],
              'lightgbm__min_data_in_leaf':[2,4,10],
              'lightgbm__num_leaves':[2,10,30,50,100,1000],
              'lightgbm__max_bin':[100,200,255,500]
              }


model = Pipeline(steps=[('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse',))])


'''=================step3. 贝叶斯优化模型参数设置==========================================
==========================================================================================='''
param_grid_bayes = {'n_estimators':(1,1000),
              'learning_rate':  (0.00001,1),
              'subsample':  (0.5,0.8),
              'max_depth':  (2,50),
              'min_data_in_leaf':(2,50),
              'num_leaves':(2,1000),
              'max_bin':(10,500)
              }

'''=================step4. GA寻优参数和模型设置============================================
// 2个参数分别对应：C、gamma
// 优化函数定义：function
// size_pop：种群规模
// max_iter: 进化次数
// n_estimators,learning_rate,subsample,max_depth,min_data_in_leaf,num_leaves,max_bin
==========================================================================================='''
# GA优化模型上下限
X_bound = [1000, 1,0.8,50,50,1000,500]
X_upper = [1,0.0001,0.5,2,2,2,10]
size_pop = 30
max_iter = 100

# 定义优化函数
def function(x):
    model = Pipeline([('Minmax', MinMaxScaler()),('lightgbm',  lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse',n_estimators=int(x[0]),learning_rate=x[1],subsample=x[2],max_depth=int(x[3]),min_data_in_leaf=int(x[4]),num_leaves=int(x[5]),max_bin=int(x[6])))])
    val = cross_val_score(model, X_train, Y_train, scoring=scoring, cv=kfold, n_jobs=1).mean()
    return -val

'''==============================step5. 模型训练============================
// 四种寻优方式可选择
// grid_search_cv: 网格搜索
// randomized_search_cv：随机搜索
// bayesian_optimization：贝叶斯优化
// ga_optimization：GA优化
// 指定相应方法优化run_flag，
// 1-全部优化，2-网格搜索，3-随机搜索，4贝叶斯优化，5-GA优化
=============================================================================='''

run_flag = 4    # 指定优化方法

# 初始化
lightgbmPO= ParameterOptimization(x_train=x_train,y_train=y_train,kfold=kfold)

if run_flag == 1:
    # 网格搜索
    grid_search_best_params = lightgbmPO.grid_search_cv(model=model,param_grid=param_grid)

    # 随机搜索模型
    randomized_search_best_params = lightgbmPO.randomized_search_cv(model=model,param_grid=param_grid,n_iter_search=n_iter_search)

    # 贝叶斯优化
    bayesian_optimization_best_params = lightgbmPO.bayesian_optimization(n_iter=n_iter,param_grid=param_grid_bayes)        # 粗调
    bayesian_optimization_finetune_params = lightgbmPO.bayesian_optimization_finetune(bayesian_optimization_best_params)   # 微调，可对粗调的多个参数进一步寻优

    # GA优化
    ga_optimization_best_params = lightgbmPO.ga_optimization(x_upper=X_upper, x_bound=X_bound,size_pop=size_pop, max_iter=max_iter)

    print(f'\n网格搜索:{grid_search_best_params}')
    print(f'随机搜索:{randomized_search_best_params}')
    print(f'贝叶斯粗调:{bayesian_optimization_best_params}')
    print(f'贝叶斯微调:{bayesian_optimization_finetune_params}')
    print(f'GA优化:{ga_optimization_best_params[0]}')

elif run_flag == 2:
    # 网格搜索
    grid_search_best_params = lightgbmPO.grid_search_cv(model=model,param_grid=param_grid)
    print(f'\n网格搜索:{grid_search_best_params}')

elif run_flag == 3:
    # 随机搜索模型
    randomized_search_best_params = lightgbmPO.randomized_search_cv(model=model,param_grid=param_grid,n_iter_search=100)
    print(f'随机搜索:{randomized_search_best_params}')

elif run_flag == 4:
    # 贝叶斯优化
    bayesian_optimization_best_params = lightgbmPO.bayesian_optimization(n_iter=n_iter,param_grid=param_grid_bayes)            # 粗调
    bayesian_optimization_finetune_params = lightgbmPO.bayesian_optimization_finetune(bayesian_optimization_best_params)   # 微调，可对粗调的多个参数进一步寻优
    print(f'贝叶斯粗调:{bayesian_optimization_best_params}')
    print(f'贝叶斯微调:{bayesian_optimization_finetune_params}')

elif run_flag == 5:
    # GA优化
    ga_optimization_best_params = lightgbmPO.ga_optimization(x_upper=X_upper, x_bound=X_bound,size_pop=size_pop, max_iter=max_iter)
    print(f'GA优化:{ga_optimization_best_params[0]}')



# TODO: 5. 参数优化
'''================== 5. 参数优化后模型输出 ==========================
// GridSearchCV 网格搜索
// 随机搜索 RandomizedSearchCV
// 贝叶斯优化 （BayesianOptimization）
// 优化算法 （GA）
============================评价指标=================================
// explained_variance_score:解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差。
// mean_absolute_error:平均绝对误差（Mean Absolute Error，MAE），用于评估预测结果和真实数据集的接近程度的程度,其值越小说明拟合效果越好。
// mean_squared_error:均方差（Mean squared error，MSE），该指标计算的是拟合数据和原始数据对应样本点的误差的平方和的均值，其值越小说明拟合效果越好。
// r2_score:判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差。
===================================================================='''

#  1. 定义模型
if run_flag == 1:
    pipelines = {}
    pipelines['调参之前'] = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse'))])
    pipelines['GridSearch'] = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse',
        n_estimators =int(grid_search_best_params['lightgbm__n_estimators']),
        learning_rate = grid_search_best_params['lightgbm__learning_rate'],
        subsample = grid_search_best_params['lightgbm__subsample'],
        max_depth = int(grid_search_best_params['lightgbm__max_depth']),
        min_data_in_leaf = int(grid_search_best_params['lightgbm__min_data_in_leaf']),
        num_leaves = int(grid_search_best_params['lightgbm__num_leaves']),
        max_bin = int(grid_search_best_params['lightgbm__max_bin'])))])


    pipelines['RandomizedSearch'] = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse',
        n_estimators = int(randomized_search_best_params['lightgbm__n_estimators']),
        learning_rate = randomized_search_best_params['lightgbm__learning_rate'],
        subsample = randomized_search_best_params['lightgbm__subsample'],
        max_depth = int(randomized_search_best_params['lightgbm__max_depth']),
        min_data_in_leaf = int(randomized_search_best_params['lightgbm__min_data_in_leaf']),
        num_leaves = int(randomized_search_best_params['lightgbm__num_leaves']),
        max_bin = int(randomized_search_best_params['lightgbm__max_bin'])))])

    pipelines['BayesianOptimization_explore'] = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse',
        n_estimators = int(bayesian_optimization_best_params['n_estimators']),
        learning_rate = bayesian_optimization_best_params['learning_rate'],
        subsample = bayesian_optimization_best_params['subsample'],
        max_depth = int(bayesian_optimization_best_params['max_depth']),
        min_data_in_leaf = int(bayesian_optimization_best_params['min_data_in_leaf']),
        num_leaves = int(bayesian_optimization_best_params['num_leaves']),
        max_bin = int(bayesian_optimization_best_params['max_bin'])))])

    pipelines['BayesianOptimization_finetune'] = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse',
        n_estimators = int(bayesian_optimization_finetune_params['n_estimators']),
        learning_rate = bayesian_optimization_finetune_params['learning_rate'],
        subsample = bayesian_optimization_finetune_params['subsample'],
        max_depth = int(bayesian_optimization_finetune_params['max_depth']),
        min_data_in_leaf = int(bayesian_optimization_finetune_params['min_data_in_leaf']),
        num_leaves = int(bayesian_optimization_finetune_params['num_leaves']),
        max_bin = int(bayesian_optimization_finetune_params['max_bin'])))])

    pipelines['ga_optimization'] = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse',
        n_estimators = int(ga_optimization_best_params[0]),
        learning_rate = ga_optimization_best_params[1],
        subsample = ga_optimization_best_params[2],
        max_depth = int(ga_optimization_best_params[3]),
        min_data_in_leaf = int(ga_optimization_best_params[4]),
        num_leaves = int(ga_optimization_best_params[5]),
        max_bin = int(ga_optimization_best_params[6])))])


elif run_flag == 2:   # 网格
    pipelines = {}
    pipelines['调参之前'] = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse'))])
    pipelines['GridSearch'] = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse',
        n_estimators =int(grid_search_best_params['lightgbm__n_estimators']),
        learning_rate = grid_search_best_params['lightgbm__learning_rate'],
        subsample = grid_search_best_params['lightgbm__subsample'],
        max_depth = int(grid_search_best_params['lightgbm__max_depth']),
        min_data_in_leaf = int(grid_search_best_params['lightgbm__min_data_in_leaf']),
        num_leaves = int(grid_search_best_params['lightgbm__num_leaves']),
        max_bin = int(grid_search_best_params['lightgbm__max_bin'])))])

elif run_flag == 3:  # 随即优化
    pipelines = {}
    pipelines['调参之前'] = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse'))])
    pipelines['RandomizedSearch'] = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse',
        n_estimators = int(randomized_search_best_params['lightgbm__n_estimators']),
        learning_rate = randomized_search_best_params['lightgbm__learning_rate'],
        subsample = randomized_search_best_params['lightgbm__subsample'],
        max_depth = int(randomized_search_best_params['lightgbm__max_depth']),
        min_data_in_leaf = int(randomized_search_best_params['lightgbm__min_data_in_leaf']),
        num_leaves = int(randomized_search_best_params['lightgbm__num_leaves']),
        max_bin = int(randomized_search_best_params['lightgbm__max_bin'])))])


elif run_flag == 4:   # 贝叶斯优化
    pipelines = {}
    pipelines['调参之前'] = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse'))])
    pipelines['BayesianOptimization_explore'] = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse',
        n_estimators = int(bayesian_optimization_best_params['n_estimators']),
        learning_rate = bayesian_optimization_best_params['learning_rate'],
        subsample = bayesian_optimization_best_params['subsample'],
        max_depth = int(bayesian_optimization_best_params['max_depth']),
        min_data_in_leaf = int(bayesian_optimization_best_params['min_data_in_leaf']),
        num_leaves = int(bayesian_optimization_best_params['num_leaves']),
        max_bin = int(bayesian_optimization_best_params['max_bin'])))])

    pipelines['BayesianOptimization_finetune'] = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse',
        n_estimators = int(bayesian_optimization_finetune_params['n_estimators']),
        learning_rate = bayesian_optimization_finetune_params['learning_rate'],
        subsample = bayesian_optimization_finetune_params['subsample'],
        max_depth = int(bayesian_optimization_finetune_params['max_depth']),
        min_data_in_leaf = int(bayesian_optimization_finetune_params['min_data_in_leaf']),
        num_leaves = int(bayesian_optimization_finetune_params['num_leaves']),
        max_bin = int(bayesian_optimization_finetune_params['max_bin'])))])

elif run_flag == 5:      # GA 优化
    pipelines = {}
    pipelines['调参之前'] = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse'))])
    pipelines['ga_optimization'] = Pipeline([('Minmax', MinMaxScaler()), ('lightgbm', lgb.LGBMRegressor(boosting_type = 'gbdt', objective= 'regression', metric = 'rmse',
        n_estimators = int(ga_optimization_best_params[0]),
        learning_rate = ga_optimization_best_params[1],
        subsample = ga_optimization_best_params[2],
        max_depth = int(ga_optimization_best_params[3]),
        min_data_in_leaf = int(ga_optimization_best_params[4]),
        num_leaves = int(ga_optimization_best_params[5]),
        max_bin = int(ga_optimization_best_params[6])))])


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
