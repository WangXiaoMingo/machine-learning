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

# 4. LightGBM示例
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

