
import pandas as pd, numpy as np, matplotlib.pyplot as plt, time
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge

train_df = pd.read_csv('mnist_train.csv')
test_df = pd.read_csv('mnist_test.csv')

def extract_by_label(df,l1:int,l2:int) -> tuple:
    '''filter MNIST df to 2 labels'''
    df = df.loc[df['label'].isin([l1,l2])]
    X = df.iloc[:,1:]
    y = df.iloc[:,0]
    return np.array(X), np.array(y)

def subsample(X, y, n:int):
    '''truncate dataframe'''
    return X[:n,:], y[:n]

def generate_data(l1, l2, small_sample):
    '''turns dataframe of entire MNIST and retuns np arrays filtered to 2 labels'''
    X_tr, y_tr = extract_by_label(train_df, l1, l2)
    X_te, y_te = extract_by_label(test_df, l1, l2)
    if small_sample:
        X_tr, y_tr = subsample(X_tr, y_tr, small_sample)
        #X_te, y_te = subsample(X_te, y_te, small_sample//5)
    return X_tr, X_te, y_tr, y_te

def error(y_true, y_pred):
    '''(fp+fn)/(fp+fn+tp+tn)'''
    N = np.size(y_true, axis=0)
    error_count = np.count_nonzero(y_true != y_pred)
    return error_count/N


def run_model_svc(l1,l2, kernel, sample):
    '''linear SVM'''
    Xtr, Xte, ytr, yte = generate_data(l1, l2, sample)
    model = SVC(kernel=kernel, degree=2)
    model.fit(Xtr,ytr)
    ytr_pred = model.predict(Xtr)
    yte_pred = model.predict(Xte)
    return error(ytr, ytr_pred), error(yte, yte_pred)

def ghetto_softmax(y,l1,l2):
    '''naive binary activator'''
    return np.array([l1 if abs(y[i]-l1)<abs(y[i]-l2) else l2 for i in range(len(y))])


def run_model_ridge(l1,l2, kernel, sample):
    '''kernel ridge regression'''
    Xtr, Xte, ytr, yte = generate_data(l1, l2, sample)
    model = KernelRidge(alpha=0.01, kernel=kernel, degree=2)
    model.fit(Xtr,ytr)
    ytr_pred_raw = model.predict(Xtr)
    yte_pred_raw = model.predict(Xte)

    ytr_pred = ghetto_softmax(ytr_pred_raw, l1, l2)
    yte_pred = ghetto_softmax(yte_pred_raw, l1, l2)

    return error(ytr, ytr_pred), error(yte, yte_pred)

    return

def main():
    samples = [1000, 2000, 4000, 6000, 8000, 10000, 12000]
    #samples = [1000,2000,4000,6000]
    kernels = ['linear', 'poly']
    #kernels = ['rbf']
    labels = [(1,9), (3,8), (1,7), (5,2)]
    for label in labels:
        l1 = label[0]
        l2 = label[1]
        for kernel in kernels:
            start = time.time()
            print('Working on {},{}'.format(l1,l2))
            tr_errs = []
            te_errs = []
            for sample in samples:
                print('Working on {}, {}'.format(kernel,sample))
                e_tr, e_te = run_model_ridge(l1, l2, kernel, sample)
                tr_errs.append(e_tr)
                te_errs.append(e_te)
            plt.plot(samples, tr_errs, label='Train error')
            plt.plot(samples, te_errs, label='Test error')
            plt.xlabel('Sample size')
            plt.ylabel('Error')
            plt.legend()
            plt.title('{}: {},{}'.format(kernel,l1, l2))
            plt.savefig('/home/patrick/Documents/AMATH_563/HW/2/plots/skl_kernelridge/{}{}{}.png'.format(kernel,l1,l2))
            plt.clf()
            end = time.time()
            print('{} seconds'.format(end-start))
    return

#main()