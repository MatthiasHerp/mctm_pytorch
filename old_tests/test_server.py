import time
import pandas as pd
import numpy as np
import mlflow
import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns

if __name__ == "__main__":

    mlflow.start_run()

    a=0
    for i in range(2):
        time.sleep(1)
        a +=1
        print(a)
    print(torch.log(torch.tensor(a)))
    b = np.random.normal(0, 1, 100)
    #plt.hist(b)
    #plt.savefig('foo.png')

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 5),
                            gridspec_kw={'wspace': 0.0, 'hspace': 0.0}, sharey=True)
    sns.scatterplot(x=b, y=b, alpha=0.6, color="k", ax=axs)

    print(os.getcwd())
    #os.chdir("/home/mherp/tmp/pycharm_project_287")

    #fig.savefig('fig_test.png')
    #mlflow.log_artifact("./fig_test.png")

    #mlflow.log_figure(fig, "test_fig")

    mlflow.end_run()

    print("done")