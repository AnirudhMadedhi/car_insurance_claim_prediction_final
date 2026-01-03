
# EDA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/train.csv')
print(df.info())
sns.countplot(x='is_claim', data=df)
plt.show()
