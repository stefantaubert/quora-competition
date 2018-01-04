import pandas 
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import data_paths

df_train = pandas.read_csv(data_paths.train)
df_test = pandas.read_csv(data_paths.test)

def show_cloud(df):
	all_questions = df['question1'].tolist() + df['question2'].tolist()
	qs = pandas.Series(all_questions).astype(str)

	#cloud = WordCloud(width=2400, height=800, background_color='white', colormap='Greys_r').generate(" ".join(qs.astype(str)))
	cloud = WordCloud(width=2400, height=1500, background_color='white').generate(" ".join(qs.astype(str)))
	plt.figure(figsize=(12, 4), facecolor='white')
	plt.imshow(cloud)
	plt.axis('off')
	plt.tight_layout(pad=0)
	plt.show()

#show_cloud(df_test)
show_cloud(df_train)
