import pandas as pd
import numpy as np

credits=pd.read_csv('/content/tmdb_5000_credits.csv')
movies_df=pd.read_csv('/content/tmdb_5000_movies.csv')

credits_column_renamed = credits.rename(index=str,columns={"movie_id":"id"})
movies_df_merge = movies_df.merge(credits_column_renamed, on='id')
movies_df_merge.head()

movies_cleaned_df = movies_df_merge.drop(columns=['homepage','title_x','title_y','status','production_countries'])
movies_cleaned_df.head()

from sklearn.feature_extraction.text import TfidfVectorizer


tfv = TfidfVectorizer(min_df=3, max_features=None,
                     strip_accents='unicode',analyzer='word',token_pattern='\w{1,}',
                     ngram_range=(1,3),
                     stop_words ='english')

movies_cleaned_df['overview'] = movies_cleaned_df['overview'].fillna('')

tfv.matrix = tfv.fit_transform(movies_cleaned_df['overview'])

from sklearn.metrics.pairwise import sigmoid_kernel
#compute sigmoid_kernal
sig=sigmoid_kernel(tfv.matrix,tfv.matrix)

#reverse mapping of indices and movie titiles
indices = pd.Series(movies_cleaned_df.index,index=movies_cleaned_df['original_title']).drop_duplicates()

def give_rec(title,sig=sig):
  #getting index with respect to original title
  idx = indices[title]

  #get the pair wise similarty scores
  sig_scores =list(enumerate(sig[idx]))

  #sort movies
  sig_scores=sorted(sig_scores,key=lambda x:x[1],reverse=True)

  #scores of 10 movies
  sig_scores=sig_scores[1:11]

  #movies indices
  moive_indices=[i[0] for i in sig_scores]

  #10 similar movies
  return movies_cleaned_df['original_title'].iloc[moive_indices]
  
  give_rec('Spy Kids')
