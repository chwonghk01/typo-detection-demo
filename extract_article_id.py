import pandas as pd

df = pd.read_parquet('./latest.snappy.parquet').set_index('article_id')
df = df[(df.publish_start_date >= '2019-02-03') & (df.publish_start_date < '2019-02-04')]
df = df[['publish_start_date']].sort_values(by='publish_start_date')
df.to_csv('article_id.csv')
