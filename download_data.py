import pymongo
import pandas as pd
from secret_info import MONGO_HOST, MONGO_ID, MONGO_PW, MONGO_DB_NAME


class DataDownloader:
    def __init__(self):
        super(DataDownloader, self).__init__()
        self.db_client = pymongo.MongoClient('mongodb://%s:%s@%s' % (MONGO_ID, MONGO_PW, MONGO_HOST))
        self.db = self.db_client['BWAI']
        self.posts_labeled_v3 = self.db.posts_labeled_v3

        self.test_ratio = 0.25

    def download_all_data(self):
        data = []
        cursor = self.posts_labeled_v3.find({})
        for i, document in enumerate(cursor):
            if i % 10000 == 0:
                print("{}/{} sentences".format(len(data), i))

            if document['label'] == -1:
                continue

            doc = {'document': document['string'],
                   'label': int(document['label'])}
            data.append(doc)

        df = pd.DataFrame(data, columns=['document', 'label'])
        df.index.name = '_id'
        df.to_csv("badword/ratings.csv")

        df_shuffled = df.sample(frac=1)
        n_train = int(len(data)*(1-self.test_ratio))
        n_test = len(data)-n_train
        df_train, df_test = df_shuffled[:n_train], df_shuffled[n_train:]

        df_train.to_csv("badword/ratings_train.csv")
        df_test.to_csv("badword/ratings_test.csv")

        print("{}/{} sentences\t{} train\t{} test".format(len(data), i, n_train, n_test))

    def download_labeled_data(self):
        data = []
        cursor = self.posts_labeled_v3.find({})
        for i, document in enumerate(cursor):
            if i % 10000 == 0:
                print("{}/{} sentences".format(len(data), i))

            if document['labeling_confirm'] == 0 or document['label'] == -1:
                continue

            doc = {'document': document['string'],
                   'label': int(document['label'])}
            data.append(doc)

        df = pd.DataFrame(data, columns=['document', 'label'])
        df.index.name = '_id'
        df.to_csv("/opt/project/badword/ratings_labeled.csv")

        df_shuffled = df.sample(frac=1)
        n_train = int(len(data)*(1-self.test_ratio))
        n_test = len(data)-n_train
        df_train, df_test = df_shuffled[:n_train], df_shuffled[n_train:]

        df_train.to_csv("/opt/project/badword/ratings_labeled_train.csv")
        df_test.to_csv("/opt/project/badword/ratings_labeled_test.csv")

        print("{}/{} sentences\t{} train\t{} test".format(len(data), i, n_train, n_test))


if __name__ == '__main__':
    downloader = DataDownloader()
    # downloader.download_all_data()
    downloader.download_labeled_data()
