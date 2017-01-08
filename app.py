from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from os import listdir, getcwd
from os.path import isfile, join, abspath, dirname
import os.path
import re
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from flask import Flask, render_template, request, url_for
import pandas
import json


def lsa_analysis(dataset):
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=1000, min_df=2, stop_words='english', use_idf=True)
    X = vectorizer.fit_transform(dataset)
    n_components = 100
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    lsaX = lsa.fit_transform(X)
    explained_variance = svd.explained_variance_ratio_.sum()
    print explained_variance
    terms = vectorizer.get_feature_names()

    n_clusters = 5
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1)
    km.fit(lsaX)

    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]

    counter = Counter(km.labels_)
    clusters = defaultdict(list)
    for i in range(n_clusters):
        cluster = {}
        #original_space_centroids[0][127]
        cluster['size'] = float(counter[i])/len(km.labels_)
        '''word = defaultdict(float)
        for ind in order_centroids[i, :10]:
            word[terms[ind].encode('ascii','ignore')] = original_space_centroids[i][ind]'''
        words = []
        for ind in order_centroids[i, :50]:
            words.append({terms[ind].encode('ascii','ignore') : original_space_centroids[i][ind]})
        cluster['words'] = words
        clusters[i] = cluster
    return clusters

def filter_data(dataset, gender, agegroup):
    count = 0
    filtered_data = []
    for user in dataset.keys():
        user_id, user_gender, age, industry, zodiac = user.split('.')[:5]
        is_in_agegroup = check_agegroup(age, agegroup)
        if(gender == ''):
            user_gender = ''
        if(agegroup == ''):
            is_in_agegroup = True
        if(gender == user_gender and is_in_agegroup):
            count = count + 1
            filtered_data.append(dataset[user])
    print count
    #if(count > 100):
    #   filtered_data[:100]
    return filtered_data

def check_agegroup(age, agegroup):
    lowerbound, upperbound = agegroup.split('-')
    age = int(age)
    if (age > int(lowerbound) and age <= int(upperbound)):
        return True;
    else:
        return False;

def make_json(dataset, gender, agegroup, jsonList):
    data = filter_data(dataset, gender, agegroup)
    X = lsa_analysis(data)
    obj = {}
    obj['age'] = agegroup
    obj['gender'] = gender
    obj['clusters'] = X
    jsonList.append(obj);


files = [f for f in listdir('blogs') if isfile(join('blogs', f))]
dataset = {}

for i,file in enumerate(files):
    with open('blogs/' + file, 'r') as f:
        if(i < 0):
            continue
        if(i > 10000):
            break
        read_data = f.read()
        pattern = r'<post>(.*?)</post>'
        posts = re.findall(pattern, read_data, re.DOTALL)
        dataset[file] = unicode(",".join(posts), errors='ignore').encode('ascii','ignore')
        # dataset.append(unicode(",".join(posts), errors='ignore').encode('ascii','ignore'))
        f.closed


jsonList = []
make_json(dataset, "male", "10-20",jsonList)
make_json(dataset, "male", "20-30",jsonList)
make_json(dataset, "male", "30-40",jsonList)
make_json(dataset, "male", "40-50",jsonList)
#make_json(dataset, "male", "50-60",jsonList)
#make_json(dataset, "male", "60-70",jsonList)
# make_json(dataset, "male", "40-50",json)
make_json(dataset, "female", "10-20",jsonList)
make_json(dataset, "female", "20-30",jsonList)
make_json(dataset, "female", "30-40",jsonList)
make_json(dataset, "female", "40-50",jsonList)
#make_json(dataset, "female", "50-60",jsonList)
#make_json(dataset, "female", "60-70",jsonList)
# make_json(dataset, "female", "40-50",json)

print "done"

# flask setup
project_root = os.path.dirname(__file__)
template_path = os.path.join(project_root, 'Templates')

app = Flask(__name__, template_folder=template_path )

@app.route("/returnjson")
def returnjson():
    return json.dumps(jsonList)
    # return pandas.json.dumps(pandas.DataFrame(jsonList));

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=14377,debug=True)
