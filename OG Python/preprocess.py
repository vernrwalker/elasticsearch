import json

#Append all sentences and rhetroles in json files to a dataframe
def preprocess(ListOfNames):
    sentences, rhetrole = [], []
    for file in ListOfNames:
        fh = open(file)
        data = json.load(fh)

        for sent in data['sentences']:
            sentences.append(sent['text'])
            rhetrole.append(sent['rhetRole'][0])

    data = pd.DataFrame({
        'sentences': sentences,
        'rhetrole': rhetrole
    })
    return data
