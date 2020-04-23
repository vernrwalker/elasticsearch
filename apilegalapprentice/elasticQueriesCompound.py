from datetime import datetime
from elasticsearch import Elasticsearch


# https://www.elastic.co/guide/en/elasticsearch/reference/current/cat-indices.html
# https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html

# THIS IS A SET OF COMPOUND QUERIES FOR ELASTICSEARCH, FORMULATED AND TESTED WITH KIBANA (run on localhost:5601)

# CHANGE "queryX" to "query" to run just that query; otherwise, last "query" will be run

def elasticsearch_query():

    #  https://elasticsearch-py.readthedocs.io/en/master/
    # by default we connect to localhost:9200
    es = Elasticsearch()

    indexName = 'la-50attrib-cases'

    # USING "bool" TO RETURN ALL DOCUMENTS (SENTENCES) SATISFYING ALL OF MULTIPLE QUERIES (!!!note that these conditions can be satisfied anywhere in the document, not necessarily in the same attribution!!!)
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "match": {
                            "rhetClass": "FindingSentence"
                        }
                    },
                    {
                        "match_phrase": {
                            "attributions.object": "psychiatric disability"
                        }
                    },
                    {
                        "match": {
                            "attributions.type": "Finding"
                        }
                    }
                ]
            }
        }
    }

    

    res = es.search(index=indexName, body=query)
    print(res)

    #THESE TWO LINES PRINT THE PROPERTIES SPECIFIED FOR ALL THE HITS
    #for hit in res["hits"]["hits"]:
        #print(hit["_source"])

# https://www.elastic.co/guide/en/elasticsearch/reference/current/search-request-body.html

    print("Got %d Hits:" % res['hits']['total']['value'])
     
elasticsearch_query()