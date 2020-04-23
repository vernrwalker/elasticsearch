from datetime import datetime
from elasticsearch import Elasticsearch


# https://www.elastic.co/guide/en/elasticsearch/reference/current/cat-indices.html
# https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html


# THIS IS A BASIC SET OF QUERIES FOR ELASTICSEARCH, FORMULATED AND TESTED WITH KIBANA (run on localhost:5601)

# CHANGE "queryX" to "query" to run just that query; otherwise, last "query" will be run

def elasticsearch_query():

    #  https://elasticsearch-py.readthedocs.io/en/master/
    # by default we connect to localhost:9200
    es = Elasticsearch()

    indexName = 'la-50attrib-cases'

    # USING "match_all" TO COUNT THE NUMBER OF DOCUMENTS IN THE DATASET
    query1 = {
        "query": {
            "match_all": {}
        }
    }


    # USING "match" TO RETURN A SPECIFIC DOCUMENT USING AN INDEX FIELD (BUILT FROM A JSON PROPERTY); this specific query should return only one sentence
    query2 = {
        "query": {
            "match": {
                "sentID": "1302554P20S1"
            }
        }
    }

    # USING "match" TO RETURN ALL DOCUMENTS (SENTENCES) WITHIN THE SAME CASE
    query3 = {
        "query": {
            "match": {
                "caseNumber": "1400029"
            }
        }
    }

    # USING "match" TO RETURN ALL DOCUMENTS CONTAINING A SPECIFIC VALUE WITHIN AN INDEX FIELD (BUILT FROM A JSON PROPERTY); this specific query should return all FindingSentences
    query4 = {
        "query": {
            "match": {
                "rhetClass": "FindingSentence"
            }
        }
    }
    
    # USING "match" TO RETURN ALL DOCUMENTS (SENTENCES) CONTAINING THE SAME VALUE WITHIN A SUB-FIELD OF AN INDEX FIELD, with operator "OR" as default (returns the count of documents with at least one hit)
    query5 = {
        "query": {
            "match": {
                "attributions.type": "Finding"                    
            }
        }
    }

    # USING "match" TO RETURN ALL DOCUMENTS (SENTENCES) CONTAINING THE SAME VALUE WITHIN A SUB-FIELD OF AN INDEX FIELD, with operator "OR" as default (returns the count of documents with at least one hit)
    query6 = {
        "query": {
            "match": {
                "attributions.polarity": "positive negative undecided"                    
            }
        }
    }

    # USING "match_phrase" TO RETURN ALL DOCUMENTS (SENTENCES) CONTAINING THE SAME PRECISE STRING WITHIN AN INDEX FIELD
    query7 = {
        "query": {
            "match_phrase": {
                "text": "psychiatric disability"                
            }
        }
    }

    # USING "match_phrase" TO RETURN ALL DOCUMENTS (SENTENCES) CONTAINING THE SAME PRECISE STRING WITHIN AN INDEX FIELD
    query8 = {
        "query": {
            "match_phrase": {
                "attributions.object": "psychiatric disability"                
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