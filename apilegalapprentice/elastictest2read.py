from datetime import datetime
from elasticsearch import Elasticsearch


# https://www.elastic.co/guide/en/elasticsearch/reference/current/cat-indices.html
# https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html






def elasticsearch_query():

    #  https://elasticsearch-py.readthedocs.io/en/master/
    # by default we connect to localhost:9200
    es = Elasticsearch()

    indexName = 'la-50attrib-cases'

    
    #THIS QUERY COUNTS THE NUMBER OF OBJECTS IN A THE DATASET
    query = {
        "query": {"match_all": {}}
    }

    

    #THIS IS A QUERY OF A MATCH-SEARCH WITHIN A JSON PROPERTY; THIS SPECIFIC QUERY SHOULD RETURN EXACTLY ONE SENTENCE
    query = {
        "query": {
            "match": {
                "sentID": {
                     "query": "1302554P20S1"
                     }
                }
            }
    }

    #THIS IS A MATCH-SEARCH OF A STRING WITHIN A JSON PROPERTY
    query = {
        "query": {
            "match": {
                "rhetClass": {
                     "query": "FindingSentence"
                     }
                }
            }
    }
    
    #THIS IS A MATCH-SEARCH OF A SUB-PROPERTY WITHIN A PROPERTY OF THE JSON OBJECT
    query = {
        "query": {
            "match": {
                "attributions.polarity": {
                     "query": ""
                     }
                }
            }
    }

    #THIS IS A MATCH-SEARCH OF A SUB-PROPERTY WITHIN A PROPERTY OF THE JSON OBJECT, with operator "OR" as default (returns freq of docs with a hit)
    query = {
        "query": {
            "match": {
                "attributions.polarity": {
                     "query": "positive negative undecided"
                     }
                }
            }
    }

    #THIS QUERIES THE NUMBER OF OBJECTS THAT HAVE A PROPERTY, AND FILTERS FOR ONLY THOSE WITH ANOTHER PROPERTY (here, the "operator" parameter requires that all four words be present [but not in that order?])
    query = {
        "query": {
            "bool": {
                "must" : {
                    "match" : {
                        "rhetClass" : "FindingSentence"}
                },
                "filter" : {
                    "match" : {
                        "text" : {
                            "query" : "the Board finds that",
                            "operator" : "and"
                        }
                    }
                }
            }
        }
    }

    #THIS QUERIES THE NUMBER OF OBJECTS THAT HAVE A PROPERTY, AND FILTERS FOR ONLY THOSE WITH ANOTHER PROPERTY (here, the match_phrase query requires that all four words be present in that order[?])
    query = {
        "query": {
            "bool": {
                "must" : {
                    "match" : {
                        "rhetClass" : "FindingSentence"}
                },
                "filter" : {
                    "match_phrase" : {
                        "text" : {
                            "query" : "the Board finds that"
                        }
                    }
                }
            }
        }
    }

    

    res = es.search(index=indexName, body=query)
    ## print(res)

    #THESE TWO LINES PRINT THE PROPERTIES SPECIFIED FOR ALL THE HITS
    #for hit in res['hits']['hits']:
        #print(hit["_source"])

# https://www.elastic.co/guide/en/elasticsearch/reference/current/search-request-body.html

    print("Got %d Hits:" % res['hits']['total']['value'])
     
elasticsearch_query()