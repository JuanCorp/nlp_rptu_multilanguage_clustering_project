from modules.data_reader import DataReader
from modules.data_saver import DataSaver
from modules.text_prep import TextPreparation
from modules.evaluation import Evaluation
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize


CONSTANTS = {
    "dataset":"scientific_papers",
    "embedding_method":"LDA",
    "model":"LDA",
    "topic_method":"LDA"
}

def run_experiment():
    print("Reading Data")
    dr = DataReader(filename="scientific_parallel.csv")
    text_data = dr.obtain_text_data()
    print(text_data.shape)
    
    print("Preparing Text")
    tp = TextPreparation(text_data.en)
    prepped_text = tp.prepare_text()

    spanish_tp = TextPreparation(text_data.es,language="spanish")
    spanish_prepped_text = spanish_tp.prepare_text(pipeline=["clean","lemmatize","filter"])
    spanish_prepped_text = spanish_prepped_text.loc[tp.indexes_to_keep]

    full_text = prepped_text.append(spanish_prepped_text)


    print("Generating Topics")
    dictionary_en = Dictionary([word_tokenize(s) for s in prepped_text.values])
    en_corpus = [dictionary_en.doc2bow(word_tokenize(text)) for text in prepped_text.values]

    dictionary_es = Dictionary([word_tokenize(s) for s in spanish_prepped_text.values])
    es_corpus = [dictionary_es.doc2bow(word_tokenize(text)) for text in spanish_prepped_text.values]


    lda_en = LdaModel(en_corpus, num_topics=20,random_state=777)
    topics_list = [t_list for t_list in lda_en.get_document_topics(en_corpus,minimum_probability=0.01)]
    topics = list()
    for l in topics_list:
        doc_topics = [t[0] for t in l]
        topics.append(doc_topics)
    lda_es = LdaModel(es_corpus, num_topics=20,random_state=777)
    topics_list = [t_list for t_list in lda_es.get_document_topics(es_corpus,minimum_probability=0.01)]
    spanish_topics = list()
    for l in topics_list:
        doc_topics = [t[0] for t in l]
        spanish_topics.append(doc_topics)
    print("Calculating Utilities")
    utils = Evaluation()
    utils.create_utility_objects(prepped_text)
    top_tokens = utils.get_top_topic_tokens_lda(topics)
    coherence = utils.get_coherence(top_tokens)

    spanish_utils = Evaluation()
    spanish_utils.create_utility_objects(spanish_prepped_text)
    spanish_top_tokens = spanish_utils.get_top_topic_tokens_lda(spanish_topics)
    spanish_coherence = spanish_utils.get_coherence(spanish_top_tokens)


    diversity= utils.get_topic_diversity(top_tokens)
    spanish_diversity= spanish_utils.get_topic_diversity(spanish_top_tokens)
    

    average_matching = utils.average_topic_matching(topics,spanish_topics)

    cla = utils.get_cross_lingual_alignment(topics,spanish_topics,prepped_text.values,spanish_prepped_text.values,lda=True)

    other_stats = utils.get_dataset_stats(prepped_text)


    final_object = {"coherence":coherence,"top_tokens":top_tokens,"spanish_coherence":spanish_coherence,"spanish_top_tokens":spanish_top_tokens,
                    "diversity":diversity,"spanish_diversity":spanish_diversity,"average_matching":average_matching,
                    "cla":cla,**CONSTANTS,**other_stats,"vocab_size":tp.vocab_size}

    print("Saving...")
    data_saver = DataSaver()
    data_saver.save_object(final_object,"results_LDA_scientific.json")
    print("Done")


if __name__ == "__main__":
    run_experiment()




