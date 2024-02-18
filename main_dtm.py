from modules.data_reader import DataReader
from modules.data_saver import DataSaver
from modules.text_prep import TextPreparation
from modules.evaluation import Evaluation
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.wrappers import DtmModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from nltk.tokenize import word_tokenize
import time


CONSTANTS = {
    "dataset":"UN_100w",
    "embedding_method":"DTM",
    "model":"DTM",
    "topic_method":"DTM"
}

def run_experiment():
    print("Reading Data")
    dr = DataReader()#filename="scientific_parallel.csv")
    text_data = dr.obtain_text_data()
    print(text_data.shape)
    
    print("Preparing Text")
    tp = TextPreparation(text_data.en)
    prepped_text = tp.prepare_text()

    spanish_tp = TextPreparation(text_data.es,language="spanish")
    spanish_prepped_text = spanish_tp.prepare_text(pipeline=["clean","lemmatize","filter"])
    spanish_prepped_text = spanish_prepped_text.loc[tp.indexes_to_keep]

    start = time.time()

    print("Generating Topics")
    dictionary_en = Dictionary([word_tokenize(s) for s in prepped_text.values])
    en_corpus = [dictionary_en.doc2bow(word_tokenize(text)) for text in prepped_text.values]

    dictionary_es = Dictionary([word_tokenize(s) for s in spanish_prepped_text.values])
    es_corpus = [dictionary_es.doc2bow(word_tokenize(text)) for text in spanish_prepped_text.values]

    path_dtm = "binaries/dtm-win64.exe"
    lda_en = DtmModel(path_dtm,corpus=en_corpus,id2word=dictionary_en, num_topics=20,time_slices=[1] * len(en_corpus),
                      lda_max_em_iter=5,lda_sequence_max_iter=10)
    print("Generated English Topics")
    top_tokens_scores = lda_en.show_topics(num_topics=20,num_words = 10) 
    top_tokens = list()
    for scores in top_tokens_scores:
        new_tokens = [t.split("*")[1] for t  in scores.split(" + ")]
        top_tokens.append(new_tokens)
    print(top_tokens)


    lda_es = DtmModel(path_dtm,corpus=es_corpus,id2word=dictionary_es, num_topics=20,time_slices=[1] * len(es_corpus),
                      lda_max_em_iter=5,lda_sequence_max_iter=10)
    spanish_top_tokens_scores = lda_es.show_topics(num_topics=20,num_words = 10) 
    spanish_top_tokens = list()
    for scores in spanish_top_tokens_scores:
        new_tokens = [t.split("*")[1] for t  in scores.split(" + ")]
        spanish_top_tokens.append(new_tokens)
    
    print(spanish_top_tokens)

    end = time.time()
    training_time = end-start
    print("Calculating Utilities")
    utils = Evaluation()
    utils.create_utility_objects(prepped_text)
    coherence = utils.get_coherence(top_tokens)

    spanish_utils = Evaluation()
    spanish_utils.create_utility_objects(spanish_prepped_text)
    spanish_coherence = spanish_utils.get_coherence(spanish_top_tokens)


    diversity= utils.get_topic_diversity(top_tokens)
    spanish_diversity= spanish_utils.get_topic_diversity(spanish_top_tokens)
    

    average_matching = 0

    cla = 0

    other_stats = utils.get_dataset_stats(prepped_text)


    final_object = {"coherence":coherence,"top_tokens":top_tokens,"spanish_coherence":spanish_coherence,"spanish_top_tokens":spanish_top_tokens,
                    "diversity":diversity,"spanish_diversity":spanish_diversity,"average_matching":average_matching,
                    "cla":cla,**CONSTANTS,**other_stats,"vocab_size":tp.vocab_size,"training_time":training_time}

    print("Saving...")
    data_saver = DataSaver()
    data_saver.save_object(final_object,"results_DTM.json")
    print("Done")


if __name__ == "__main__":
    run_experiment()




